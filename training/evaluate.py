"""微调前后、提示词前后的对比评估。同一个基座，同一批数据，跑几种配置：

  base_v1_legacy  基座 + 原提示词，不传合法 id 列表（第 4 周旧测试脚本的调用方式）
  base_v1         基座 + 原提示词，传合法 id 列表（agent.py 实际运行时的方式）
  base_v2         基座 + 优化后的提示词
  ft_ft           基座 + LoRA + 微调专用提示词
  ft_v2           LoRA + v2 提示词，看微调模型换提示词后稳不稳

评测集两个：test（合成的，指令和训练集零重叠）、hard50（第 4 周的固定 50 题）。

    python -m training.evaluate --data data --adapter outputs/lora_gui --out results
    python -m training.evaluate --data data --backend mock --out results_mock --limit 60   # 无 GPU 自测，结果不能写进报告
"""
import argparse
import json
import os
import random
import time

from training.common import aggregate, build_messages, load_jsonl, parse_decision, score_case

CONFIGS = {
    "base_v1_legacy": dict(adapter=False, prompt="v1", valid_ids=False),
    "base_v1": dict(adapter=False, prompt="v1", valid_ids=True),
    "base_v2": dict(adapter=False, prompt="v2", valid_ids=True),
    "ft_ft": dict(adapter=True, prompt="ft", valid_ids=True),
    "ft_v2": dict(adapter=True, prompt="v2", valid_ids=True),
}
DEFAULT_CONFIGS = ["base_v1_legacy", "base_v1", "base_v2", "ft_ft"]


class HFBackend:
    def __init__(self, base, adapter, max_new_tokens=128, batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
            torch.float16 if use_cuda else torch.float32)
        tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else base
        self.tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, trust_remote_code=True,
                                                     device_map={"": 0} if use_cuda else None)
        self.has_adapter = bool(adapter)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model

    def generate(self, messages_list, use_adapter):
        """返回 [(text, latency_per_sample)]"""
        torch = self.torch
        outs = []
        for i in range(0, len(messages_list), self.batch_size):
            chunk = messages_list[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            t0 = time.time()
            with torch.no_grad():
                if self.has_adapter and not use_adapter:
                    with self.model.disable_adapter():
                        gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                                  pad_token_id=self.tok.pad_token_id)
                else:
                    gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                              pad_token_id=self.tok.pad_token_id)
            dt = (time.time() - t0) / len(chunk)
            new = gen[:, enc["input_ids"].shape[1]:]
            for s in self.tok.batch_decode(new, skip_special_tokens=True):
                outs.append((s, dt))
        return outs


class MockBackend:
    """只用来自测流水线：按配置随机给不同的正确率，不代表任何真实模型的能力。"""
    P = {"base_v1_legacy": 0.3, "base_v1": 0.5, "base_v2": 0.65, "ft_ft": 0.95, "ft_v2": 0.9}

    def __init__(self):
        self.rng = random.Random(0)
        self.cfg = None

    def generate(self, messages_list, use_adapter, recs=None, cfg_name=None):
        outs = []
        p = self.P.get(cfg_name, 0.5)
        for rec in recs:
            t = rec["target"]
            if self.rng.random() < p:
                d = {"action": t["action"], "target_id": t["target_id"]}
                if t["action"] == "type":
                    d["text"] = t.get("text") or ""
                s = "```json\n" + json.dumps(d, ensure_ascii=False) + "\n```"
            elif self.rng.random() < 0.3:
                s = "抱歉，我无法确定。"
            else:
                s = json.dumps({"action": "click", "target_id": 9999}, ensure_ascii=False)
            outs.append((s, 0.01))
        return outs


def run_config(backend, cfg_name, recs, mock):
    cfg = CONFIGS[cfg_name]
    msgs = [build_messages(r, prompt=cfg["prompt"], include_valid_ids=cfg["valid_ids"]) for r in recs]
    if mock:
        outs = backend.generate(msgs, cfg["adapter"], recs=recs, cfg_name=cfg_name)
    else:
        outs = backend.generate(msgs, cfg["adapter"])
    rows = []
    for r, (raw, lat) in zip(recs, outs):
        dec = parse_decision(raw)
        rows.append({
            "id": r["id"], "kind": r.get("kind"), "instruction": r["instruction"],
            "expected": r["target"], "expected_action": r["target"]["action"],
            "raw": raw, "decision": dec, "score": score_case(dec, r), "latency": lat,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录（含 adapter_model.safetensors）")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="每个评测集最多评多少条（调试用）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    a = ap.parse_args()

    mock = a.backend == "mock"
    cfgs = a.configs or (DEFAULT_CONFIGS if not mock else DEFAULT_CONFIGS + ["ft_v2"])
    if not mock and not a.adapter:
        cfgs = [c for c in cfgs if not CONFIGS[c]["adapter"]]
        print("[提示] 未提供 --adapter，只评估基座配置。")
    os.makedirs(a.out, exist_ok=True)

    datasets = {"test": load_jsonl(os.path.join(a.data, "test.jsonl")),
                "hard50": load_jsonl(os.path.join(a.data, "hard50.jsonl"))}
    if a.limit:
        datasets = {k: v[:a.limit] for k, v in datasets.items()}

    backend = MockBackend() if mock else HFBackend(a.base, a.adapter, a.max_new_tokens, a.batch)
    summary = {"mock": mock, "base": a.base, "adapter": a.adapter, "configs": {},
               "dataset_sizes": {k: len(v) for k, v in datasets.items()}}

    for cfg_name in cfgs:
        summary["configs"][cfg_name] = {}
        for ds_name, recs in datasets.items():
            print(f"[eval] {cfg_name} on {ds_name} ({len(recs)} 条)...")
            rows = run_config(backend, cfg_name, recs, mock)
            metrics = aggregate(rows)
            summary["configs"][cfg_name][ds_name] = metrics
            with open(os.path.join(a.out, f"{cfg_name}__{ds_name}.json"), "w", encoding="utf-8") as f:
                json.dump({"config": cfg_name, "dataset": ds_name, "mock": mock,
                           "metrics": metrics, "records": rows}, f, ensure_ascii=False, indent=1)
            o = metrics["overall"]
            print(f"       exact={o['exact']}%  json={o['json_valid']}%  id_halluc={o['id_hallucination']}%  "
                  f"type_exact={metrics['type'].get('exact')}%  lat={o['avg_latency_s']}s")

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"评估完成，结果在 {a.out}/summary.json" + ("  （MOCK 自测数据）" if mock else ""))


if __name__ == "__main__":
    main()
