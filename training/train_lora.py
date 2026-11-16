"""用 PEFT 给 GUI 决策任务做 LoRA 微调。

输入是 system（合法 id）+ user（OCR 元素字典和指令），输出是一段 JSON 动作。
loss 只算 assistant 回答的部分，prompt 部分的 label 设成 -100。

默认基座 Qwen2.5-1.5B-Instruct，Colab T4 能直接跑，显存紧就加 --load_4bit。

    python -m training.train_lora --data data --out outputs/lora_gui
    python -m training.train_lora --data data --out outputs/lora_gui --load_4bit --merge
"""
import argparse
import glob
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from training.common import build_messages, load_jsonl


class SFTDataset(Dataset):
    def __init__(self, records, tok, max_len):
        self.items = []
        self.dropped = 0
        for rec in records:
            msgs = build_messages(rec, prompt="ft", with_answer=True)
            prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            answer_text = msgs[-1]["content"] + tok.eos_token
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            f_ids = tok(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len:
                self.dropped += 1
                continue
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            self.items.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/lora_gui")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA：4bit 量化加载基座模型（需 bitsandbytes）")
    ap.add_argument("--merge", action="store_true", help="训练后额外导出合并后的完整模型（供 Ollama/GGUF 转换）")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else (torch.float16 if use_cuda else torch.float32)
    print(f"[env] cuda={use_cuda} dtype={dtype} base={a.base}")
    if not use_cuda:
        print("[警告] 没检测到 GPU，训练会很慢，建议用 Colab 的 T4。")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(a.base, **kwargs)
    model.config.use_cache = False

    if a.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for p in model.parameters():          # fp16 训练要求可训练参数是 fp32
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    train_recs = load_jsonl(os.path.join(a.data, "train.jsonl"))
    val_recs = load_jsonl(os.path.join(a.data, "val.jsonl"))
    train_ds = SFTDataset(train_recs, tok, a.max_len)
    val_ds = SFTDataset(val_recs, tok, a.max_len)
    print(f"[data] train={len(train_ds)} (丢弃超长 {train_ds.dropped}) val={len(val_ds)} (丢弃 {val_ds.dropped})")

    steps_per_epoch = math.ceil(len(train_ds) / (a.batch * a.grad_accum))
    warmup_steps = max(1, int(0.05 * steps_per_epoch * a.epochs))
    print(f"[train] steps/epoch={steps_per_epoch} total={int(steps_per_epoch * a.epochs)} warmup={warmup_steps}")

    args = TrainingArguments(
        output_dir=os.path.join(a.out, "checkpoints"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16_ok,
        fp16=(use_cuda and not bf16_ok),
        gradient_checkpointing=False,     # 前面已经手动开了
        optim="paged_adamw_8bit" if a.load_4bit else "adamw_torch",
        report_to="none",
        seed=a.seed,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
    ckpts = sorted(glob.glob(os.path.join(a.out, "checkpoints", "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if ckpts:
        print(f"[resume] 从 {ckpts[-1]} 继续训练")
    result = trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)

    trainer.save_model(a.out)             # 只存 LoRA 适配器，几十 MB
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "train_metrics": result.metrics, "log_history": trainer.state.log_history},
                  f, ensure_ascii=False, indent=2)
    print(f"LoRA 适配器已保存到 {a.out}")

    if a.merge:
        if a.load_4bit:
            print("[提示] 4bit 加载的模型没法直接合并，去掉 --load_4bit 重新加载基座再 merge_and_unload。")
        else:
            merged = model.merge_and_unload()
            merged_dir = a.out.rstrip("/") + "_merged"
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(merged_dir)
            print(f"合并后的完整模型已保存到 {merged_dir}")


if __name__ == "__main__":
    main()
