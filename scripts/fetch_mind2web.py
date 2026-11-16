"""从 Hugging Face 流式读 Mind2Web，抽前 N 条整理成 data/dataset_sample_100.json。

streaming=True 不用把几十 GB 的数据整个下下来。只留任务指令、网站领域和动作数量，
指令为空或没有动作序列的当脏数据跳过。需要 pip install datasets。
"""
import argparse
import json
import os

from datasets import load_dataset


def fetch(dataset_name="osunlp/Mind2Web", split="train", num_samples=100, out="data/dataset_sample_100.json"):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    samples, skipped = [], 0
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        instruction = (item.get("confirmed_task") or "").strip()
        actions = item.get("action_reprs") or []
        if not instruction or not actions:
            skipped += 1
            continue
        samples.append({
            "task_id": f"mind2web_task_{i}",
            "website": item.get("domain", ""),
            "user_instruction": instruction,
            "total_actions_required": len(actions),
        })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
    print(f"保存 {len(samples)} 条（跳过 {skipped} 条脏数据）-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", default="data/dataset_sample_100.json")
    a = ap.parse_args()
    fetch(num_samples=a.num, out=a.out)
