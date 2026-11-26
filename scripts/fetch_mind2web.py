"""Stream Mind2Web from Hugging Face and save the first N tasks as data/dataset_sample_100.json.

streaming=True avoids downloading tens of GB. Only the task instruction, website domain and number of actions
are kept; samples with an empty instruction or no action sequence are skipped. Requires pip install datasets.
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
    print(f"saved {len(samples)} samples (skipped {skipped} invalid) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", default="data/dataset_sample_100.json")
    a = ap.parse_args()
    fetch(num_samples=a.num, out=a.out)
