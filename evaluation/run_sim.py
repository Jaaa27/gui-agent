"""跑理想感知条件下的仿真，把结果写成和真机评估相同格式的 CSV。

    python -m evaluation.run_sim --out results/sim_runs.csv
    python -m evaluation.run_sim --ext --out results/sim_ext.csv     第 8 周扩展动作的 4 个任务，前后对比
"""
import argparse
import csv
import logging
import os

from .record import COLUMNS
from .sim import run_sim
from .tasks import EXT_TASKS, TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sim_runs.csv")
    ap.add_argument("--ext", action="store_true", help="跑扩展动作任务（有/没有按键滚动等待两种条件）")
    a = ap.parse_args()
    logging.getLogger("gui_agent").disabled = True       # 仿真不需要刷日志

    if a.ext:
        rows = run_sim(EXT_TASKS, True, direct_actions=False) + run_sim(EXT_TASKS, True)
    else:
        rows = run_sim(TASKS, True) + run_sim(TASKS, False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        print(f"{cond}: {sum(r['final_ok'] for r in rs)}/{len(rs)} 通过")
    print("已写入", a.out)


if __name__ == "__main__":
    main()
