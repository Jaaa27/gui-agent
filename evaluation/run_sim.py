"""Run the ideal-perception simulation and write the results as a CSV in the same format as the real-machine evaluation.

    python -m evaluation.run_sim --out results/sim_runs.csv
    python -m evaluation.run_sim --ext --out results/sim_ext.csv     the 4 week-8 extended-action tasks, before vs after
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
    ap.add_argument("--ext", action="store_true", help="run the extended-action tasks (with and without hotkey/scroll/wait)")
    a = ap.parse_args()
    logging.getLogger("gui_agent").disabled = True       # no logging needed in the simulation

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
        print(f"{cond}: {sum(r['final_ok'] for r in rs)}/{len(rs)} passed")
    print("written to", a.out)


if __name__ == "__main__":
    main()
