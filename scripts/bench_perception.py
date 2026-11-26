"""Benchmark the perception module's speed and missed text on your own machine.

    python -m scripts.bench_perception --repeat 5
    python -m scripts.bench_perception --idle-check     # first check the verification threshold on an idle screen

Scans the current screen repeatedly and compares full-resolution OCR with downscaled OCR
(max_side = none / auto / 1600 / 1280 / 1024), then measures a cache hit when the screen is unchanged.
Recall uses the full-resolution texts as the reference and counts how many survive downscaling.
Arrange the screen as you normally use it and keep it still. Results go to results/perception_bench.json.
"""
import argparse
import json
import os
import statistics
import time

from gui_agent.matcher import normalize
from gui_agent.perceiver import ScreenPerceiver


def timed(fn, repeat):
    ts, last = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        last = fn()
        ts.append(time.perf_counter() - t0)
    return ts, last


def texts_of(elements):
    return {normalize(v["text"]) for v in elements.values() if normalize(v["text"])}


def recall(ref, got):
    if not ref:
        return None
    hit = sum(1 for r in ref if any(r == g or r in g or g in r for g in got))
    return round(100.0 * hit / len(ref), 1)


def idle_check(n=20, gap=0.5):
    """Take n signatures of an idle screen and count cells whose difference between neighbouring frames exceeds cell_delta.
    The result should be well below min_cells (default 3); otherwise this machine is noisier than the synthetic data
    and the threshold should be raised."""
    p = ScreenPerceiver(reader=object())          # signature only, no OCR needed
    prev, worst = p.capture_signature(), 0
    for _ in range(n):
        time.sleep(gap)
        cur = p.capture_signature()
        worst = max(worst, int((abs(cur - prev) > p.cell_delta).sum()))
        prev = cur
    print(f"idle screen, {n} frames: at most {worst} cells changed by more than {p.cell_delta} grey levels "
          f"(min_cells={p.min_cells})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default="results/perception_bench.json")
    ap.add_argument("--idle-check", action="store_true", help="only check signature noise on an idle screen; no OCR")
    a = ap.parse_args()
    if a.idle_check:
        idle_check()
        return

    rows = []
    base = ScreenPerceiver(max_side=0, cache=False)       # no downscaling - the reference
    base.scan_screen()                                    # warm-up run; the first call initializes CUDA
    ts, (_, ref_els) = timed(base.scan_screen, a.repeat)
    ref = texts_of(ref_els)
    rows.append({"config": "full resolution", "avg_s": round(statistics.mean(ts), 3),
                 "elements": len(ref_els), "recall": 100.0})

    for ms in ("auto", 1600, 1280, 1024):
        p = ScreenPerceiver(max_side=ms, cache=False)
        p.scan_screen()
        ts, (_, els) = timed(p.scan_screen, a.repeat)
        rows.append({"config": f"max_side={ms}", "avg_s": round(statistics.mean(ts), 3),
                     "elements": len(els), "recall": recall(ref, texts_of(els))})

    p = ScreenPerceiver(max_side=1600, cache=True)
    p.scan_screen()
    ts, _ = timed(p.scan_screen, a.repeat)
    rows.append({"config": "max_side=1600 + cache hit", "avg_s": round(statistics.mean(ts), 3),
                 "elements": "-", "recall": "-"})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"repeat": a.repeat, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"\n{'config':<30}{'mean time (s)':>14}{'elements':>10}{'recall (%)':>12}")
    for r in rows:
        print(f"{r['config']:<30}{r['avg_s']:>14}{str(r['elements']):>10}{str(r['recall']):>12}")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
