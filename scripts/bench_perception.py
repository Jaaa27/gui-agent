"""在自己的电脑上测感知模块的速度和漏识情况。

    python -m scripts.bench_perception --repeat 5
    python -m scripts.bench_perception --idle-check     # 先在静止的屏幕上核对验证阈值

对当前屏幕连续扫描，比较整屏识别和缩放识别（max_side = 不缩放/auto/1600/1280/1024），
再测一次画面没变时命中缓存的情况。召回率是拿整屏识别出的文字当参照，看缩放后还剩多少。
测的时候把屏幕摆成平时用的样子并保持不动，结果写到 results/perception_bench.json。
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
    """屏幕保持不动，连续取 n 帧指纹，看相邻两帧有几个格子超过 cell_delta。
    结果应该远小于 min_cells（默认 3），否则说明这台机器上的屏幕噪声比合成数据大，要调高阈值。"""
    p = ScreenPerceiver(reader=object())          # 只用指纹，不需要 OCR
    prev, worst = p.capture_signature(), 0
    for _ in range(n):
        time.sleep(gap)
        cur = p.capture_signature()
        worst = max(worst, int((abs(cur - prev) > p.cell_delta).sum()))
        prev = cur
    print(f"静止屏幕 {n} 帧，相邻帧最多 {worst} 个格子超过 {p.cell_delta} 灰度级"
          f"（min_cells={p.min_cells}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default="results/perception_bench.json")
    ap.add_argument("--idle-check", action="store_true", help="只检查静止屏幕的指纹噪声，不跑 OCR")
    a = ap.parse_args()
    if a.idle_check:
        idle_check()
        return

    rows = []
    base = ScreenPerceiver(max_side=0, cache=False)       # 不缩放，当参照
    base.scan_screen()                                    # 先跑一次预热，第一次会初始化 CUDA
    ts, (_, ref_els) = timed(base.scan_screen, a.repeat)
    ref = texts_of(ref_els)
    rows.append({"config": "整屏(不缩放)", "avg_s": round(statistics.mean(ts), 3),
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
    rows.append({"config": "max_side=1600 + 命中缓存", "avg_s": round(statistics.mean(ts), 3),
                 "elements": "-", "recall": "-"})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"repeat": a.repeat, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"\n{'配置':<30}{'平均耗时(s)':>12}{'元素数':>8}{'召回率(%)':>11}")
    for r in rows:
        print(f"{r['config']:<30}{r['avg_s']:>12}{str(r['elements']):>8}{str(r['recall']):>11}")
    print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
