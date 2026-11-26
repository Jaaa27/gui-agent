"""How the verification threshold was chosen: measure signature differences for various "screen changes" on synthetic screens.

    python -m scripts.study_threshold

Method: draw a fake 1920x1080 desktop (gradient wallpaper + a browser window + text) and apply two kinds of
changes, judging before/after with the same signature as ScreenPerceiver (64x36 grayscale, INTER_AREA):
  noise - things that should NOT count as "the screen changed" after a click: the mouse cursor moving,
          the text caret blinking, the clock ticking one minute
  real  - what appears when a click/type actually worked: a popup menu, typing into a search box,
          sending a message, a dialog, a full page switch
Each change is repeated N times with random position/content. Two criteria are compared:
  old: whole-image mean grey difference mean(|a-b|)/255 > 0.01
  new: number of cells whose grey difference exceeds cell_delta >= min_cells (current)
Then (cell_delta, min_cells) is swept, choosing the smallest min_cells with zero false triggers on noise.
Synthetic screens are not real ones: real machines have more noise (video, animations, ads), so this only
sets the initial values; check on a real machine with python -m scripts.bench_perception --idle-check.

The change names are Chinese because they are the keys of results/week6/threshold_study.json, which the
(Chinese) final-report figures read: 光标挪动 cursor move, 输入光标闪烁 caret blink, 时钟走一分钟 clock tick,
弹出菜单 popup menu, 发出一条消息 send message, 弹对话框 dialog, 整页切换 page switch,
搜索框打字N字符 typing N characters into the search box.
"""
import json
import os
import random
import sys

import cv2
import numpy as np

SIG = (64, 36)
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX


def signature(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, SIG, interpolation=cv2.INTER_AREA).astype(np.float32)


def diff(a, b):
    return float(np.mean(np.abs(signature(a) - signature(b))) / 255.0)


def base_screen(rng):
    """Gradient wallpaper + a browser window with a title bar, address bar, search box and some lines of text."""
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    c0, c1 = np.array([120, 90, 40], np.float32), np.array([200, 160, 90], np.float32)
    img = (c0 * (1 - x) + c1 * x) * (1 - 0.3 * y)
    img = np.broadcast_to(img, (H, W, 3)).astype(np.uint8).copy()
    cv2.rectangle(img, (160, 60), (1760, 1000), (250, 250, 250), -1)          # window
    cv2.rectangle(img, (160, 60), (1760, 110), (225, 225, 225), -1)           # title bar
    cv2.rectangle(img, (160, 110), (1760, 170), (240, 240, 240), -1)          # address bar
    cv2.rectangle(img, (500, 400), (1300, 440), (255, 255, 255), -1)          # search box
    cv2.rectangle(img, (500, 400), (1300, 440), (150, 150, 150), 1)
    for i in range(14):
        n = rng.randint(20, 60)
        cv2.putText(img, "lorem ipsum " * 5 + str(i), (200, 230 + i * 8 * 4 % 150 + i * 30),
                    FONT, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        _ = n
    return img


def put_text(img, s, org, scale=0.7):
    cv2.putText(img, s, org, FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)


# ---- noise ----
def cursor_move(img, rng):
    a, b = img.copy(), img.copy()
    for im in (a, b):
        px, py = rng.randint(100, W - 100), rng.randint(100, H - 100)
        pts = np.array([[px, py], [px, py + 19], [px + 5, py + 15], [px + 12, py + 15]], np.int32)
        cv2.fillPoly(im, [pts], (255, 255, 255))
        cv2.polylines(im, [pts], True, (0, 0, 0), 1)
    return a, b


def caret_blink(img, rng):
    b = img.copy()
    x = rng.randint(520, 1200)
    cv2.line(b, (x, 405), (x, 435), (0, 0, 0), 2)
    return img, b


def clock_tick(img, rng):
    a, b = img.copy(), img.copy()
    for im, t in ((a, "10:41"), (b, "10:42")):
        cv2.rectangle(im, (1800, 1050), (1900, 1075), (30, 30, 30), -1)
        cv2.putText(im, t, (1810, 1070), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return a, b


# ---- real changes ----
def popup_menu(img, rng):
    b = img.copy()
    x, y = rng.randint(200, 1400), rng.randint(200, 600)
    cv2.rectangle(b, (x, y), (x + 260, y + 300), (255, 255, 255), -1)
    cv2.rectangle(b, (x, y), (x + 260, y + 300), (120, 120, 120), 1)
    for i in range(8):
        put_text(b, "menu item %d" % i, (x + 14, y + 30 + i * 34), 0.6)
    return img, b


def type_in_box(img, rng):
    b = img.copy()
    n = rng.randint(4, 24)                         # 4-24 characters typed
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(n))
    put_text(b, s, (510, 428), 0.7)
    return img, b


def send_message(img, rng):
    b = img.copy()
    put_text(b, "ok see you at the lab tomorrow morning " + str(rng.randint(0, 99)), (200, 800 + rng.randint(0, 60)), 0.7)
    return img, b


def dialog(img, rng):
    b = img.copy()
    cv2.rectangle(b, (700, 350), (1200, 650), (255, 255, 255), -1)
    cv2.rectangle(b, (700, 350), (1200, 650), (90, 90, 90), 2)
    put_text(b, "Do you want to save changes?", (730, 450), 0.8)
    return img, b


def page_switch(img, rng):
    b = img.copy()
    cv2.rectangle(b, (160, 170), (1760, 1000), tuple(int(v) for v in np.array([rng.randint(20, 235)] * 3)), -1)
    for i in range(10):
        put_text(b, "another page body text line %d" % i, (200, 230 + i * 50), 0.8)
    return img, b


NOISE = {"光标挪动": cursor_move, "输入光标闪烁": caret_blink, "时钟走一分钟": clock_tick}


def typing(n):
    def f(img, rng):
        b = img.copy()
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        put_text(b, s, (510, 428), 0.7)
        return img, b
    return f


# Typing without pressing Enter; Enter usually leads to a results page (page switch)
REAL = {"弹出菜单": popup_menu, "发出一条消息": send_message, "弹对话框": dialog, "整页切换": page_switch,
        "搜索框打字16字符": typing(16), "搜索框打字8字符": typing(8), "搜索框打字4字符": typing(4)}


def cells_changed(a, b, delta):
    return int(np.sum(np.abs(signature(a) - signature(b)) > delta))


def collect(n=200, seed=7):
    """Generate n before/after pairs per change. Returns {name: (kind, old-criterion values, {delta: changed-cell counts})}."""
    rng = random.Random(seed)
    out = {}
    for kind, group in (("noise", NOISE), ("real", REAL)):
        for name, fn in group.items():
            old, new = [], {d: [] for d in DELTAS}
            for _ in range(n):
                a, b = fn(base_screen(rng), rng)
                old.append(diff(a, b))
                for d in DELTAS:
                    new[d].append(cells_changed(a, b, d))
            out[name] = (kind, np.array(old), {d: np.array(v) for d, v in new.items()})
    return out


DELTAS = (6, 12, 24)
OLD_THR = 0.01


def main():
    data = collect()
    res = {"old": {}, "new": {}, "sweep": {}}
    print("old criterion mean>0.01:")
    for name, (kind, old, _) in data.items():
        rate = float(np.mean(old > OLD_THR))
        res["old"][name] = {"kind": kind, "trigger_rate": rate, "median_value": float(np.median(old))}
        print(f"  {kind:5s} {name:10s} judged 'changed' {rate:6.1%}   median {np.median(old):.5f}")

    # Sweep: min_cells = max noise cells + 1, then compare detection of real changes for each delta
    print("\nnew criterion, sweeping delta:")
    best = None
    for d in DELTAS:
        noise_max = max(int(v[2][d].max()) for v in data.values() if v[0] == "noise")
        m = noise_max + 1
        rates = {name: float(np.mean(v[2][d] >= m)) for name, v in data.items() if v[0] == "real"}
        big = min(r for k, r in rates.items() if "打字" not in k)   # exclude the typing cases ("打字" = typing)
        res["sweep"][str(d)] = {"noise_max_cells": noise_max, "min_cells": m, "detect": rates}
        print(f"  delta={d:2d}: noise max {noise_max} cells -> min_cells={m}; detection " +
              ", ".join(f"{k} {r:.0%}" for k, r in rates.items()))
        score = (big, sum(rates.values()))
        if best is None or score > best[0]:
            best = (score, d, m)
    _, d, m = best
    print(f"\nchosen: cell_delta={d}, min_cells={m}")
    for name, (kind, _, new) in data.items():
        res["new"][name] = {"kind": kind, "trigger_rate": float(np.mean(new[d] >= m)),
                            "cells_min": int(new[d].min()), "cells_median": float(np.median(new[d])),
                            "cells_max": int(new[d].max())}
    res["chosen"] = {"cell_delta": d, "min_cells": m}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "week6",
                       "threshold_study.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
