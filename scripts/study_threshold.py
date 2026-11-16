"""验证阈值 change_thr 是怎么定的：在合成屏幕上量各种"画面变化"的指纹差。

    python -m scripts.study_threshold

做法：画一张 1920x1080 的假桌面（渐变壁纸 + 浏览器窗口 + 一堆文字），对它做两类改动，
用 ScreenPerceiver 里同样的指纹（64x36 灰度，INTER_AREA）判断前后画面变没变：
  噪声类 —— 点击之后我们不希望算作"画面变了"的：光标挪位置、输入光标闪烁、时钟走一分钟
  变化类 —— 点击/输入真的起作用时会出现的：弹出菜单、搜索框里打字、发出一条消息、
            弹对话框、整页切换
每种改动随机位置/内容重复 N 次。比较两种判据：
  旧：全图平均灰度差 mean(|a-b|)/255 > 0.01
  新：灰度差超过 cell_delta 的格子数 >= min_cells（现在用的）
再扫一遍 (cell_delta, min_cells)，按"噪声类一次都不误触发"挑最小的 min_cells。
合成画面不是真实屏幕：真机上的噪声（视频、动画、广告）更多，所以这只是定初值的依据，
真机上用 python -m scripts.bench_perception --idle-check 再核对一次。
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
    """渐变壁纸 + 一个浏览器窗口，窗口里有标题栏、地址栏、搜索框和若干行文字。"""
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    c0, c1 = np.array([120, 90, 40], np.float32), np.array([200, 160, 90], np.float32)
    img = (c0 * (1 - x) + c1 * x) * (1 - 0.3 * y)
    img = np.broadcast_to(img, (H, W, 3)).astype(np.uint8).copy()
    cv2.rectangle(img, (160, 60), (1760, 1000), (250, 250, 250), -1)          # 窗口
    cv2.rectangle(img, (160, 60), (1760, 110), (225, 225, 225), -1)           # 标题栏
    cv2.rectangle(img, (160, 110), (1760, 170), (240, 240, 240), -1)          # 地址栏
    cv2.rectangle(img, (500, 400), (1300, 440), (255, 255, 255), -1)          # 搜索框
    cv2.rectangle(img, (500, 400), (1300, 440), (150, 150, 150), 1)
    for i in range(14):
        n = rng.randint(20, 60)
        cv2.putText(img, "lorem ipsum " * 5 + str(i), (200, 230 + i * 8 * 4 % 150 + i * 30),
                    FONT, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        _ = n
    return img


def put_text(img, s, org, scale=0.7):
    cv2.putText(img, s, org, FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)


# ---- 噪声类 ----
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


# ---- 变化类 ----
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
    n = rng.randint(4, 24)                         # 打了 4~24 个字符
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


# 打字不按回车时的变化，按回车通常会跳到结果页（整页切换）
REAL = {"弹出菜单": popup_menu, "发出一条消息": send_message, "弹对话框": dialog, "整页切换": page_switch,
        "搜索框打字16字符": typing(16), "搜索框打字8字符": typing(8), "搜索框打字4字符": typing(4)}


def cells_changed(a, b, delta):
    return int(np.sum(np.abs(signature(a) - signature(b)) > delta))


def collect(n=200, seed=7):
    """每类改动生成 n 对前后画面，返回 {名字: (类别, 旧判据的值数组, {delta: 变化格数数组})}。"""
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
    print("旧判据 mean>0.01：")
    for name, (kind, old, _) in data.items():
        rate = float(np.mean(old > OLD_THR))
        res["old"][name] = {"kind": kind, "trigger_rate": rate, "median_value": float(np.median(old))}
        print(f"  {kind:5s} {name:10s} 判为'变了'的比例 {rate:6.1%}   中位数 {np.median(old):.5f}")

    # 扫参数：min_cells 取噪声类最大格数 + 1，再比较各 delta 下变化类的检出率
    print("\n新判据，按 delta 扫描：")
    best = None
    for d in DELTAS:
        noise_max = max(int(v[2][d].max()) for v in data.values() if v[0] == "noise")
        m = noise_max + 1
        rates = {name: float(np.mean(v[2][d] >= m)) for name, v in data.items() if v[0] == "real"}
        big = min(r for k, r in rates.items() if "打字" not in k)
        res["sweep"][str(d)] = {"noise_max_cells": noise_max, "min_cells": m, "detect": rates}
        print(f"  delta={d:2d}: 噪声最多 {noise_max} 格 -> min_cells={m}; 检出率 " +
              ", ".join(f"{k} {r:.0%}" for k, r in rates.items()))
        score = (big, sum(rates.values()))
        if best is None or score > best[0]:
            best = (score, d, m)
    _, d, m = best
    print(f"\n选 cell_delta={d}, min_cells={m}")
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
