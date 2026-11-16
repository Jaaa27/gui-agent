"""动态区域屏蔽的效果：画面里有一块一直在播的"视频"时，验证还能不能用。

    python -m scripts.study_mask

合成桌面（和 study_threshold 同一套）上放一块 320x180 的视频区，每一帧内容随机变化。
对比三种做法下的两个比例：
  误触发率：什么都没做（只有视频在播），验证却说"画面变了"
  检出率：真的弹出了菜单/对话框/发了消息，验证说"画面变了"
做法：不屏蔽 / calibrate() 自动屏蔽 / 把视频区手动写进 ignore_regions。
"""
import json
import os
import random

import cv2
import numpy as np

from gui_agent.perceiver import ScreenPerceiver
from scripts.study_threshold import H, W, base_screen, dialog, popup_menu, send_message

VIDEO = (1300, 760, 1620, 940)        # x1, y1, x2, y2


def with_video(img, rng):
    """视频画面用低频随机图代替（8x5 的随机色块放大），像真实视频一样大块大块地变，
    而不是逐像素的白噪声——白噪声在 30x30 的格子里一平均就没了，测不出问题。"""
    out = img.copy()
    x1, y1, x2, y2 = VIDEO
    small = np.random.default_rng(rng.randint(0, 1 << 30)).integers(0, 256, (5, 8, 3), dtype=np.uint8)
    out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return out


def popup_away_from_video(img, rng):
    # 菜单的位置随机，但不盖住视频区，否则看不出屏蔽本身的影响
    while True:
        a, b = popup_menu(img, rng)
        if not (b[VIDEO[1]:VIDEO[3], VIDEO[0]:VIDEO[2]] != a[VIDEO[1]:VIDEO[3], VIDEO[0]:VIDEO[2]]).any():
            return a, b


REAL = {"弹出菜单": popup_away_from_video, "弹对话框": dialog, "发出一条消息": send_message}


def make_perceiver(mode, rng, base):
    frames = iter([])

    def capture():
        return next(frames)
    p = ScreenPerceiver(reader=object(), capture_fn=capture)
    if mode == "auto":
        frames = iter([with_video(base, rng) for _ in range(8)])
        p.calibrate(frames=8, sleep=lambda s: None)
    elif mode == "manual":
        x1, y1, x2, y2 = VIDEO
        p.add_ignore_regions([(x1 / W, y1 / H, x2 / W, y2 / H)])
    return p


def run(n=100, seed=11):
    rng = random.Random(seed)
    res = {}
    for mode in ("none", "auto", "manual"):
        false_trig, detect, masked = 0, {k: 0 for k in REAL}, []
        for _ in range(n):
            base = base_screen(rng)
            p = make_perceiver(mode, rng, base)
            masked.append(0 if p.mask is None else int(p.mask.sum()))
            a, b = with_video(base, rng), with_video(base, rng)
            false_trig += p.signature_changed(p.signature_of(a), p.signature_of(b))
            for name, fn in REAL.items():
                x, y = fn(base, rng)
                detect[name] += p.signature_changed(p.signature_of(with_video(x, rng)),
                                                    p.signature_of(with_video(y, rng)))
        res[mode] = {"false_trigger": false_trig / n, "detect": {k: v / n for k, v in detect.items()},
                     "masked_cells": float(np.mean(masked)), "total_cells": 64 * 36}
        print(mode, res[mode])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "week8",
                       "mask_study.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
