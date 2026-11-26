"""Effect of dynamic-region masking: does verification still work while a "video" keeps playing on screen?

    python -m scripts.study_mask

A 320x180 video region whose content changes randomly every frame is placed on the synthetic desktop
(the same one as study_threshold). Two rates are compared under three settings:
  false-trigger rate: nothing happened (only the video played), yet verification says "the screen changed"
  detection rate: a menu/dialog/message really appeared, and verification says "the screen changed"
Settings: no mask / calibrate() automatic mask / the video region put into ignore_regions by hand.
Change names (keys of results/week8/mask_study.json): 弹出菜单 popup menu, 弹对话框 dialog, 发出一条消息 send message.
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
    """The video is a low-frequency random image (8x5 random colour blocks, upscaled) that changes in large patches
    like real video, rather than per-pixel white noise, which averages out inside a 30x30 cell and hides the problem."""
    out = img.copy()
    x1, y1, x2, y2 = VIDEO
    small = np.random.default_rng(rng.randint(0, 1 << 30)).integers(0, 256, (5, 8, 3), dtype=np.uint8)
    out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return out


def popup_away_from_video(img, rng):
    # Random menu position, but never over the video region, so the effect of the mask itself is visible
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
