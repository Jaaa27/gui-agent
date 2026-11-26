"""How much does downscaling (max_side) cost? Computed once, then measured with tesseract on synthetic screens.

    python -m scripts.study_downscale

1. Computed: scale factor, pixel ratio (the detector is fully convolutional, so time is roughly proportional
   to pixels), and the text size after downscaling.
2. Measured: synthetic 1920x1080 / 2560x1440 / 3840x2160 screens (Chinese and English UI words rendered with
   Noto Sans CJK) are downscaled to max_side and given to tesseract; we count recall, the coordinate error after
   mapping back to the original image, and the time of the resize itself.

Note that tesseract is not EasyOCR: absolute recall numbers do not carry over. What carries over is the trend
"below which text size does recognition start to miss". Use scripts.bench_perception for EasyOCR on a real machine.
"""
import csv
import io
import json
import os
import random
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
WORDS = ["Settings", "Search", "Google Chrome", "File", "Edit", "View", "Downloads", "Documents", "Notepad",
         "Calculator", "Network", "Bluetooth", "Display", "Sound", "Privacy", "Update", "Account", "Help",
         "文件", "设置", "搜索", "关闭", "最近使用", "桌面", "下载", "文档", "新建", "打开方式", "发送", "回收站"]
RES = [(1920, 1080), (2560, 1440), (3840, 2160)]
MAX_SIDES = [0, 1600, 1280, 1024]


def render(w, h, font_px, seed=0):
    """Lay out words in a grid on a light grey background. Returns the image and [(text, (x1,y1,x2,y2)), ...]."""
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (243, 243, 243))
    dr = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, font_px, index=0)
    cw, ch = int(font_px * 11), int(font_px * 2.6)           # cell size proportional to the font size
    boxes = []
    for gy in range(int(ch * 0.5), h - ch, ch):
        for gx in range(int(cw * 0.2), w - cw, cw):
            t = rng.choice(WORDS)
            dr.text((gx, gy), t, font=f, fill=(25, 25, 25))
            boxes.append((t, dr.textbbox((gx, gy), t, font=f)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), boxes


def ocr(img):
    """tesseract (Chinese + English). Returns [(text, centre x, centre y)] per word/character block."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.png")
        cv2.imwrite(p, img)
        out = subprocess.run(["tesseract", p, "stdout", "-l", "eng+chi_sim", "--psm", "11", "tsv"],
                             capture_output=True, text=True).stdout
    rows = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        if t and float(r["conf"]) > 0:
            x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
            rows.append((t, x, y, x + w, y + h))
    return rows


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def score(boxes, found, scale):
    """A target counts as found if the recognized text inside its (slightly enlarged) box contains the target text.
    Also returns the centre error of found targets, in original-image pixels."""
    hit, errs = 0, []
    for t, (x1, y1, x2, y2) in boxes:
        pad = 4
        got, cx, cy, n = "", 0, 0, 0
        for (ft, fx1, fy1, fx2, fy2) in found:
            fcx, fcy = (fx1 + fx2) / 2 / scale, (fy1 + fy2) / 2 / scale
            if x1 - pad <= fcx <= x2 + pad and y1 - pad <= fcy <= y2 + pad:
                got += ft
                cx += fcx
                cy += fcy
                n += 1
        if n and norm(t) and norm(t) in norm(got):
            hit += 1
            errs.append(float(np.hypot(cx / n - (x1 + x2) / 2, cy / n - (y1 + y2) / 2)))
    return hit, errs


def downscale(img, max_side):
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img, 1.0
    s = max_side / float(max(h, w))
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


# For each resolution, two UI text sizes: the usual size at 100% scaling, and the size after system scaling (physical px)
CASES = [((1920, 1080), 12), ((1920, 1080), 16),
         ((2560, 1440), 12), ((2560, 1440), 18),
         ((3840, 2160), 12), ((3840, 2160), 18), ((3840, 2160), 24)]


def main():
    rows = []
    for (w, h), fpx in CASES:
        img, boxes = render(w, h, fpx)
        for ms in MAX_SIDES:
            t0 = time.perf_counter()
            small, s = downscale(img, ms)
            t_resize = time.perf_counter() - t0
            t0 = time.perf_counter()
            found = ocr(small)
            t_ocr = time.perf_counter() - t0
            hit, errs = score(boxes, found, s)
            rows.append({"res": f"{w}x{h}", "font_px": fpx, "max_side": ms, "scale": round(s, 3),
                         "pixel_ratio": round(s * s, 3), "font_after_px": round(fpx * s, 1),
                         "targets": len(boxes), "recall": round(100.0 * hit / len(boxes), 1),
                         "center_err_mean": round(float(np.mean(errs)), 2) if errs else None,
                         "center_err_max": round(float(np.max(errs)), 2) if errs else None,
                         "resize_ms": round(1000 * t_resize, 1), "ocr_s": round(t_ocr, 2)})
            r = rows[-1]
            print(f"{r['res']:>9} font {fpx:>3}px max_side={ms:>4} scale {r['scale']:.3f} font after {r['font_after_px']:>5} "
                  f"recall {r['recall']:>5}% centre err mean {r['center_err_mean']} max {r['center_err_max']} "
                  f"resize {r['resize_ms']}ms tesseract {r['ocr_s']}s", flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "week6",
                       "downscale_cost.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
