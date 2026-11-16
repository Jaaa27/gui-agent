"""降采样(max_side)的代价到底多大：算一遍、再在合成屏幕上用 tesseract 实测一遍。

    python -m scripts.study_downscale

1. 算出来的：缩放比例、像素数比例（检测网络是全卷积，耗时大致跟像素数成正比）、
   文字缩小之后的字号。
2. 量出来的：把 1920x1080 / 2560x1440 / 3840x2160 的合成屏幕（Noto Sans CJK 渲染的中英文界面词）
   按 max_side 缩小后交给 tesseract，数召回率、坐标放回原图后的误差、缩放本身的耗时。

注意 tesseract 不是 EasyOCR：召回率的绝对值不能直接搬到 EasyOCR 上，能参考的是
"字号缩到多小开始漏识"这条趋势。EasyOCR 在真机上的数字用 scripts.bench_perception 测。
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
    """在浅灰底上按网格摆文字，返回图像和 [(文字, (x1,y1,x2,y2)), ...]。"""
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (243, 243, 243))
    dr = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, font_px, index=0)
    cw, ch = int(font_px * 11), int(font_px * 2.6)           # 格子宽高跟字号成比例
    boxes = []
    for gy in range(int(ch * 0.5), h - ch, ch):
        for gx in range(int(cw * 0.2), w - cw, cw):
            t = rng.choice(WORDS)
            dr.text((gx, gy), t, font=f, fill=(25, 25, 25))
            boxes.append((t, dr.textbbox((gx, gy), t, font=f)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), boxes


def ocr(img):
    """tesseract 中英文，返回 [(文字, 中心x, 中心y)]（按行内的词/字符块）。"""
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
    """一个目标算命中：落在它框内（放大一圈）的识别结果拼起来包含目标文字。
    同时返回命中目标的中心误差（原图像素）。"""
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


# 每种分辨率下取两种界面字号：100% 缩放下的常见字号，和系统放大后的字号（物理像素）
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
            print(f"{r['res']:>9} 字号{fpx:>3}px max_side={ms:>4} 比例{r['scale']:.3f} 缩后字号{r['font_after_px']:>5} "
                  f"召回{r['recall']:>5}% 中心误差均值{r['center_err_mean']} 最大{r['center_err_max']} "
                  f"缩放{r['resize_ms']}ms tesseract {r['ocr_s']}s", flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "week6",
                       "downscale_cost.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
