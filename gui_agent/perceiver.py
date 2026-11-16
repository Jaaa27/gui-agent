"""屏幕感知：截图 -> OCR -> 给每段文字编号（Set-of-Mark）。

和最早的 ScreenPerceiver 接口一样，scan_screen() 返回 (标注图路径, {id: {"text", "center"}})。
easyocr / mss 都是用到才 import，这样没装也能 import 本模块，测试里直接塞假的 reader 进来。

第 8 周加的三件事：
  * max_side="auto"：按系统缩放比例算缩放系数，保证缩小后界面字号不低于 MIN_FONT_PX
    （第 6 周的实验里，缩小后字号低于约 13 像素召回率开始明显下降）
  * 画面指纹可以屏蔽一部分格子：calibrate() 在静止屏幕上找出自己会变的区域（视频、动图、时钟），
    或者用 ignore_regions 手动指定，比较画面变化时跳过这些格子
  * OCR 引擎可选 PaddleOCR（engine="paddle"），输出转成和 EasyOCR 一样的格式
"""
import math
import os
import time

import cv2
import numpy as np


def enable_dpi_awareness():
    """Windows 上开启 DPI 感知，要赶在第一次截图和创建 pyautogui 之前调用。

    不开的话，系统缩放 125%/150% 时截图像素和鼠标坐标对不上，点击会整体偏。
    """
    if os.name != "nt":
        return False
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return True
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return True
        except Exception:
            return False


def system_ui_scale():
    """Windows 的显示缩放比例（125% 返回 1.25），拿不到返回 None。

    开了 DPI 感知以后 pyautogui.size() 返回的就是物理像素，没法再用"截图宽 / 逻辑宽"推出缩放比例，
    所以直接问系统。GetDpiForSystem 要 Win10 1607 以上。
    """
    if os.name != "nt":
        return None
    import ctypes
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96.0
    except Exception:
        return None


BASE_FONT_PX = 12      # Windows 100% 缩放下界面文字（9 号 Segoe UI）大约 12 像素
MIN_FONT_PX = 13       # 缩小后希望保住的字号


def paddle_to_easyocr(result):
    """PaddleOCR 的输出转成 EasyOCR 的 [(四点框, 文字, 置信度), ...]。

    2.x 的 ocr() 返回 [[ [框, (文字, 分数)], ... ]]（每张图一个列表，没识别到是 [None]）；
    3.x 的 predict() 返回每张图一个类字典对象，字段是 rec_texts / rec_scores / rec_polys。两种都认。
    """
    out = []
    if not result:
        return out
    for page in result:
        if page is None:
            continue
        if hasattr(page, "get") and page.get("rec_texts") is not None:
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys")
            for box, text, score in zip(polys, page["rec_texts"], page["rec_scores"]):
                out.append(([[float(p[0]), float(p[1])] for p in box], str(text), float(score)))
            continue
        for item in page:
            if item is None:
                continue
            box, (text, score) = item[0], item[1]
            out.append(([[float(p[0]), float(p[1])] for p in box], str(text), float(score)))
    return out


class PaddleReader:
    """包一层，让 PaddleOCR 有和 easyocr.Reader 一样的 readtext()。"""

    def __init__(self, lang="ch", ocr=None):
        if ocr is None:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang=lang)
        self.ocr = ocr

    def readtext(self, img):
        fn = getattr(self.ocr, "predict", None) or self.ocr.ocr
        return paddle_to_easyocr(fn(img))


class ScreenPerceiver:
    def __init__(self, langs=("en", "ch_sim"), max_side="auto", min_conf=0.3, use_gpu=None,
                 cache=True, sig_size=(64, 36), cell_delta=6, min_cells=3, ignore_regions=None,
                 logical_size=None, reader=None, capture_fn=None, engine="easyocr", ui_scale=None):
        # max_side: OCR 之前把长边缩到这个像素以内，0 就是不缩，"auto" 按系统缩放比例自己算
        # cell_delta / min_cells: 指纹里灰度差超过 cell_delta 的格子数达到 min_cells 才算画面变了
        #   取值依据见 scripts/study_threshold.py 和第 6 周报告 3.4 节
        # logical_size: pyautogui 认为的屏幕大小，不传就自己去问
        # reader / capture_fn: 只给测试用
        enable_dpi_awareness()
        self.max_side = max_side
        self.min_conf = min_conf
        self.cache = cache
        self.sig_size = sig_size
        self.cell_delta = cell_delta
        self.min_cells = min_cells
        self.mask = None               # 和指纹一样大的 bool 数组，True 的格子不参与比较
        self.ui_scale = ui_scale       # 系统缩放比例，不传就去问系统
        self._capture_fn = capture_fn
        if reader is None:
            reader = PaddleReader() if engine == "paddle" else self._make_reader(langs, use_gpu)
        self.reader = reader
        if ignore_regions:
            self.add_ignore_regions(ignore_regions)
        self._logical_size = logical_size
        self._last_sig = None
        self._last = None          # (标注图, 元素字典)，画面没变时直接复用
        self.last_timing = {}

    @staticmethod
    def _make_reader(langs, use_gpu):
        import easyocr
        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
        return easyocr.Reader(list(langs), gpu=bool(use_gpu))

    def _screen_logical_size(self):
        if self._logical_size is None:
            try:
                import pyautogui
                self._logical_size = tuple(pyautogui.size())
            except Exception:
                self._logical_size = False      # 问不到就别再问了
        return self._logical_size or None

    # ---- 截图和画面指纹 ----
    def capture_screen(self):
        if self._capture_fn is not None:
            return self._capture_fn()
        import mss
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])      # 主屏幕
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def signature_of(self, img):
        # 缩成 64x36 的灰度图当指纹，用来判断画面变没变
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.resize(g, self.sig_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def capture_signature(self):
        return self.signature_of(self.capture_screen())

    def add_ignore_regions(self, regions):
        """regions 是 [(x1, y1, x2, y2), ...]，用 0~1 的相对坐标，比如任务栏右下角的时钟。"""
        h, w = self.sig_size[1], self.sig_size[0]
        if self.mask is None:
            self.mask = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in regions:
            c1, c2 = int(math.floor(x1 * w)), int(math.ceil(x2 * w))
            r1, r2 = int(math.floor(y1 * h)), int(math.ceil(y2 * h))
            self.mask[max(r1, 0):min(r2, h), max(c1, 0):min(c2, w)] = True

    def calibrate(self, frames=8, gap=0.3, sleep=time.sleep):
        """屏幕不动的时候连续取几帧指纹，自己在变的格子（视频、动图）以后不参与比较。

        变化的格子会向外扩一格，因为动态内容的边缘在相邻格子里也会抖。返回屏蔽了多少格。
        """
        sigs = []
        for i in range(frames):
            if i:
                sleep(gap)
            sigs.append(self.capture_signature())
        dyn = np.zeros(sigs[0].shape, bool)
        for a, b in zip(sigs, sigs[1:]):
            dyn |= np.abs(a - b) > self.cell_delta
        if dyn.any():
            dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.mask = dyn if self.mask is None else (self.mask | dyn)
        self._last_sig = None
        return int(self.mask.sum())

    def signature_changed(self, a, b):
        # 不用全图平均差：小面积的变化（弹出菜单只占 4% 的画面）会被平均掉，
        # 反而是调大阈值之后把真正的变化当成"没反应"
        d = np.abs(a - b) > self.cell_delta
        if self.mask is not None and self.mask.shape == d.shape:
            d &= ~self.mask
        return int(np.sum(d)) >= self.min_cells

    def invalidate(self):
        """点击/输入之后调一下，保证下一次扫描不吃缓存。"""
        self._last_sig = None
        self._last = None

    # ---- OCR ----
    def auto_max_side(self, img_shape):
        """按系统缩放比例算长边上限：界面字号约 BASE_FONT_PX * 缩放比例，缩小后要不低于 MIN_FONT_PX。

        100% 缩放时字号只有 12 像素，本来就低于 13，所以不缩；150% 的 4K 屏可以缩到 0.72。
        """
        h, w = img_shape[:2]
        dpi = self.ui_scale or system_ui_scale()
        if not dpi:
            ls = self._screen_logical_size()
            dpi = (w / float(ls[0])) if ls else 1.0
        s = MIN_FONT_PX / (BASE_FONT_PX * max(dpi, 1.0))
        return 0 if s >= 1 else int(math.ceil(max(h, w) * s))

    def _downscale(self, img):
        h, w = img.shape[:2]
        max_side = self.auto_max_side(img.shape) if self.max_side == "auto" else self.max_side
        if not max_side or max(h, w) <= max_side:
            return img, 1.0
        scale = max_side / float(max(h, w))
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return small, scale

    def _to_screen_coords(self, x, y, scale, img_shape):
        # OCR 图上的坐标 -> 原截图像素 -> pyautogui 坐标
        px, py = x / scale, y / scale
        ls = self._screen_logical_size()
        if ls:
            h, w = img_shape[:2]
            px, py = px * ls[0] / w, py * ls[1] / h
        return int(round(px)), int(round(py))

    def extract_and_mark(self, img, draw=True):
        small, scale = self._downscale(img)
        results = self.reader.readtext(small)
        marked = img.copy() if draw else None
        elements = []
        for bbox, text, conf in results:
            if float(conf) < self.min_conf:
                continue
            text = str(text).strip()
            if not text:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1, x2, y2 = min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale
            eid = len(elements)       # 过滤完再编号，id 是连续的
            cx, cy = self._to_screen_coords((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, scale, img.shape)
            if draw:
                cv2.rectangle(marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(marked, f"[{eid}]", (int(x1), max(int(y1) - 10, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elements.append({"id": eid, "text": text, "center": (cx, cy)})
        return marked, elements

    def scan_screen(self, save_path="som_test_result.png"):
        t0 = time.perf_counter()
        img = self.capture_screen()
        t_cap = time.perf_counter() - t0
        sig = self.signature_of(img)

        if (self.cache and self._last is not None and self._last_sig is not None
                and not self.signature_changed(self._last_sig, sig)):
            marked, elements_dict = self._last
            cv2.imwrite(save_path, marked)
            self.last_timing = {"capture": t_cap, "ocr": 0.0, "total": time.perf_counter() - t0,
                                "cache_hit": True, "n": len(elements_dict)}
            return save_path, elements_dict

        t1 = time.perf_counter()
        marked, elements = self.extract_and_mark(img)
        t_ocr = time.perf_counter() - t1
        cv2.imwrite(save_path, marked)
        elements_dict = {e["id"]: {"text": e["text"], "center": e["center"]} for e in elements}
        self._last, self._last_sig = (marked, elements_dict), sig
        self.last_timing = {"capture": t_cap, "ocr": t_ocr, "total": time.perf_counter() - t0,
                            "cache_hit": False, "n": len(elements_dict)}
        return save_path, elements_dict
