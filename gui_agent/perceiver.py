"""Screen perception: screenshot -> OCR -> number every text block (Set-of-Mark).

Same interface as the original ScreenPerceiver: scan_screen() returns (annotated image path, {id: {"text", "center"}}).
easyocr / mss are imported lazily, so this module imports without them and tests can inject a fake reader.

Added in week 8:
  * max_side="auto": the downscale factor is derived from the display scale so that UI text stays at least MIN_FONT_PX
    (in the week-6 study, recall dropped clearly once text fell below about 13 px)
  * the screen signature can mask cells: calibrate() finds regions that change on their own on an idle screen
    (video, animations, the clock), or ignore_regions sets them by hand; masked cells are skipped when comparing
  * PaddleOCR as an optional engine (engine="paddle"), its output converted to the EasyOCR format
"""
import math
import os
import time

import cv2
import numpy as np


def enable_dpi_awareness():
    """Enable DPI awareness on Windows. Must be called before the first screenshot and before pyautogui is created.

    Without it, at 125%/150% display scaling the screenshot pixels and mouse coordinates disagree and every click is offset.
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
    """Windows display scale (1.25 for 125%), or None if it cannot be read.

    Once DPI awareness is on, pyautogui.size() returns physical pixels, so the scale can no longer be derived from
    "screenshot width / logical width"; we ask the system instead. GetDpiForSystem needs Windows 10 1607 or later.
    """
    if os.name != "nt":
        return None
    import ctypes
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96.0
    except Exception:
        return None


BASE_FONT_PX = 12      # UI text (9 pt Segoe UI) is about 12 px at 100% scaling
MIN_FONT_PX = 13       # text size to keep after downscaling


def paddle_to_easyocr(result):
    """Convert PaddleOCR output to EasyOCR's [(4-point box, text, confidence), ...].

    2.x ocr() returns [[ [box, (text, score)], ... ]] (one list per image, [None] if nothing was found);
    3.x predict() returns one dict-like object per image with rec_texts / rec_scores / rec_polys. Both are handled.
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
    """Wrapper that gives PaddleOCR the same readtext() as easyocr.Reader."""

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
        # max_side: downscale the long side to at most this many pixels before OCR; 0 = never, "auto" = from the display scale
        # cell_delta / min_cells: the screen counts as changed when at least min_cells signature cells differ by more than cell_delta
        #   see scripts/study_threshold.py and section 3.4 of the week-6 report for how these were chosen
        # logical_size: screen size as pyautogui sees it; queried if not given
        # reader / capture_fn: for tests only
        enable_dpi_awareness()
        self.max_side = max_side
        self.min_conf = min_conf
        self.cache = cache
        self.sig_size = sig_size
        self.cell_delta = cell_delta
        self.min_cells = min_cells
        self.mask = None               # bool array the size of the signature; True cells are ignored
        self.ui_scale = ui_scale       # display scale; queried from the system if not given
        self._capture_fn = capture_fn
        if reader is None:
            reader = PaddleReader() if engine == "paddle" else self._make_reader(langs, use_gpu)
        self.reader = reader
        if ignore_regions:
            self.add_ignore_regions(ignore_regions)
        self._logical_size = logical_size
        self._last_sig = None
        self._last = None          # (annotated image, element dict), reused while the screen is unchanged
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
                self._logical_size = False      # could not get it; don't ask again
        return self._logical_size or None

    # ---- screenshot and screen signature ----
    def capture_screen(self):
        if self._capture_fn is not None:
            return self._capture_fn()
        import mss
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])      # primary monitor
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def signature_of(self, img):
        # Shrink to a 64x36 grayscale image as the signature used to tell whether the screen changed
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.resize(g, self.sig_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def capture_signature(self):
        return self.signature_of(self.capture_screen())

    def add_ignore_regions(self, regions):
        """regions is [(x1, y1, x2, y2), ...] in 0..1 relative coordinates, e.g. the clock in the bottom-right of the taskbar."""
        h, w = self.sig_size[1], self.sig_size[0]
        if self.mask is None:
            self.mask = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in regions:
            c1, c2 = int(math.floor(x1 * w)), int(math.ceil(x2 * w))
            r1, r2 = int(math.floor(y1 * h)), int(math.ceil(y2 * h))
            self.mask[max(r1, 0):min(r2, h), max(c1, 0):min(c2, w)] = True

    def calibrate(self, frames=8, gap=0.3, sleep=time.sleep):
        """Take several signatures of an idle screen; cells that change on their own (video, animations) are ignored from then on.

        Changed cells are dilated by one cell, because the edges of dynamic content flicker in neighbouring cells. Returns the number of masked cells.
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
        # Not a whole-image mean difference: a small change (a popup menu covering 4% of the screen) gets averaged away,
        # and raising the threshold to compensate then reports real changes as "no effect"
        d = np.abs(a - b) > self.cell_delta
        if self.mask is not None and self.mask.shape == d.shape:
            d &= ~self.mask
        return int(np.sum(d)) >= self.min_cells

    def invalidate(self):
        """Call after a click/type so the next scan does not use the cache."""
        self._last_sig = None
        self._last = None

    # ---- OCR ----
    def auto_max_side(self, img_shape):
        """Long-side limit from the display scale: UI text is about BASE_FONT_PX * scale and must stay >= MIN_FONT_PX after downscaling.

        At 100% the text is only 12 px, already below 13, so no downscaling; a 4K screen at 150% can go down to 0.72.
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
        # OCR image coordinates -> original screenshot pixels -> pyautogui coordinates
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
            eid = len(elements)       # number after filtering, so IDs are contiguous
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
