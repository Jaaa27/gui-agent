"""
Test doubles: fake pyautogui / pyperclip, and a "screen state machine" (World) with a fake perceiver,
fake controller and fake LLM, used to inject all kinds of faults into the executor.
Screen texts are Chinese UI strings (记事本 Notepad, 回收站 Recycle Bin, 关闭 Close, 书签 Bookmarks, 搜索 Search).
"""
import copy
import sys
import time
import types
import zlib

import numpy as np


# ----------------------------------------------------------------------
# Fake pyautogui / pyperclip (the sandbox has no display)
# ----------------------------------------------------------------------
def install_fake_gui_modules():
    pg = types.ModuleType("pyautogui")
    pg.FAILSAFE = True
    pg.calls = []
    pg.size = lambda: (1920, 1080)
    class FailSafeException(Exception):
        pass
    pg.FailSafeException = FailSafeException
    for name in ("moveTo", "click", "typewrite", "press", "hotkey", "scroll", "dragTo"):
        setattr(pg, name, (lambda n: (lambda *a, **k: pg.calls.append((n, a, k))))(name))
    pc = types.ModuleType("pyperclip")
    pc.store = "原来的剪贴板"
    pc.copy = lambda t: setattr(pc, "store", t)
    pc.paste = lambda: pc.store
    sys.modules["pyautogui"] = pg
    sys.modules["pyperclip"] = pc
    return pg, pc


# ----------------------------------------------------------------------
# Screen state machine
# ----------------------------------------------------------------------
def E(text, x, y):
    return {"text": text, "center": (x, y)}


def make_states():
    return {
        "desktop": {
            "elements": {10: E("Google Chrome", 100, 100), 11: E("Microsoft Edge", 100, 150),
                         12: E("记事本", 100, 200), 13: E("回收站", 100, 250)},
            "on_click": {10: "chrome", 11: "edge", 12: "notepad"}},
        "chrome": {
            "elements": {20: E("Search Google or type a URL", 500, 300), 21: E("关闭", 1900, 10),
                         22: E("书签", 300, 60)},
            "on_click": {21: "desktop"}, "on_type": {20: "results"}},
        "chrome_dup": {          # two "Close" (关闭) elements; the first one does nothing when clicked
            "elements": {21: E("关闭", 1900, 10), 23: E("关闭", 1900, 60), 22: E("书签", 300, 60)},
            "on_click": {23: "desktop"}},
        "chrome_btn": {          # a search button and a search box at the same time
            "elements": {41: E("搜索", 900, 300), 20: E("Search Google or type a URL", 500, 300)},
            "on_click": {41: "chrome_btn"}, "on_type": {20: "results"}},
        "oob": {"elements": {10: E("Google Chrome", 5000, 5000)}, "on_click": {10: "chrome"}},
        "results": {"elements": {30: E("McMaster 选课 - 搜索结果", 600, 400), 31: E("首页", 100, 60)},
                    "on_click": {31: "desktop"}},
        "edge": {"elements": {50: E("新建标签页", 300, 60)}, "on_click": {}},
        "notepad": {"elements": {60: E("无标题 - 记事本", 300, 60)}, "on_click": {}},
    }


class World:
    def __init__(self, start="desktop"):
        self.states = make_states()
        self.state = start
        self.typed = []
        self.last_clicked = None

    def elements(self):
        return copy.deepcopy(self.states[self.state]["elements"])

    def goto(self, s):
        self.state = s

    def sig_value(self):
        return zlib.crc32(self.state.encode()) % 256


class FakePerceiver:
    def __init__(self, world, rng=None, p_miss=0.0, miss_texts=(), miss_first_n=0, exc_first_n=0,
                 miss_rules=None):
        self.world, self.rng, self.p_miss = world, rng, p_miss
        self.miss_texts = set(miss_texts)
        self.miss_first_n, self.exc_first_n = miss_first_n, exc_first_n
        self.miss_rules = list(miss_rules or [])        # [[state, {texts}, count], ...]
        self.scans = 0
        self.invalidated = 0

    def scan_screen(self, save_path="fake.png"):
        self.scans += 1
        if self.exc_first_n > 0:
            self.exc_first_n -= 1
            raise RuntimeError("OCR engine crashed (simulated)")
        els = self.world.elements()
        hide = set()
        if self.miss_first_n > 0:
            self.miss_first_n -= 1
            hide = self.miss_texts
        elif self.rng is not None and self.rng.random() < self.p_miss:
            hide = self.miss_texts
        for rule in self.miss_rules:
            if rule[0] == self.world.state and rule[2] > 0:
                rule[2] -= 1
                hide = hide | rule[1]
        return save_path, {k: v for k, v in els.items() if v["text"] not in hide}

    def capture_signature(self):
        return np.full((4, 4), self.world.sig_value(), dtype=np.float32)

    def signature_changed(self, a, b):
        return float(np.mean(np.abs(a - b)) / 255.0) > 0.01

    def invalidate(self):
        self.invalidated += 1


class FakeController:
    width, height = 1920, 1080

    def __init__(self, world, rng=None, p_fail=0.0, p_noeffect=0.0, fail_first_n=0, noeffect_ids=()):
        self.world, self.rng = world, rng
        self.p_fail, self.p_noeffect = p_fail, p_noeffect
        self.fail_first_n = fail_first_n
        self.noeffect_ids = set(noeffect_ids)
        self.clicks, self.types = [], []

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        self.clicks.append((x, y))
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            return False
        if self.rng is not None and self.rng.random() < self.p_fail:
            return False
        st = self.world.states[self.world.state]
        eid = next((k for k, v in st["elements"].items()
                    if abs(v["center"][0] - x) <= 2 and abs(v["center"][1] - y) <= 2), None)
        self.world.last_clicked = eid
        effect = eid not in self.noeffect_ids
        if effect and self.rng is not None and self.rng.random() < self.p_noeffect:
            effect = False
        if effect and eid in st.get("on_click", {}):
            self.world.goto(st["on_click"][eid])
        return True

    def type_text(self, text, interval=0.05, press_enter=False):
        self.types.append(text)
        self.world.typed.append(text)
        st = self.world.states[self.world.state]
        if press_enter and self.world.last_clicked in st.get("on_type", {}):
            self.world.goto(st["on_type"][self.world.last_clicked])
        return True


class KeyController(FakeController):
    """Fake controller that can press keys and scroll. keys is {state: {"ctrl+s": next state}}."""
    def __init__(self, world, keys=None, scroll_to=None, **kw):
        super().__init__(world, **kw)
        self.keys, self.scroll_to, self.pressed, self.scrolled = keys or {}, scroll_to or {}, [], []

    def hotkey(self, *keys):
        self.pressed.append("+".join(keys))
        nxt = self.keys.get(self.world.state, {}).get("+".join(keys))
        if nxt:
            self.world.goto(nxt)
        return True

    def scroll(self, notches, x=None, y=None):
        self.scrolled.append((notches, x, y))
        nxt = self.scroll_to.get(self.world.state)
        if nxt:
            self.world.goto(nxt)
        return True


class FakeAgent:
    """Each item in script is the behaviour of one call: good / none / bad_id / timeout / raise"""
    def __init__(self, script, oracle, timeout_sleep=1.0):
        self.script, self.oracle, self.timeout_sleep = list(script), oracle, timeout_sleep
        self.calls = []

    def think(self, instruction, image_path, elements_dict, valid_ids=None):
        self.calls.append({"instruction": instruction, "valid_ids": list(valid_ids or [])})
        mode = self.script.pop(0) if self.script else "good"
        if mode == "none":
            return None
        if mode == "bad_id":
            return {"action": "click", "target_id": 9999, "reason": "hallucination"}
        if mode == "timeout":
            time.sleep(self.timeout_sleep)
            return None
        if mode == "raise":
            raise RuntimeError("ollama connection failed (simulated)")
        return self.oracle(instruction, elements_dict)
