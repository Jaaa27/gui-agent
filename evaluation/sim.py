"""Ideal-perception simulation: run the 20 tasks on a fake Windows desktop with the real planner / matcher / executor.

The text in each UI state is written from memory of the Chinese Windows 11 UI; OCR is assumed to be perfect and
clicks never fail. The results are therefore only a "logical upper bound": a task that fails here will certainly
fail on a real machine, while a task that passes here can still be dragged down by OCR misses, UI differences
and click offsets. Real numbers come from runner.py on a real PC.

The screen texts below are Chinese UI strings (e.g. "回收站" = Recycle Bin, "新标签页" = New Tab, "图片" = Images,
"下载" = Downloads, "文档" = Documents, "另存为" = Save as). Condition values stay in Chinese to match the
recorded CSVs (see evaluation/analyze.py: cond_label).
"""
import re
import zlib

import numpy as np

def _web_nav(text):
    # Browser address/search bar: a URL-like text opens the site, anything else is a search
    return ("site" if re.match(r"^[\w-]+(\.[\w-]+)+$", text.strip()) else "results"), text


# Each state: elements = texts on screen (order = id); click = clicking a text -> next state;
# type = typing into an input box containing a keyword -> next state (may be a function that receives the typed text)
STATES = {
    "desktop": {"elements": ["回收站", "Google Chrome", "Microsoft Edge"],
                "click": {"Google Chrome": "chrome", "Microsoft Edge": "edge"}},
    "chrome": {"elements": ["新标签页", "Gmail", "图片", "在 Google 上搜索，或输入网址"],
               "type": {"搜索": _web_nav}},
    "edge": {"elements": ["新建标签页", "搜索或输入网址"],
             "type": {"搜索": _web_nav}},
    "results": {"elements": ["{q} - 搜索", "全部", "图片", "视频", "新闻"],
                "click": {"图片": "results_images"}, "scroll_down": "results_more", "keys": {"ctrl+t": "chrome"}},
    "results_more": {"elements": ["{q} - 搜索", "相关搜索", "下一页"], "click": {}, "scroll_up": "results"},
    "results_images": {"elements": ["{q} - 图片", "全部", "图片", "视频"], "click": {}},
    "site": {"elements": ["{q}", "Wikipedia", "百度一下", "百度", "新闻"], "click": {}, "keys": {"ctrl+t": "chrome"}},
    "explorer": {"elements": ["主页", "桌面", "下载", "文档", "图片", "音乐", "视频", "此电脑"],
                 "click": {"下载": "explorer_dl", "文档": "explorer_docs"}},
    "explorer_dl": {"elements": ["下载", "桌面", "文档", "此电脑 > 下载"], "click": {}},
    "explorer_docs": {"elements": ["文档", "桌面", "下载", "此电脑 > 文档"], "click": {}},
    # Notepad: the edit area is empty and has no text, so there is no "input box" to find
    "notepad": {"elements": ["无标题 - 记事本", "文件", "编辑", "查看"],
                "click": {"文件": "notepad_file", "编辑": "notepad_edit", "查看": "notepad_view"},
                "keys": {"ctrl+s": "notepad_saveas"}},
    "notepad_saveas": {"elements": ["另存为", "文件名", "保存类型", "保存", "取消"], "click": {}},
    "notepad_file": {"elements": ["新建标签页", "新窗口", "打开", "保存", "另存为", "退出"], "click": {}},
    "notepad_edit": {"elements": ["撤消", "剪切", "复制", "粘贴", "查找", "替换"], "click": {}},
    "notepad_view": {"elements": ["缩放", "状态栏", "自动换行"], "click": {}},
    "calculator": {"elements": ["计算器", "标准", "7", "8", "9", "4", "5", "6", "1", "2", "3", "0", "="],
                   "click": {"7": "calculator_7"}},
    "calculator_7": {"elements": ["计算器", "标准", "7", "8", "9"], "click": {}},
    "notepad_calc": {"elements": ["计算器", "标准", "7", "8", "9"], "click": {}},
}
LAUNCH = {"start chrome": "chrome", "start msedge": "edge", "notepad": "notepad",
          "calc": "calculator", "explorer": "explorer", "start explorer": "explorer"}


class SimWorld:
    def __init__(self, start="desktop"):
        self.state, self.query, self.last_clicked = start, "", None

    def texts(self):
        return [t.format(q=self.query) for t in STATES[self.state]["elements"]]

    def goto(self, s):
        self.state = s

    def signature(self):
        return zlib.crc32(f"{self.state}|{self.query}".encode()) % 256

    def launch(self, cmd):
        target = LAUNCH.get(cmd)
        if target is None:
            return False
        # Opening Calculator while Notepad is open: both are on screen
        self.state = "notepad_calc" if (self.state.startswith("notepad") and target == "calculator") else target
        return True


class SimPerceiver:
    def __init__(self, world):
        self.world = world

    def scan_screen(self, save_path="sim.png"):
        els = {i: {"text": t, "center": (100 + (i % 6) * 250, 100 + (i // 6) * 120)}
               for i, t in enumerate(self.world.texts())}
        return save_path, els

    def capture_signature(self):
        return np.full((4, 4), self.world.signature(), dtype=np.float32)

    def signature_changed(self, a, b):
        return float(np.mean(np.abs(a - b)) / 255.0) > 0.01

    def invalidate(self):
        pass


class SimController:
    width, height = 1920, 1080

    def __init__(self, world):
        self.world = world

    def _element_at(self, x, y):
        for i, t in enumerate(self.world.texts()):
            cx, cy = 100 + (i % 6) * 250, 100 + (i // 6) * 120
            if abs(cx - x) <= 2 and abs(cy - y) <= 2:
                return t
        return None

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        text = self._element_at(x, y)
        self.world.last_clicked = text
        nxt = STATES[self.world.state].get("click", {}).get(text)
        if nxt:
            self.world.goto(nxt)
        return True

    def hotkey(self, *keys):
        nxt = STATES[self.world.state].get("keys", {}).get("+".join(keys))
        if nxt:
            self.world.goto(nxt)
        return True

    def scroll(self, notches, x=None, y=None):
        nxt = STATES[self.world.state].get("scroll_up" if notches > 0 else "scroll_down")
        if nxt:
            self.world.goto(nxt)
        return True

    def type_text(self, text, interval=0.05, press_enter=False):
        st = STATES[self.world.state]
        clicked = self.world.last_clicked or ""
        for key, fn in st.get("type", {}).items():
            if key in clicked and press_enter:
                nxt, q = fn(text)
                self.world.query = q
                self.world.goto(nxt)
                break
        return True


def run_sim(tasks, condition_launch, repeat=1, direct_actions=True):
    """Return a list of records (same fields as the real-machine evaluation). condition_launch=False = visual path only.
    direct_actions=False simulates the pre-week-8 version, where hotkey/scroll/wait were treated as clicks."""
    import time
    from gui_agent.executor import ExecConfig, TaskExecutor
    from gui_agent.monitor import TaskMonitor
    from gui_agent.planner import TaskPlanner
    from .record import check_expect, make_row

    # "sim-ideal perception / system launch allowed | vision only" (kept in Chinese to match recorded CSVs)
    cond = "sim-理想感知/允许系统启动" if condition_launch else "sim-理想感知/仅视觉"
    if not direct_actions:
        cond += "/无按键滚动等待"            # "no hotkey/scroll/wait"
    rows = []
    for task in tasks:
        for rep in range(1, repeat + 1):
            w = SimWorld(task.start_state)
            ex = TaskExecutor(SimPerceiver(w), SimController(w), None, TaskPlanner(),
                              TaskMonitor(trace_path=None),
                              ExecConfig(launch_enabled=condition_launch, use_llm=False,
                                         direct_actions=direct_actions),
                              launcher=lambda cmd, w=w: w.launch(cmd), sleep=lambda s: None)
            t0 = time.perf_counter()
            res = ex.run(task.instruction)
            wall = time.perf_counter() - t0
            auto = check_expect(w.texts(), task.expect)
            rows.append(make_row(task, res, condition=cond, resolution="仿真", dpi_scale="",   # "仿真" = simulation
                                 repeat=rep, auto_ok=auto, manual_ok=None, wall_s=wall,
                                 run_id=f"{cond}-{task.id}-{rep}"))
    return rows
