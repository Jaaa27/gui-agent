"""Tests for the week-8 features: hotkey/scroll/wait, the post-typing check, re-planning, dynamic-region masking,
automatic downscaling, PaddleOCR output conversion, and the simulation of the extended tasks."""
import logging
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from tests.fakes import FakeController, FakePerceiver, KeyController, World, install_fake_gui_modules
PG, PC = install_fake_gui_modules()

import gui_agent.controller as controller_mod
from gui_agent.controller import DesktopController
from gui_agent.executor import ExecConfig, TaskExecutor
from gui_agent.matcher import detect_intent, parse_direct, parse_navigate, text_visible
from gui_agent.monitor import TaskMonitor
from gui_agent.perceiver import PaddleReader, ScreenPerceiver, paddle_to_easyocr
from gui_agent.planner import TaskPlanner

logging.getLogger("gui_agent").disabled = True


def make(world, ctrl=None, replanner=None, **cfg):
    base = dict(launch_enabled=False, use_llm=False)
    base.update(cfg)
    return TaskExecutor(FakePerceiver(world), ctrl or KeyController(world), None, TaskPlanner(),
                        TaskMonitor(trace_path=None), ExecConfig(**base), sleep=lambda s: None,
                        replanner=replanner)


class TestParseDirect(unittest.TestCase):
    def test_hotkeys(self):
        self.assertEqual(parse_direct("按 Ctrl+S"), ("hotkey", ["ctrl", "s"]))
        self.assertEqual(parse_direct("按下回车键"), ("hotkey", ["enter"]))
        self.assertEqual(parse_direct("press ctrl + shift + t"), ("hotkey", ["ctrl", "shift", "t"]))
        self.assertEqual(parse_direct("按 F5"), ("hotkey", ["f5"]))

    def test_button_named_like_key_is_still_click(self):
        self.assertIsNone(parse_direct("按下 搜索按钮"))
        self.assertEqual(detect_intent("按下 搜索按钮"), "click")

    def test_scroll(self):
        self.assertEqual(parse_direct("向下滚动"), ("scroll", ("down", 5)))
        self.assertEqual(parse_direct("向上滚动 3 次"), ("scroll", ("up", 3)))
        self.assertEqual(parse_direct("滚动到底部"), ("scroll", ("down", 50)))
        self.assertEqual(parse_direct("scroll up"), ("scroll", ("up", 5)))

    def test_wait(self):
        self.assertEqual(parse_direct("等待 3 秒"), ("wait", 3.0))
        self.assertEqual(parse_direct("稍等"), ("wait", 2.0))
        self.assertEqual(parse_direct("等待 999 秒"), ("wait", 30.0))

    def test_not_direct(self):
        for s in ("等待戈多", "搜索 等待", "点击 滚动条", "打开 Chrome"):
            self.assertIsNone(parse_direct(s), s)

    def test_planner_splits_new_verbs(self):
        steps = [s.text for s in TaskPlanner().decompose("打开 Chrome 然后 搜索 天气 然后 向下滚动")]
        self.assertEqual(steps, ["打开 Chrome", "搜索 天气", "向下滚动"])
        steps = [s.text for s in TaskPlanner().decompose("打开记事本 然后 按 Ctrl+S")]
        self.assertEqual(steps, ["打开记事本", "按 Ctrl+S"])

    def test_text_visible(self):
        self.assertTrue(text_visible("McMaster 选课", ["McMaster 选课 - 搜索结果"]))
        self.assertTrue(text_visible("Python 教程", ["Pyth0n 教程 - 搜索"]))      # one OCR error
        self.assertFalse(text_visible("你好", ["无标题 - 记事本", "文件"]))


class TestController(unittest.TestCase):
    # An earlier test file may have installed its own fake pyautogui; use the one controller actually imported
    def setUp(self):
        self.pg = controller_mod.pyautogui
        self.pg.calls.clear()
        pg = self.pg
        pg.moveTo = lambda *a, **k: pg.calls.append(("moveTo", a, k))   # another test may have replaced it with a raising version

    def test_hotkey_and_press(self):
        c = DesktopController(move_duration=0)
        self.assertTrue(c.hotkey("ctrl", "s"))
        self.assertTrue(c.hotkey("enter"))
        self.assertEqual([x[0] for x in self.pg.calls if x[0] != "moveTo"], ["hotkey", "press"])
        self.assertEqual(self.pg.calls[0][1], ("ctrl", "s"))
        self.assertFalse(c.hotkey())

    def test_scroll_uses_wheel_unit(self):
        c = DesktopController(move_duration=0, wheel_unit=120)
        self.assertTrue(c.scroll(-3, 960, 540))
        self.assertEqual(self.pg.calls[-1], ("scroll", (-360,), {}))
        self.assertFalse(c.scroll(-3, 5000, 5000))           # out of bounds: no move


class TestDirectActions(unittest.TestCase):
    def test_hotkey_step(self):
        w = World()
        ctrl = KeyController(w, keys={"desktop": {"ctrl+t": "edge"}})
        r = make(w, ctrl).run("按 Ctrl+T")
        self.assertTrue(r.ok)
        self.assertEqual(r.steps[0].route, "direct")
        self.assertEqual(ctrl.pressed, ["ctrl+t"])
        self.assertFalse(r.steps[0].unverified)

    def test_hotkey_without_visible_effect_is_not_retried(self):
        w = World()
        ctrl = KeyController(w)
        r = make(w, ctrl).run("按 Ctrl+V")
        self.assertTrue(r.ok)
        self.assertEqual(r.steps[0].attempts, 1)          # a retry would paste twice
        self.assertEqual(ctrl.pressed, ["ctrl+v"])
        self.assertTrue(r.steps[0].unverified)
        self.assertEqual(r.summary["unverified_steps"], 1)

    def test_scroll_goes_to_screen_center(self):
        w = World("results")
        ctrl = KeyController(w, scroll_to={"results": "desktop"})
        r = make(w, ctrl).run("向下滚动 2 次")
        self.assertTrue(r.ok)
        self.assertEqual(ctrl.scrolled, [(-2, 960, 540)])

    def test_wait(self):
        waited = []
        w = World()
        ex = make(w)
        ex.sleep = waited.append
        r = ex.run("等待 1.5 秒")
        self.assertTrue(r.ok)
        self.assertIn(1.5, waited)

    def test_controller_without_hotkey(self):
        w = World()
        r = make(w, FakeController(w)).run("按 Ctrl+S")
        self.assertFalse(r.ok)
        self.assertEqual(r.steps[0].reason, "action_failed")

    def test_ablation_treats_hotkey_as_click(self):
        w = World()
        r = make(w, direct_actions=False).run("按 Ctrl+S")
        self.assertFalse(r.ok)
        self.assertEqual(r.steps[0].reason, "no_candidate")


class NavController(KeyController):
    """Ctrl+L then typing a URL with Enter opens the "site" state, whose page shows the URL."""
    def __init__(self, world, opens=True, **kw):
        super().__init__(world, **kw)
        self.opens = opens
        world.states["site"] = {"elements": {70: {"text": "github.com/zhanz565/gui-agent", "center": (400, 60)}},
                                "on_click": {}}

    def type_text(self, text, interval=0.05, press_enter=False):
        super().type_text(text, interval, press_enter)
        if self.opens and press_enter and self.pressed[-1:] == ["ctrl+l"]:
            self.world.goto("site")
        return True


class TestNavigate(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_navigate("go to github.com/zhanz565/gui-agent"), "github.com/zhanz565/gui-agent")
        self.assertEqual(parse_navigate("visit https://www.wikipedia.org"), "https://www.wikipedia.org")
        self.assertEqual(parse_navigate("访问 www.baidu.com"), "www.baidu.com")
        self.assertEqual(detect_intent("go to github.com/x"), "navigate")
        for not_url in ("go to settings", "open notepad", "open Chrome", "go to the Images tab"):
            self.assertIsNone(parse_navigate(not_url), not_url)

    def test_planner_keeps_url_step(self):
        steps = [s.text for s in TaskPlanner().decompose(
            "open Chrome then go to github.com/zhanz565/gui-agent then scroll down")]
        self.assertEqual(steps, ["open Chrome", "go to github.com/zhanz565/gui-agent", "scroll down"])

    def test_uses_address_bar(self):
        w = World("chrome")
        ctrl = NavController(w)
        r = make(w, ctrl).run("go to github.com/zhanz565/gui-agent")
        self.assertTrue(r.ok)
        self.assertEqual(r.steps[0].route, "direct")
        self.assertEqual(ctrl.pressed, ["ctrl+l"])
        self.assertEqual(w.typed, ["github.com/zhanz565/gui-agent"])
        self.assertEqual(ctrl.clicks, [])                  # never clicks a search box on the page
        self.assertFalse(r.steps[0].unverified)

    def test_no_effect_is_retried(self):
        w = World("chrome")
        ctrl = NavController(w, opens=False)
        r = make(w, ctrl).run("go to github.com/zhanz565/gui-agent")
        self.assertFalse(r.ok)
        self.assertEqual(r.steps[0].reason, "no_screen_change")
        self.assertEqual(r.steps[0].attempts, 3)


class TestTextCheck(unittest.TestCase):
    def test_payload_visible(self):
        w = World("chrome")
        r = make(w).run("搜索 McMaster 选课")
        self.assertTrue(r.ok)
        self.assertFalse(r.steps[0].unverified)

    def test_payload_missing_marks_unverified(self):
        # The fake results page only shows "McMaster 选课 - 搜索结果", so after searching "天气" (weather) that word is not on screen
        w = World("chrome")
        r = make(w).run("搜索 天气")
        self.assertTrue(r.ok)
        self.assertTrue(r.steps[0].unverified)
        self.assertEqual(r.steps[0].attempts, 1)
        self.assertEqual(w.typed, ["天气"])                # not typed twice

    def test_can_turn_off(self):
        w = World("chrome")
        r = make(w, verify_text=False).run("搜索 天气")
        self.assertFalse(r.steps[0].unverified)


class TestReplan(unittest.TestCase):
    def test_replan_recovers(self):
        seen = {}

        def replanner(instruction, done, failed, reason, texts):
            seen.update(failed=failed, reason=reason, texts=texts)
            return ["点击 Google Chrome"]
        w = World()
        r = make(w, replanner=replanner, on_step_fail="replan", max_retries=1).run("点击 不存在的东西")
        self.assertTrue(r.ok)
        self.assertEqual(w.state, "chrome")
        self.assertEqual(r.summary["replans"], 1)
        self.assertEqual(seen["reason"], "no_candidate")
        self.assertIn("Google Chrome", seen["texts"])

    def test_bad_replan_is_ignored(self):
        w = World()
        r = make(w, replanner=lambda *a: "不是列表", on_step_fail="replan", max_retries=1).run(
            "点击 不存在的东西 然后 点击 Google Chrome")
        self.assertFalse(r.ok)
        self.assertEqual([s.reason for s in r.steps], ["no_candidate", "skipped"])

    def test_replan_only_once(self):
        calls = []

        def replanner(*a):
            calls.append(a)
            return ["点击 还是不存在"]
        w = World()
        r = make(w, replanner=replanner, on_step_fail="replan", max_retries=1).run("点击 不存在的东西")
        self.assertFalse(r.ok)
        self.assertEqual(len(calls), 1)


class TestPerceiverWeek8(unittest.TestCase):
    def frames(self, seq):
        it = iter(seq)
        return lambda: next(it)

    def test_calibrate_masks_dynamic_area(self):
        rng = np.random.default_rng(0)
        base = np.full((360, 640, 3), 200, np.uint8)
        vids = []
        for _ in range(10):
            f = base.copy()
            f[40:120, 400:560] = rng.integers(0, 256, (1, 1, 3), dtype=np.uint8)   # a "video" that changes colour as a block
            vids.append(f)
        p = ScreenPerceiver(reader=object(), capture_fn=self.frames(vids))
        sa, sb = p.signature_of(vids[0]), p.signature_of(vids[1])
        self.assertTrue(p.signature_changed(sa, sb))
        n = p.calibrate(frames=8, sleep=lambda s: None)
        self.assertGreater(n, 0)
        self.assertFalse(p.signature_changed(sa, p.signature_of(vids[9])))
        popup = vids[9].copy()
        popup[200:300, 50:150] = 0                               # a window pops up outside the video region
        self.assertTrue(p.signature_changed(sa, p.signature_of(popup)))

    def test_ignore_regions(self):
        p = ScreenPerceiver(reader=object(), ignore_regions=[(0.9, 0.9, 1.0, 1.0)])
        a = np.zeros((360, 640, 3), np.uint8)
        b = a.copy()
        b[330:360, 580:640] = 255                                # clock in the bottom-right corner
        self.assertFalse(p.signature_changed(p.signature_of(a), p.signature_of(b)))

    def test_auto_max_side(self):
        p = ScreenPerceiver(reader=object(), ui_scale=1.0)
        self.assertEqual(p.auto_max_side((1080, 1920, 3)), 0)       # no downscaling at 100%
        p = ScreenPerceiver(reader=object(), ui_scale=1.5)
        self.assertEqual(p.auto_max_side((2160, 3840, 3)), 2774)    # 18 px text shrinks to 13 px
        p = ScreenPerceiver(reader=object(), ui_scale=1.25)
        self.assertEqual(p.auto_max_side((1080, 1920, 3)), 1664)

    def test_paddle_v2_format(self):
        box = [[0, 0], [10, 0], [10, 5], [0, 5]]
        out = paddle_to_easyocr([[[box, ("文件", 0.98)], [box, ("编辑", 0.9)]]])
        self.assertEqual([o[1] for o in out], ["文件", "编辑"])
        self.assertEqual(paddle_to_easyocr([None]), [])

    def test_paddle_v3_format(self):
        page = {"rec_texts": ["搜索"], "rec_scores": [0.95],
                "rec_polys": [np.array([[1, 2], [30, 2], [30, 12], [1, 12]])]}
        out = paddle_to_easyocr([page])
        self.assertEqual(out[0][1:], ("搜索", 0.95))
        self.assertEqual(out[0][0][2], [30.0, 12.0])

    def test_paddle_reader_plugs_into_perceiver(self):
        class FakePaddle:
            def ocr(self, img):
                return [[[[[10, 10], [60, 10], [60, 30], [10, 30]], ("设置", 0.9)]]]
        img = np.zeros((360, 640, 3), np.uint8)
        p = ScreenPerceiver(reader=PaddleReader(ocr=FakePaddle()), capture_fn=lambda: img.copy(),
                            max_side=0, logical_size=(640, 360))
        _, els = p.scan_screen(os.path.join(ROOT, "results", "_tmp_paddle.png"))
        os.remove(os.path.join(ROOT, "results", "_tmp_paddle.png"))
        self.assertEqual(els[0]["text"], "设置")
        self.assertEqual(els[0]["center"], (35, 20))


class TestExtSim(unittest.TestCase):
    def test_ext_tasks(self):
        from evaluation.sim import run_sim
        from evaluation.tasks import EXT_TASKS
        new = run_sim(EXT_TASKS, True)
        old = run_sim(EXT_TASKS, True, direct_actions=False)
        self.assertEqual(sum(r["final_ok"] for r in new), len(EXT_TASKS))
        self.assertEqual(sum(r["final_ok"] for r in old), 0)


if __name__ == "__main__":
    unittest.main()
