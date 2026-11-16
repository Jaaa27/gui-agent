import json, os, sys, tempfile, unittest, logging
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np, cv2
from tests.fakes import install_fake_gui_modules
PG, PC = install_fake_gui_modules()
from tests.cases import HARD50, HARD50_ELEMENTS
from gui_agent.matcher import (normalize, candidate_keywords, rank_candidates, detect_intent, extract_payload,
                     find_input_boxes, _partial_ratio)
from gui_agent.planner import TaskPlanner
from gui_agent.perceiver import ScreenPerceiver
from gui_agent.controller import DesktopController, DryRunController
from gui_agent.monitor import TaskMonitor, setup_logging
from gui_agent.executor import find_app_command


class FakeReader:
    """返回固定结果；记录收到的图像尺寸，用来验证降采样。bbox 按收到图像的 1/2 位置给。"""
    def __init__(self, boxes):
        self.boxes, self.seen = boxes, []
    def readtext(self, img):
        self.seen.append(img.shape[:2])
        return self.boxes(img.shape[:2])


class TestMatcher(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize(" Google_Chrome.exe "), "googlechromeexe")
    def test_keyword_tiers_keep_english_words(self):
        self.assertIn("weather", candidate_keywords("点击 Weather"))
    def test_exact_beats_contains(self):
        els = {1: {"text": "关闭窗口"}, 2: {"text": "关闭"}}
        self.assertEqual(rank_candidates("关闭", els)[0]["id"], 2)
    def test_fuzzy_ocr_typo(self):
        els = {1: {"text": "report_fina1.docx"}, 2: {"text": "other"}}
        r = rank_candidates("打开 report_final.docx", els)
        self.assertEqual((r[0]["id"], r[0]["how"]), (1, "fuzzy"))
    def test_no_match_returns_empty(self):
        self.assertEqual(rank_candidates("点击 xyz不存在", {1: {"text": "abc"}}), [])
    def test_short_keyword_no_fuzzy(self):
        self.assertEqual(rank_candidates("点击 ab", {1: {"text": "ac"}}), [])
    def test_partial_ratio(self):
        self.assertEqual(_partial_ratio("abc", "xxabcxx"), 1.0)
        self.assertEqual(_partial_ratio("", "abc"), 0.0)
    def test_intents(self):
        self.assertEqual(detect_intent("点击发送按钮"), "click")
        self.assertEqual(detect_intent("搜索 天气"), "type")
        self.assertEqual(detect_intent("搜狐"), "click")
        self.assertEqual(detect_intent("Google Chrome"), "click")
        self.assertEqual(detect_intent("type hello in the search box"), "type")
    def test_payload(self):
        self.assertEqual(extract_payload("点击输入聊天信息框，打字说 下周赛道见"), "下周赛道见")
        self.assertEqual(extract_payload("搜索 如何发送邮件"), "如何发送邮件")
        self.assertEqual(extract_payload("在输入框打字"), "")
    def test_input_box_prefers_search_for_search(self):
        els = {1: {"text": "搜索"}, 2: {"text": "Search Google or type a URL"}, 3: {"text": "输入聊天信息..."}}
        self.assertEqual(find_input_boxes("搜索 天气", els, "天气")[0]["id"], 2)
    def test_hard50_actions_all_correct(self):
        # 只检查动作类型（点击/输入）和输入文本；id 选择在 run_routing_eval 里统计
        for ins, act, tid, text in HARD50:
            self.assertEqual(detect_intent(ins), act, ins)
            if text is not None:
                self.assertEqual(extract_payload(ins), text, ins)
    def test_app_command(self):
        self.assertEqual(find_app_command("打开 Chrome")[1], "start chrome")
        self.assertIsNone(find_app_command("搜索 chrome 教程"))
        self.assertIsNone(find_app_command("点击 Weather"))


class TestPlanner(unittest.TestCase):
    def split(self, s):
        return [x.text for x in TaskPlanner().decompose(s)]
    def test_then(self):
        self.assertEqual(self.split("打开 Chrome 然后 搜索 天气"), ["打开 Chrome", "搜索 天气"])
    def test_implicit(self):
        self.assertEqual(self.split("打开 Chrome 搜索 天气"), ["打开 Chrome", "搜索 天气"])
    def test_no_split_inside_content(self):
        self.assertEqual(len(self.split("发送 我们再见")), 1)
        self.assertEqual(len(self.split("打开 发送记录.xlsx")), 1)
        self.assertEqual(len(self.split("点击输入聊天信息框，打字说 下周赛道见")), 1)
    def test_english(self):
        self.assertEqual(len(self.split("open chrome and then search weather")), 2)
    def test_max_steps(self):
        s = "然后".join(["点击 a"] * 12)
        self.assertEqual(len(TaskPlanner(max_steps=8).decompose(s)), 8)
    def test_llm_fallback_valid_and_invalid(self):
        long = "请帮我把桌面上那个很久以前建立的项目文件夹整理一下并且发给同事"
        p = TaskPlanner(llm_decompose=lambda s: ["点击 a", "点击 b"])
        self.assertEqual(len(p.decompose(long)), 2)
        p = TaskPlanner(llm_decompose=lambda s: "垃圾")
        self.assertEqual(len(p.decompose(long)), 1)
        def bad(s): raise RuntimeError
        self.assertEqual(len(TaskPlanner(llm_decompose=bad).decompose(long)), 1)


class TestPerceiver(unittest.TestCase):
    def make(self, img_hw=(2160, 3840), **kw):
        h, w = img_hw
        # 在 OCR 图中央放一个 100x40 的框
        def boxes(shape):
            sh, sw = shape
            cx, cy = sw // 2, sh // 2
            bb = [[cx - 50, cy - 20], [cx + 50, cy - 20], [cx + 50, cy + 20], [cx - 50, cy + 20]]
            return [(bb, "按钮", 0.9), (bb, "低置信", 0.1), (bb, "  ", 0.9)]
        reader = FakeReader(boxes)
        img = np.zeros((h, w, 3), np.uint8)
        p = ScreenPerceiver(reader=reader, capture_fn=lambda: img.copy(), logical_size=kw.pop("logical_size", (w, h)), **kw)
        return p, reader, img
    def test_downscale_and_coordinate_restore(self):
        p, reader, img = self.make(max_side=1600)
        path, els = p.scan_screen(os.path.join(tempfile.mkdtemp(), "o.png"))
        self.assertEqual(max(reader.seen[0]), 1600)
        self.assertEqual(list(els), [0])                     # 低置信与空文本被过滤，编号连续
        cx, cy = els[0]["center"]
        self.assertLessEqual(abs(cx - 1920), 2); self.assertLessEqual(abs(cy - 1080), 2)
    def test_no_downscale(self):
        p, reader, img = self.make(max_side=0)
        p.scan_screen(os.path.join(tempfile.mkdtemp(), "o.png"))
        self.assertEqual(reader.seen[0], (2160, 3840))
    def test_dpi_scaling(self):
        # 截图 3840x2160，但 PyAutoGUI 认为屏幕是 2560x1440（150% 缩放）
        p, _, _ = self.make(max_side=0, logical_size=(2560, 1440))
        _, els = p.scan_screen(os.path.join(tempfile.mkdtemp(), "o.png"))
        self.assertEqual(els[0]["center"], (1280, 720))
    def test_cache_hit_and_invalidate(self):
        p, reader, img = self.make(img_hw=(360, 640), max_side=0)
        tmp = os.path.join(tempfile.mkdtemp(), "o.png")
        p.scan_screen(tmp); p.scan_screen(tmp)
        self.assertEqual(len(reader.seen), 1); self.assertTrue(p.last_timing["cache_hit"])
        p.invalidate(); p.scan_screen(tmp)
        self.assertEqual(len(reader.seen), 2)
    def test_signature_change_detection(self):
        p, _, _ = self.make(img_hw=(360, 640), max_side=0)
        a = np.zeros((360, 640, 3), np.uint8); b = a.copy(); b[:, :320] = 255
        c = a.copy(); cv2.circle(c, (5, 5), 2, (255, 255, 255), -1)       # 极小的变化（像光标闪烁）
        sa = p.signature_of(a)
        self.assertTrue(p.signature_changed(sa, p.signature_of(b)))
        self.assertFalse(p.signature_changed(sa, p.signature_of(c)))
        d = a.copy(); d[100:200, 200:290] = 255                           # 弹出菜单大小的变化，只占画面 4%
        self.assertTrue(p.signature_changed(sa, p.signature_of(d)))
    def test_changed_frame_invalidates_cache(self):
        state = {"img": np.zeros((360, 640, 3), np.uint8)}
        reader = FakeReader(lambda s: [])
        p = ScreenPerceiver(reader=reader, capture_fn=lambda: state["img"].copy(), logical_size=(640, 360), max_side=0)
        tmp = os.path.join(tempfile.mkdtemp(), "o.png")
        p.scan_screen(tmp); state["img"][:] = 255; p.scan_screen(tmp)
        self.assertEqual(len(reader.seen), 2)


class TestController(unittest.TestCase):
    def setUp(self):
        PG.calls.clear(); PC.store = "旧剪贴板"
        self.c = DesktopController(move_duration=0, paste_delay=0)
    def test_click_ok_and_oob(self):
        self.assertTrue(self.c.click(10, 10)); self.assertFalse(self.c.click(5000, 10)); self.assertFalse(self.c.click(-1, 5))
    def test_ascii_uses_typewrite(self):
        self.assertTrue(self.c.type_text("abc", press_enter=True))
        names = [x[0] for x in PG.calls]
        self.assertIn("typewrite", names); self.assertEqual(names[-1], "press")
    def test_chinese_uses_clipboard_and_restores(self):
        self.assertTrue(self.c.type_text("你好"))
        self.assertIn("hotkey", [x[0] for x in PG.calls]); self.assertEqual(PC.store, "旧剪贴板")
    def test_missing_pyperclip_returns_false(self):
        saved = sys.modules.pop("pyperclip")
        sys.modules["pyperclip"] = None            # 让 import 抛 ImportError
        try:
            self.assertFalse(self.c.type_text("你好"))
        finally:
            sys.modules["pyperclip"] = saved
    def test_failsafe_is_reraised(self):
        PG.moveTo = lambda *a, **k: (_ for _ in ()).throw(PG.FailSafeException())
        with self.assertRaises(PG.FailSafeException):
            self.c.click(10, 10)
    def test_generic_exception_returns_false(self):
        PG.moveTo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        self.assertFalse(self.c.click(10, 10))
    def test_dry_run(self):
        d = DryRunController(); self.assertTrue(d.click(1, 1)); self.assertTrue(d.type_text("a"))


class TestLogging(unittest.TestCase):
    def test_trace_and_summary(self):
        from gui_agent.planner import Step
        from gui_agent.executor import Outcome, StepResult
        d = tempfile.mkdtemp(); tp = os.path.join(d, "t.jsonl")
        m = TaskMonitor(logging.getLogger("t"), trace_path=tp)
        s = Step(1, "点击 a"); m.start_task("点击 a", [s]); m.step_start(s, 1)
        m.attempt_end(s, 1, Outcome(False, "no_screen_change", timings={"perceive": 0.5}), 1.0)
        m.attempt_end(s, 2, Outcome(True, timings={"perceive": 0.4}), 1.0)
        m.step_end(s, True, 2, "", 2.0)
        summ = m.end_task(True, [StepResult(1, "点击 a", True, 2)])
        self.assertEqual((summ["attempts"], summ["retries"], summ["stage_s"]["perceive"]), (2, 1, 0.9))
        with open(tp, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        self.assertEqual([l["ev"] for l in lines], ["task_start", "step_start", "attempt", "attempt", "step_end", "task_end"])
    def test_setup_idempotent(self):
        d = tempfile.mkdtemp(); n = "uniq_logger_x"
        a = setup_logging(d, name=n); b = setup_logging(d, name=n)
        self.assertEqual(len(b.handlers), 2)
    def test_trace_write_failure_does_not_raise(self):
        m = TaskMonitor(logging.getLogger("t2"), trace_path=None)
        m.trace_path = "/proc/nonexistent_dir/t.jsonl"
        m.task_id = "x"; m._emit(ev="x")       # 不应抛出


if __name__ == "__main__":
    unittest.main(verbosity=1)
