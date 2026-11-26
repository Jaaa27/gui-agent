import logging
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.analyze import analyze, wilson
from evaluation.record import check_expect
from evaluation.sim import run_sim
from evaluation.tasks import TASKS
from gui_agent.planner import TaskPlanner


class TestTasks(unittest.TestCase):
    def test_counts_and_levels(self):
        self.assertEqual(len(TASKS), 20)
        self.assertEqual(len({t.id for t in TASKS}), 20)
        self.assertEqual([sum(1 for t in TASKS if t.level == l) for l in (1, 2, 3)], [6, 8, 6])

    def test_planner_splits_as_declared(self):
        p = TaskPlanner()
        for t in TASKS:
            self.assertEqual([s.text for s in p.decompose(t.instruction)], t.steps, t.id)

    def test_expect_groups(self):
        self.assertTrue(check_expect(["无标题 - 记事本"], [["记事本", "Notepad"]]))
        self.assertFalse(check_expect(["计算器"], [["计算器"], ["记事本"]]))
        self.assertTrue(check_expect(["NOTEPAD"], [["notepad"]]))


class TestAnalysis(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(10, 20)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(wilson(0, 0), (0.0, 0.0))
        self.assertAlmostEqual(wilson(20, 20)[1], 1.0, places=6)

    def test_sim_numbers(self):
        logging.getLogger("gui_agent").disabled = True
        rows = run_sim(TASKS, True) + run_sim(TASKS, False)
        a = analyze(rows)
        by = {k: v["success"] for k, v in a["by_condition"].items()}
        self.assertEqual(sorted(by.values()), [10, 19])
        # T17 (typing into the blank edit area) is expected to fail
        # "允许系统启动" = system launch allowed (condition value recorded by sim.py)
        failed = {r["task_id"] for r in rows if r["condition"].endswith("允许系统启动") and not r["final_ok"]}
        self.assertEqual(failed, {"T17"})


if __name__ == "__main__":
    unittest.main()
