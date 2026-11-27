"""Task executor: decompose -> run step by step -> verify -> retry on failure.

One attempt at a step is: (launch via a system command if possible) -> perceive ->
decide (rules first, LLM as fallback) -> act -> check that the screen changed.
Failures are classified, and each retry handles its cause:

  perception error / no text found   wait and rescan (the page may still be loading)
  no rule candidate                  ask the LLM
  LLM gave nothing / bad ID / timed out  ask again, removing elements already tried
  controller returned failure        retry the same target; switch after two failures
  clicked but the screen didn't change   blacklist the target and try the next candidate

Once every candidate has been tried, the blacklist is cleared, because some clicks
only fail intermittently.

"The screen changed" does not mean "the right thing was clicked"; it only catches
clicks that hit nothing. So after a type step the screen is scanned again to check
that the typed text appears. If it doesn't, the step is marked unverified but NOT
retried, since a retry would type the text twice.

Hotkeys, scrolling and waiting need no target and run directly (route=direct).
Hotkeys and scrolling don't always change the screen, so no change is also only
marked unverified, so that something like Ctrl+V is never executed twice.

What happens after a step finally fails is set by on_step_fail: "abort" stops,
"continue" carries on, and "replan" hands the current screen text and the failure
reason to a replanner (usually the LLM) to plan the remaining steps.
"""
import concurrent.futures
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from .matcher import (candidate_keywords, detect_intent, extract_payload, find_input_boxes,
                      parse_direct, rank_candidates, text_visible)
from .monitor import TaskMonitor
from .planner import Step, TaskPlanner

# Failure reasons
PERCEPTION_ERROR = "perception_error"
NO_ELEMENTS = "no_elements"
NO_CANDIDATE = "no_candidate"
LLM_NO_OUTPUT = "llm_no_output"
LLM_INVALID = "llm_invalid"
LLM_TIMEOUT = "llm_timeout"
ACTION_FAILED = "action_failed"
NO_SCREEN_CHANGE = "no_screen_change"
UNEXPECTED = "unexpected"
BUDGET = "budget_exceeded"
SKIPPED = "skipped"

# Apps that can be launched directly with a system command (Windows).
# Keys are names after matcher.normalize(); the Chinese keys are the Chinese app names.
APP_COMMANDS = {
    "chrome": "start chrome", "googlechrome": "start chrome", "谷歌浏览器": "start chrome",
    "edge": "start msedge", "microsoftedge": "start msedge",
    "记事本": "notepad", "notepad": "notepad",
    "计算器": "calc", "calculator": "calc", "calc": "calc",
    "文件资源管理器": "explorer", "资源管理器": "explorer", "fileexplorer": "explorer",
}


def find_app_command(instruction):
    """Return a launch command only if the intent is "click" and a keyword is exactly an app name.

    v1 used `"chrome" in instruction`, so "search for chrome extensions" also launched Chrome.
    """
    if detect_intent(instruction) != "click":
        return None
    for kw in candidate_keywords(instruction):
        if kw in APP_COMMANDS:
            return kw, APP_COMMANDS[kw]
    return None


def default_launcher(cmd):
    """Start the command and return immediately.

    The exit code is not used: explorer.exe returns 1 even when the window opens
    (the real-machine evaluation caught this - every "open File Explorer" fell back to
    the visual path). Whether the launch worked is decided by the screen-change check
    in TaskExecutor instead.
    """
    try:
        subprocess.Popen(cmd, shell=True)
        return True
    except Exception:
        return False


@dataclass
class ExecConfig:
    max_retries: int = 3
    backoff: tuple = (0.5, 1.0, 2.0)
    settle_delay: float = 0.8        # wait for the UI to settle before comparing screens
    focus_delay: float = 0.5         # wait after clicking an input box before typing
    launch_delay: float = 2.0
    step_gap: float = 0.5
    verify: bool = True
    use_llm: bool = True
    llm_timeout: float = 60.0
    launch_enabled: bool = (os.name == "nt")
    on_step_fail: str = "abort"      # abort: skip the remaining steps; continue: keep going; replan: re-plan
    max_replans: int = 1
    task_budget_s: float = 300.0
    direct_actions: bool = True      # run hotkey/scroll/wait directly; off = treat them as clicks (pre-week-8 behaviour)
    verify_text: bool = True         # after typing, rescan and check that the text appears


@dataclass
class Outcome:
    ok: bool
    reason: str = ""
    route: str = ""
    detail: str = ""
    target_id: Optional[int] = None
    timings: dict = field(default_factory=dict)
    unverified: bool = False        # the action ran, but nothing confirmed it worked


@dataclass
class StepResult:
    index: int
    text: str
    ok: bool
    attempts: int
    reason: str = ""
    route: str = ""
    elapsed: float = 0.0
    unverified: bool = False


@dataclass
class TaskResult:
    instruction: str
    ok: bool
    steps: list
    summary: dict


class _LLMTimeout(Exception):
    pass


class TaskExecutor:
    def __init__(self, perceiver, controller, agent=None, planner=None, monitor=None,
                 config=None, launcher=default_launcher, sleep=time.sleep, replanner=None):
        self.perceiver = perceiver
        self.controller = controller
        self.agent = agent
        self.planner = planner or TaskPlanner()
        self.monitor = monitor or TaskMonitor(trace_path=None)
        self.cfg = config or ExecConfig()
        self.launcher = launcher
        self.sleep = sleep          # tests pass a no-op so they don't actually wait
        self.replanner = replanner  # (instruction, done steps, failed step, reason, screen texts) -> [step texts]
        self.log = self.monitor.log
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _signature(self):
        fn = getattr(self.perceiver, "capture_signature", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as e:
            self.log.debug("failed to read screen signature: %s", e)
            return None

    def _changed(self, a, b):
        fn = getattr(self.perceiver, "signature_changed", None)
        return True if fn is None else bool(fn(a, b))

    def _call_llm(self, instruction, img, elements, valid_ids):
        fut = self._pool.submit(self.agent.think, instruction, img, elements, valid_ids)
        try:
            return fut.result(timeout=self.cfg.llm_timeout)
        except concurrent.futures.TimeoutError:
            # A stuck thread can't be killed; use a fresh pool so later calls don't queue behind it
            self._pool.shutdown(wait=False)
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            raise _LLMTimeout()

    def _screen_texts(self):
        inval = getattr(self.perceiver, "invalidate", None)
        if inval:
            inval()
        _, elements = self.perceiver.scan_screen()
        return [str(v.get("text", "")) for v in elements.values()]

    def _direct(self, kind, arg, tm):
        """Hotkey / scroll / wait / navigate."""
        cfg, ctrl = self.cfg, self.controller
        if kind == "wait":
            self.sleep(arg)
            return Outcome(True, route="direct", detail=f"wait {arg:g}s", timings=tm)
        if kind == "navigate":
            return self._navigate(arg, tm)
        fn = getattr(ctrl, "hotkey" if kind == "hotkey" else "scroll", None)
        if fn is None:
            return Outcome(False, ACTION_FAILED, route="direct", detail=f"controller does not support {kind}", timings=tm)
        before = self._signature()
        t0 = time.perf_counter()
        if kind == "hotkey":
            ok, detail = fn(*arg), "+".join(arg)
        else:
            direction, n = arg
            w, h = getattr(ctrl, "width", None), getattr(ctrl, "height", None)
            x, y = (w // 2, h // 2) if w and h else (None, None)     # scroll the window in the middle of the screen
            ok, detail = fn(n if direction == "up" else -n, x, y), f"{direction} {n}"
        tm["act"] = time.perf_counter() - t0
        inval = getattr(self.perceiver, "invalidate", None)
        if inval:
            inval()
        if ok is False:
            return Outcome(False, ACTION_FAILED, route="direct", detail=detail, timings=tm)
        unverified = False
        if cfg.verify:
            self.sleep(cfg.settle_delay)
            after = self._signature()
            if before is not None and after is not None and not self._changed(before, after):
                # Happens when already scrolled to the end, or the hotkey has no visible effect; don't retry
                self.log.info("  screen unchanged after %s, marked unverified", detail)
                unverified = True
        return Outcome(True, route="direct", detail=detail, timings=tm, unverified=unverified)

    def _navigate(self, url, tm):
        """Focus the browser address bar with Ctrl+L, type the URL and press Enter."""
        cfg, ctrl = self.cfg, self.controller
        if not (hasattr(ctrl, "hotkey") and hasattr(ctrl, "type_text")):
            return Outcome(False, ACTION_FAILED, route="direct", detail="controller does not support navigate", timings=tm)
        before = self._signature()
        t0 = time.perf_counter()
        ok = ctrl.hotkey("ctrl", "l") is not False
        if ok:
            self.sleep(cfg.focus_delay)
            ok = ctrl.type_text(url, press_enter=True) is not False
        tm["act"] = time.perf_counter() - t0
        inval = getattr(self.perceiver, "invalidate", None)
        if inval:
            inval()
        detail = f"navigate {url}"
        if not ok:
            return Outcome(False, ACTION_FAILED, route="direct", detail=detail, timings=tm)
        if not cfg.verify:
            return Outcome(True, route="direct", detail=detail, timings=tm)
        self.sleep(cfg.settle_delay)
        after = self._signature()
        if before is not None and after is not None and not self._changed(before, after):
            # Nothing happened at all (e.g. no browser window in front). Retrying is safe: Ctrl+L selects the
            # whole address bar, so the URL replaces whatever was typed before.
            return Outcome(False, NO_SCREEN_CHANGE, route="direct", detail=detail, timings=tm)
        unverified = False
        if cfg.verify_text:
            try:
                if not text_visible(url.split("://")[-1], self._screen_texts()):
                    self.log.warning("  %r not found on screen after navigating, marked unverified", url)
                    unverified = True
            except Exception as e:
                self.log.debug("post-navigation check failed: %s", e)
        return Outcome(True, route="direct", detail=detail, timings=tm, unverified=unverified)

    def _attempt(self, step, st):
        cfg, text, tm = self.cfg, step.text, {}

        if cfg.direct_actions:
            direct = parse_direct(text)
            if direct:
                return self._direct(direct[0], direct[1], tm)

        # Try launching the app with a system command first; fall back to the visual path below
        if cfg.launch_enabled and not st["launch_tried"]:
            app = find_app_command(text)
            if app:
                st["launch_tried"] = True
                before = self._signature()
                ok = self.launcher(app[1])
                if ok:
                    self.sleep(cfg.launch_delay)
                    after = self._signature()
                    if (not cfg.verify or before is None or after is None or self._changed(before, after)):
                        return Outcome(True, route="tool_use", detail=app[1])
                    self.log.warning("launch command %r ran but the screen did not change; using the visual path", app[1])
                else:
                    self.log.warning("launch command %r failed; using the visual path", app[1])

        # Perceive
        before = self._signature()
        t0 = time.perf_counter()
        try:
            img, elements = self.perceiver.scan_screen()
        except Exception as e:
            return Outcome(False, PERCEPTION_ERROR, detail=repr(e))
        tm["perceive"] = time.perf_counter() - t0
        if not elements:
            return Outcome(False, NO_ELEMENTS, timings=tm)

        # Decide: rules first
        t0 = time.perf_counter()
        intent = detect_intent(text)
        if intent not in ("click", "type"):
            intent = "click"          # with direct_actions off, treat it as a click (pre-week-8 behaviour)
        payload = extract_payload(text) if intent == "type" else ""
        cands = find_input_boxes(text, elements, payload) if intent == "type" else rank_candidates(text, elements)
        usable = [c for c in cands if c["id"] not in st["tried"]]
        if cands and not usable:
            self.log.info("every candidate has been tried; clearing the blacklist and retrying the best one")
            st["tried"].clear()
            usable = cands

        route = "rule"
        if usable:
            top = usable[0]
            decision = {"action": intent, "target_id": top["id"], "reason": top["how"]}
            if intent == "type":
                decision["text"] = payload
                if not payload:
                    self.log.warning("no text to type was found in the instruction; the input box will only be clicked")
        elif cfg.use_llm and self.agent is not None:
            route = "llm"
            # Texts of 30+ characters are almost always logs or paragraphs; drop them, and drop elements already tried
            pruned = {int(k): v for k, v in elements.items()
                      if 0 < len(str(v.get("text", ""))) < 30 and int(k) not in st["tried"]}
            if not pruned:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, NO_CANDIDATE, route=route, detail="no usable elements left after pruning", timings=tm)
            # The hint is in Chinese to match the Chinese system prompt in agent.py:
            # "Note: elements [...] were already tried with no effect; do not pick them again."
            hint = f"\n（注意：元素 {sorted(st['tried'])} 已经试过没有效果，不要再选。）" if st["tried"] else ""
            try:
                raw = self._call_llm(text + hint, img, pruned, [str(k) for k in pruned])
            except _LLMTimeout:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, LLM_TIMEOUT, route=route, detail=f">{cfg.llm_timeout}s", timings=tm)
            except Exception as e:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, LLM_NO_OUTPUT, route=route, detail=repr(e), timings=tm)
            if not isinstance(raw, dict) or "action" not in raw or "target_id" not in raw:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, LLM_NO_OUTPUT, route=route, timings=tm)
            try:
                tid = int(raw["target_id"])
            except (TypeError, ValueError):
                tid = None
            if raw["action"] not in ("click", "type") or tid not in pruned:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, LLM_INVALID, route=route, target_id=tid,
                               detail=f"action={raw.get('action')} target_id={raw.get('target_id')}", timings=tm)
            decision = {"action": raw["action"], "target_id": tid, "reason": "llm"}
            if raw["action"] == "type":
                decision["text"] = raw.get("text") or payload
        else:
            tm["decide"] = time.perf_counter() - t0
            return Outcome(False, NO_CANDIDATE, route=route, timings=tm)
        tm["decide"] = time.perf_counter() - t0

        # Act. Check once more that the ID is really in the dictionary so we never click at random
        tid = decision["target_id"]
        if tid not in elements:
            return Outcome(False, LLM_INVALID, route=route, target_id=tid, detail="id not in the element dictionary", timings=tm)
        x, y = (int(round(v)) for v in elements[tid]["center"])
        t0 = time.perf_counter()
        try:
            if decision["action"] == "click":
                n = 2 if re.search(r"双击|double\s*click", text, re.I) else 1
                ok = self.controller.click(x, y, num_of_clicks=n) if n > 1 else self.controller.click(x, y)
            else:
                ok = self.controller.click(x, y)
                if ok is not False:
                    self.sleep(cfg.focus_delay)
                    ok = self.controller.type_text(decision.get("text", ""), press_enter=True)
        except Exception as e:
            if type(e).__name__ == "FailSafeException":
                raise                          # emergency stop - never retry
            tm["act"] = time.perf_counter() - t0
            return Outcome(False, ACTION_FAILED, route=route, target_id=tid, detail=repr(e), timings=tm)
        tm["act"] = time.perf_counter() - t0
        inval = getattr(self.perceiver, "invalidate", None)
        if inval:
            inval()
        if ok is False:
            st["fail_count"][tid] = st["fail_count"].get(tid, 0) + 1
            if st["fail_count"][tid] >= 2:      # switch target after two failures in a row
                st["tried"].add(tid)
            return Outcome(False, ACTION_FAILED, route=route, target_id=tid,
                           detail=f"coords ({x},{y})", timings=tm)

        # Verify: did the screen signature change enough?
        if cfg.verify:
            t0 = time.perf_counter()
            self.sleep(cfg.settle_delay)
            after = self._signature()
            tm["verify"] = time.perf_counter() - t0
            if before is not None and after is not None and not self._changed(before, after):
                st["tried"].add(tid)
                return Outcome(False, NO_SCREEN_CHANGE, route=route, target_id=tid, timings=tm)

        # Type steps: a changed screen is not enough; check the typed text is actually on screen
        unverified = False
        typed = decision.get("text", "") if decision["action"] == "type" else ""
        if cfg.verify_text and typed:
            t0 = time.perf_counter()
            try:
                if not text_visible(typed, self._screen_texts()):
                    unverified = True
                    self.log.warning("  typed text %r not found on screen, marked unverified", typed)
            except Exception as e:
                self.log.debug("post-typing check failed: %s", e)
            tm["text_check"] = time.perf_counter() - t0
        return Outcome(True, route=route, target_id=tid, timings=tm, unverified=unverified)

    def run_step(self, step, total):
        cfg = self.cfg
        self.monitor.step_start(step, total)
        st = {"tried": set(), "fail_count": {}, "launch_tried": False}
        t_step = time.perf_counter()
        last = Outcome(False, UNEXPECTED)
        attempts = 0
        for attempt in range(1, cfg.max_retries + 1):
            attempts = attempt
            t_a = time.perf_counter()
            try:
                last = self._attempt(step, st)
            except Exception as e:
                if type(e).__name__ == "FailSafeException":
                    raise
                last = Outcome(False, UNEXPECTED, detail=repr(e))
            self.monitor.attempt_end(step, attempt, last, time.perf_counter() - t_a)
            if last.ok:
                break
            if attempt < cfg.max_retries:
                self.sleep(cfg.backoff[min(attempt - 1, len(cfg.backoff) - 1)])
        elapsed = time.perf_counter() - t_step
        self.monitor.step_end(step, last.ok, attempts, last.reason, elapsed)
        return StepResult(step.index, step.text, last.ok, attempts, "" if last.ok else last.reason,
                          last.route, elapsed, unverified=last.ok and last.unverified)

    def _replan(self, instruction, results, failed):
        """Ask the replanner for the remaining steps given the current screen. Invalid results are discarded."""
        try:
            texts = self._screen_texts()
        except Exception:
            texts = []
        try:
            cand = self.replanner(instruction, [r.text for r in results if r.ok], failed.text,
                                  failed.reason, texts)
        except Exception as e:
            self.log.warning("re-planning failed: %s", e)
            return None
        if (isinstance(cand, list) and 1 <= len(cand) <= self.planner.max_steps
                and all(isinstance(c, str) and c.strip() for c in cand)):
            return [c.strip() for c in cand]
        self.log.warning("re-planning returned an invalid result, ignoring it: %r", cand)
        return None

    def run(self, instruction):
        cfg = self.cfg
        t0 = time.perf_counter()
        steps = self.planner.decompose(instruction)
        self.monitor.start_task(instruction, steps)
        results, ok_all, replans = [], True, 0
        recovered_from = None        # after a re-plan, only the new plan's steps decide success
        i = 0
        while i < len(steps):
            step = steps[i]
            if time.perf_counter() - t0 > cfg.task_budget_s:
                self.log.error("task exceeded its %.0fs time budget; skipping the remaining steps", cfg.task_budget_s)
                results += [StepResult(s.index, s.text, False, 0, BUDGET) for s in steps[i:]]
                ok_all = False
                break
            sr = self.run_step(step, len(steps))
            results.append(sr)
            if sr.ok:
                if i < len(steps) - 1:
                    self.sleep(cfg.step_gap)
                i += 1
                continue
            ok_all = False
            if (cfg.on_step_fail == "replan" and self.replanner is not None
                    and replans < cfg.max_replans):
                new = self._replan(instruction, results, sr)
                if new:
                    replans += 1
                    self.log.info("step %d failed; re-planned %d new steps: %s", step.index, len(new), new)
                    steps = steps[:i + 1] + [Step(step.index + 1 + j, t) for j, t in enumerate(new)]
                    recovered_from = len(results)
                    i += 1
                    continue
            if cfg.on_step_fail in ("abort", "replan"):
                self.log.error("step %d failed; skipping the remaining %d step(s)", step.index, len(steps) - i - 1)
                results += [StepResult(s.index, s.text, False, 0, SKIPPED) for s in steps[i + 1:]]
                break
            i += 1
        if recovered_from is not None and len(results) > recovered_from:
            ok_all = all(r.ok for r in results[recovered_from:])
        summary = self.monitor.end_task(ok_all, [r for r in results if r.reason != SKIPPED])
        summary["replans"] = replans
        return TaskResult(instruction, ok_all, results, summary)
