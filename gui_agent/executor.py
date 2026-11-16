"""任务执行器：拆解 -> 逐步执行 -> 验证 -> 失败重试。

单步的一次尝试大致是：（能用系统命令启动就先试）-> 感知 -> 决策（规则优先，
大模型兜底）-> 执行 -> 看画面有没有变化。失败会归类成几种原因，重试时对症处理：

  感知异常 / 没识别到文字   等一会儿再扫（页面可能还在加载）
  规则没候选               交给大模型
  大模型没输出/ID 不合法/超时  再问一次，试过的元素从字典里拿掉
  控制器返回失败            同一个目标再试，连着失败两次就换目标
  点了但画面没变            这个目标进黑名单，下次换下一个候选

候选全试过一遍就清空黑名单重来，有些点击只是偶尔没生效。
注意"画面变了"不等于"点对了"，这个验证只能挡住点空的情况。所以输入类步骤做完后
还会再扫一次屏幕，看输入的内容有没有出现；没出现只标记 unverified，不重试（重试会重复输入）。

按键、滚动、等待这三种动作不需要找目标，直接执行（route=direct）。按键和滚动不一定改变画面，
画面没变也不重试，只标记 unverified，免得 Ctrl+V 之类的被执行两次。

步骤最终失败后的处理由 on_step_fail 决定：abort 停止、continue 接着做、
replan 把当前屏幕上的文字和失败原因交给 replanner（一般是大模型）重新规划剩下的步骤。
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

# 失败原因
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

# 可以直接用系统命令启动的应用（Windows）。key 是 normalize 之后的名字
APP_COMMANDS = {
    "chrome": "start chrome", "googlechrome": "start chrome", "谷歌浏览器": "start chrome",
    "edge": "start msedge", "microsoftedge": "start msedge",
    "记事本": "notepad", "notepad": "notepad",
    "计算器": "calc", "calculator": "calc", "calc": "calc",
    "文件资源管理器": "explorer", "资源管理器": "explorer", "fileexplorer": "explorer",
}


def find_app_command(instruction):
    """点击意图 + 关键词正好是某个应用名，才返回启动命令。

    以前是 "chrome" in instruction，"搜索 chrome 插件"也会被当成启动 Chrome。
    """
    if detect_intent(instruction) != "click":
        return None
    for kw in candidate_keywords(instruction):
        if kw in APP_COMMANDS:
            return kw, APP_COMMANDS[kw]
    return None


def default_launcher(cmd):
    try:
        return subprocess.run(cmd, shell=True, timeout=10).returncode == 0
    except Exception:
        return False


@dataclass
class ExecConfig:
    max_retries: int = 3
    backoff: tuple = (0.5, 1.0, 2.0)
    settle_delay: float = 0.8        # 动作做完等界面稳定再比较画面
    focus_delay: float = 0.5         # 点完输入框等一会儿再打字
    launch_delay: float = 2.0
    step_gap: float = 0.5
    verify: bool = True
    use_llm: bool = True
    llm_timeout: float = 60.0
    launch_enabled: bool = (os.name == "nt")
    on_step_fail: str = "abort"      # abort：后面的步骤不做了；continue：继续做；replan：重新规划
    max_replans: int = 1
    task_budget_s: float = 300.0
    direct_actions: bool = True      # 按键/滚动/等待直接执行；关掉就和第 7 周之前一样当成点击
    verify_text: bool = True         # 输入后再扫一次屏，看输入的内容在不在


@dataclass
class Outcome:
    ok: bool
    reason: str = ""
    route: str = ""
    detail: str = ""
    target_id: Optional[int] = None
    timings: dict = field(default_factory=dict)
    unverified: bool = False        # 动作做了，但没法确认做对了


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
        self.sleep = sleep          # 测试里换成空函数，免得真的等
        self.replanner = replanner  # (指令, 已完成步骤, 失败步骤, 失败原因, 屏幕文字) -> [步骤文字]
        self.log = self.monitor.log
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _signature(self):
        fn = getattr(self.perceiver, "capture_signature", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as e:
            self.log.debug("读取画面指纹失败: %s", e)
            return None

    def _changed(self, a, b):
        fn = getattr(self.perceiver, "signature_changed", None)
        return True if fn is None else bool(fn(a, b))

    def _call_llm(self, instruction, img, elements, valid_ids):
        fut = self._pool.submit(self.agent.think, instruction, img, elements, valid_ids)
        try:
            return fut.result(timeout=self.cfg.llm_timeout)
        except concurrent.futures.TimeoutError:
            # 卡住的线程杀不掉，换个新线程池，不然后面的调用要排在它后面
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
        """按键 / 滚动 / 等待。"""
        cfg, ctrl = self.cfg, self.controller
        if kind == "wait":
            self.sleep(arg)
            return Outcome(True, route="direct", detail=f"等待 {arg:g}s", timings=tm)
        fn = getattr(ctrl, "hotkey" if kind == "hotkey" else "scroll", None)
        if fn is None:
            return Outcome(False, ACTION_FAILED, route="direct", detail=f"控制器不支持 {kind}", timings=tm)
        before = self._signature()
        t0 = time.perf_counter()
        if kind == "hotkey":
            ok, detail = fn(*arg), "+".join(arg)
        else:
            direction, n = arg
            w, h = getattr(ctrl, "width", None), getattr(ctrl, "height", None)
            x, y = (w // 2, h // 2) if w and h else (None, None)     # 滚屏幕中间那个窗口
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
                # 已经滚到底、或者快捷键本来就没有可见效果，都会这样；不重试
                self.log.info("  %s 之后画面没变化，标记为未确认", detail)
                unverified = True
        return Outcome(True, route="direct", detail=detail, timings=tm, unverified=unverified)

    def _attempt(self, step, st):
        cfg, text, tm = self.cfg, step.text, {}

        if cfg.direct_actions:
            direct = parse_direct(text)
            if direct:
                return self._direct(direct[0], direct[1], tm)

        # 先试系统命令启动应用，失败了就走下面的视觉流程
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
                    self.log.warning("启动命令返回成功但画面没变化: %s，改走视觉路径", app[1])
                else:
                    self.log.warning("启动命令失败: %s，改走视觉路径", app[1])

        # 感知
        before = self._signature()
        t0 = time.perf_counter()
        try:
            img, elements = self.perceiver.scan_screen()
        except Exception as e:
            return Outcome(False, PERCEPTION_ERROR, detail=repr(e))
        tm["perceive"] = time.perf_counter() - t0
        if not elements:
            return Outcome(False, NO_ELEMENTS, timings=tm)

        # 决策：规则优先
        t0 = time.perf_counter()
        intent = detect_intent(text)
        if intent not in ("click", "type"):
            intent = "click"          # direct_actions 关掉时，按第 7 周之前的做法当成点击
        payload = extract_payload(text) if intent == "type" else ""
        cands = find_input_boxes(text, elements, payload) if intent == "type" else rank_candidates(text, elements)
        usable = [c for c in cands if c["id"] not in st["tried"]]
        if cands and not usable:
            self.log.info("候选都试过一遍了，清空黑名单再试最优的")
            st["tried"].clear()
            usable = cands

        route = "rule"
        if usable:
            top = usable[0]
            decision = {"action": intent, "target_id": top["id"], "reason": top["how"]}
            if intent == "type":
                decision["text"] = payload
                if not payload:
                    self.log.warning("指令里没抠出要输入的内容，只会点一下输入框")
        elif cfg.use_llm and self.agent is not None:
            route = "llm"
            # 超过 30 个字的基本是日志或段落，丢掉；试过的元素也去掉
            pruned = {int(k): v for k, v in elements.items()
                      if 0 < len(str(v.get("text", ""))) < 30 and int(k) not in st["tried"]}
            if not pruned:
                tm["decide"] = time.perf_counter() - t0
                return Outcome(False, NO_CANDIDATE, route=route, detail="剪枝后没有可用元素", timings=tm)
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

        # 执行。动手之前再确认一次 id 真的在字典里，防止乱点
        tid = decision["target_id"]
        if tid not in elements:
            return Outcome(False, LLM_INVALID, route=route, target_id=tid, detail="id 不在元素字典中", timings=tm)
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
                raise                          # 紧急停止，别重试
            tm["act"] = time.perf_counter() - t0
            return Outcome(False, ACTION_FAILED, route=route, target_id=tid, detail=repr(e), timings=tm)
        tm["act"] = time.perf_counter() - t0
        inval = getattr(self.perceiver, "invalidate", None)
        if inval:
            inval()
        if ok is False:
            st["fail_count"][tid] = st["fail_count"].get(tid, 0) + 1
            if st["fail_count"][tid] >= 2:      # 同一个目标连着失败两次就换一个
                st["tried"].add(tid)
            return Outcome(False, ACTION_FAILED, route=route, target_id=tid,
                           detail=f"坐标({x},{y})", timings=tm)

        # 验证：前后画面指纹差得够不够大
        if cfg.verify:
            t0 = time.perf_counter()
            self.sleep(cfg.settle_delay)
            after = self._signature()
            tm["verify"] = time.perf_counter() - t0
            if before is not None and after is not None and not self._changed(before, after):
                st["tried"].add(tid)
                return Outcome(False, NO_SCREEN_CHANGE, route=route, target_id=tid, timings=tm)

        # 输入类：画面变了还不够，看输入的内容是不是真的出现在屏幕上
        unverified = False
        typed = decision.get("text", "") if decision["action"] == "type" else ""
        if cfg.verify_text and typed:
            t0 = time.perf_counter()
            try:
                if not text_visible(typed, self._screen_texts()):
                    unverified = True
                    self.log.warning("  输入后没在屏幕上找到 %r，标记为未确认", typed)
            except Exception as e:
                self.log.debug("输入后的复查失败: %s", e)
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
        """让 replanner 根据当前屏幕重新规划剩下的步骤。不合格的结果一律丢掉。"""
        try:
            texts = self._screen_texts()
        except Exception:
            texts = []
        try:
            cand = self.replanner(instruction, [r.text for r in results if r.ok], failed.text,
                                  failed.reason, texts)
        except Exception as e:
            self.log.warning("重新规划失败: %s", e)
            return None
        if (isinstance(cand, list) and 1 <= len(cand) <= self.planner.max_steps
                and all(isinstance(c, str) and c.strip() for c in cand)):
            return [c.strip() for c in cand]
        self.log.warning("重新规划的结果不合格，放弃: %r", cand)
        return None

    def run(self, instruction):
        cfg = self.cfg
        t0 = time.perf_counter()
        steps = self.planner.decompose(instruction)
        self.monitor.start_task(instruction, steps)
        results, ok_all, replans = [], True, 0
        recovered_from = None        # 重新规划之后，只看新计划的步骤是否全部成功
        i = 0
        while i < len(steps):
            step = steps[i]
            if time.perf_counter() - t0 > cfg.task_budget_s:
                self.log.error("任务超出时间预算 %.0fs，剩下的步骤跳过", cfg.task_budget_s)
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
                    self.log.info("步骤 %d 失败，重新规划出 %d 步: %s", step.index, len(new), new)
                    steps = steps[:i + 1] + [Step(step.index + 1 + j, t) for j, t in enumerate(new)]
                    recovered_from = len(results)
                    i += 1
                    continue
            if cfg.on_step_fail in ("abort", "replan"):
                self.log.error("步骤 %d 最终失败，后面 %d 步不做了", step.index, len(steps) - i - 1)
                results += [StepResult(s.index, s.text, False, 0, SKIPPED) for s in steps[i + 1:]]
                break
            i += 1
        if recovered_from is not None and len(results) > recovered_from:
            ok_all = all(r.ok for r in results[recovered_from:])
        summary = self.monitor.end_task(ok_all, [r for r in results if r.reason != SKIPPED])
        summary["replans"] = replans
        return TaskResult(instruction, ok_all, results, summary)
