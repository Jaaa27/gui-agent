"""Logging and task monitoring.

setup_logging configures the normal logs (console + file). TaskMonitor writes every step and attempt
of a task as one JSON line to trace.jsonl and returns a summary when the task ends.
Do not put emoji in log messages: the Windows GBK console raises an encoding error.
"""
import json
import logging
import os
import time
import uuid
from collections import Counter
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG, name="gui_agent"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:                 # don't add handlers twice, or every message is printed several times
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))

    fh = RotatingFileHandler(os.path.join(log_dir, "agent.log"), maxBytes=2_000_000,
                             backupCount=3, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class TaskMonitor:
    """The executor calls this at key points to record what happened; it never makes decisions itself."""

    def __init__(self, logger=None, trace_path="logs/trace.jsonl"):
        self.log = logger or logging.getLogger("gui_agent")
        self.trace_path = trace_path
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        self.events = []
        self.task_id = None
        self._t0 = 0.0
        self._stage = Counter()      # accumulated time per stage: perceive / decide / act / verify

    def _emit(self, **ev):
        ev.update(task=self.task_id, t=round(time.time() - self._t0, 3))
        self.events.append(ev)
        if self.trace_path:
            try:
                with open(self.trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            except OSError as e:           # failing to write the trace must not break the task
                self.log.warning("failed to write trace: %s", e)

    def start_task(self, instruction, steps):
        self.task_id = uuid.uuid4().hex[:8]
        self._t0 = time.time()
        self.events = []
        self._stage = Counter()
        self.log.info("task start [%s] %r, %d step(s)", self.task_id, instruction, len(steps))
        for s in steps:
            self.log.debug("  planned step %d: %s", s.index, s.text)
        self._emit(ev="task_start", instruction=instruction, steps=[s.text for s in steps])

    def step_start(self, step, total):
        self.log.info("step %d/%d start: %s", step.index, total, step.text)
        self._emit(ev="step_start", step=step.index, text=step.text)

    def attempt_end(self, step, attempt, outcome, elapsed):
        for k, v in (outcome.timings or {}).items():
            self._stage[k] += v
        level = logging.INFO if outcome.ok else logging.WARNING
        self.log.log(level, "  step %d attempt %d %s | route=%s target=%s %s%s | %.2fs",
                     step.index, attempt, "ok" if outcome.ok else "FAILED",
                     outcome.route or "-", outcome.target_id if outcome.target_id is not None else "-",
                     ("reason=" + outcome.reason) if not outcome.ok else "",
                     (" (" + outcome.detail + ")") if outcome.detail else "", elapsed)
        self._emit(ev="attempt", step=step.index, attempt=attempt, ok=outcome.ok, reason=outcome.reason,
                   route=outcome.route, target=outcome.target_id, detail=outcome.detail,
                   timings={k: round(v, 3) for k, v in (outcome.timings or {}).items()},
                   elapsed=round(elapsed, 3))

    def step_end(self, step, ok, attempts, reason, elapsed):
        self.log.info("step %d %s (%d attempt(s), %.2fs)%s", step.index, "done" if ok else "FAILED",
                      attempts, elapsed, "" if ok else " final reason=" + reason)
        self._emit(ev="step_end", step=step.index, ok=ok, attempts=attempts, reason=reason,
                   elapsed=round(elapsed, 3))

    def end_task(self, ok, results):
        total = time.time() - self._t0
        reasons = Counter(r.reason for r in results if not r.ok and r.reason)
        attempts = sum(r.attempts for r in results)
        done = sum(1 for r in results if r.ok)
        summary = {
            "task": self.task_id, "ok": ok, "steps_total": len(results), "steps_ok": done,
            "attempts": attempts, "retries": max(0, attempts - sum(1 for r in results if r.attempts)),
            "elapsed_s": round(total, 2), "stage_s": {k: round(v, 2) for k, v in self._stage.items()},
            "failure_reasons": dict(reasons),
            "unverified_steps": sum(1 for r in results if getattr(r, "unverified", False)),
        }
        self.log.info("task %s [%s] steps %d/%d, %d attempt(s), %.2fs",
                      "succeeded" if ok else "FAILED", self.task_id, done, len(results), attempts, total)
        self._emit(ev="task_end", **{k: v for k, v in summary.items() if k != "task"})
        return summary
