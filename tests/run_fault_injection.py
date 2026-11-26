"""Fault injection: a fake screen state machine + controllable faults, to test the executor's detection / retry / fallback behaviour.

Instructions are Chinese (点击 click, 打开 open, 搜索 search, 关闭 close, 主页 Home, 不存在的东西 "something that doesn't exist").
"""
import json, logging, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tests.fakes import install_fake_gui_modules, World, FakePerceiver, FakeController, FakeAgent, KeyController
PG, PC = install_fake_gui_modules()
from gui_agent.executor import ExecConfig, TaskExecutor
from gui_agent.monitor import TaskMonitor, setup_logging
from gui_agent.planner import TaskPlanner

RES = os.path.join(ROOT, "results", "week6")
LOGDIR = os.path.join(RES, "sample_logs")


def oracle(ins, els):
    for k, v in els.items():
        if "首页" in v["text"]:
            return {"action": "click", "target_id": int(k)}
    return {"action": "click", "target_id": int(next(iter(els)))}


def mk(world, perc=None, ctrl=None, agent=None, launcher=None, replanner=None, **cfg):
    base = dict(launch_enabled=launcher is not None, use_llm=agent is not None, llm_timeout=5.0)
    base.update(cfg)
    perc = perc or FakePerceiver(world)
    ctrl = ctrl or FakeController(world)
    ex = TaskExecutor(perc, ctrl, agent, TaskPlanner(), TaskMonitor(trace_path=None), ExecConfig(**base),
                      launcher=launcher or (lambda c: False), sleep=lambda s: None, replanner=replanner)
    return ex, perc, ctrl


def scenarios():
    S = []

    def add(name, instruction, expect_ok, expect_state, build, check=None):
        S.append(dict(name=name, instruction=instruction, expect_ok=expect_ok, expect_state=expect_state,
                      build=build, check=check))

    add("S01 normal single-step click", "点击 Google Chrome", True, "chrome", lambda: (lambda w: (w, mk(w)))(World()))
    add("S02 two-step task (click, then type)", "点击 Google Chrome 然后 搜索 McMaster 选课", True, "results",
        lambda: (lambda w: (w, mk(w)))(World()))
    add("S03 implicit two steps \"open X search Y\"", "打开 Google Chrome 搜索 天气", True, "results",
        lambda: (lambda w: (w, mk(w)))(World()))
    add("S04 perception raises twice, recovers on the 3rd attempt", "点击 Google Chrome", True, "chrome",
        lambda: (lambda w: (w, mk(w, perc=FakePerceiver(w, exc_first_n=2))))(World()))
    add("S05 perception always fails (expected failure: perception_error)", "点击 Google Chrome", False, "desktop",
        lambda: (lambda w: (w, mk(w, perc=FakePerceiver(w, exc_first_n=99))))(World()),
        lambda r: r.steps[0].reason == "perception_error")
    add("S06 target text missed on the 1st scan, found on the 2nd", "点击 Google Chrome", True, "chrome",
        lambda: (lambda w: (w, mk(w, perc=FakePerceiver(w, miss_texts={"Google Chrome"}, miss_first_n=1))))(World()))
    add("S07 first 2 clicks fail, 3rd succeeds", "点击 Google Chrome", True, "chrome",
        lambda: (lambda w: (w, mk(w, ctrl=FakeController(w, fail_first_n=2))))(World()))
    add("S08 two elements named \"Close\"; the first has no effect -> try the next", "关闭", True, "desktop",
        lambda: (lambda w: (w, mk(w, ctrl=FakeController(w, noeffect_ids={21}))))(World("chrome_dup")),
        lambda r: r.steps[0].attempts == 2)
    add("S09 coordinates out of bounds (expected failure: action_failed)", "点击 Google Chrome", False, "oob",
        lambda: (lambda w: (w, mk(w)))(World("oob")),
        lambda r: r.steps[0].reason == "action_failed")
    add("S10 no rule candidate -> LLM gives an invalid ID, then the right one", "点击 主页", True, "desktop",
        lambda: (lambda w: (w, mk(w, agent=FakeAgent(["bad_id", "good"], oracle))))(World("results")))
    add("S11 LLM times out once, then recovers", "点击 主页", True, "desktop",
        lambda: (lambda w: (w, mk(w, agent=FakeAgent(["timeout", "good"], oracle, timeout_sleep=0.6), llm_timeout=0.2)))(World("results")))
    add("S12 LLM always raises (expected failure: llm_no_output)", "点击 主页", False, "results",
        lambda: (lambda w: (w, mk(w, agent=FakeAgent(["raise"] * 9, oracle))))(World("results")),
        lambda r: r.steps[0].reason == "llm_no_output")
    add("S13 no LLM and no rule candidate (expected failure: no_candidate)", "点击 主页", False, "results",
        lambda: (lambda w: (w, mk(w)))(World("results")),
        lambda r: r.steps[0].reason == "no_candidate")
    add("S14 step 1 fails + abort: remaining steps skipped", "点击 不存在的东西 然后 点击 Google Chrome", False, "desktop",
        lambda: (lambda w: (w, mk(w)))(World()),
        lambda r: [x.reason for x in r.steps] == ["no_candidate", "skipped"])
    add("S15 step 1 fails + continue: remaining steps still run", "点击 不存在的东西 然后 点击 Google Chrome", False, "chrome",
        lambda: (lambda w: (w, mk(w, on_step_fail="continue")))(World()),
        lambda r: [x.ok for x in r.steps] == [False, True])
    add("S16 task time budget exhausted", "点击 Google Chrome", False, "desktop",
        lambda: (lambda w: (w, mk(w, task_budget_s=-1)))(World()),
        lambda r: r.steps[0].reason == "budget_exceeded")

    def s17():
        w = World()
        return w, mk(w, launcher=lambda c: (w.goto("chrome") or True))
    add("S17 launch via system command succeeds (tool-use path)", "打开 Chrome", True, "chrome", s17,
        lambda r: r.steps[0].route == "tool_use" and r.steps[0].attempts == 1)

    def s18():
        w = World()
        return w, mk(w, launcher=lambda c: False)
    add("S18 launch via system command fails -> fall back to the visual path", "点击 Google Chrome", True, "chrome", s18,
        lambda r: r.steps[0].route == "rule")
    add("S19 \"search chrome tutorial\" must not launch Chrome", "搜索 chrome 教程", True, "desktop", lambda: None)

    # ---- week 8 ----
    def s21():
        w = World()
        return w, mk(w, ctrl=KeyController(w, keys={"desktop": {"ctrl+t": "edge"}}))
    add("S21 hotkey runs directly (no target lookup)", "按 Ctrl+T", True, "edge", s21,
        lambda r: r.steps[0].route == "direct" and not r.steps[0].unverified)

    def s22():
        w = World()
        return w, mk(w, ctrl=KeyController(w))
    add("S22 screen unchanged after a hotkey: no retry, marked unverified", "按 Ctrl+V", True, "desktop", s22,
        lambda r: r.steps[0].attempts == 1 and r.steps[0].unverified)

    def s23():
        w = World("results")
        return w, mk(w, ctrl=KeyController(w, scroll_to={"results": "desktop"}))
    add("S23 scroll", "向下滚动 3 次", True, "desktop", s23, lambda r: r.steps[0].route == "direct")
    add("S24 typed text not found on screen: marked unverified, not typed again", "搜索 天气", True, "results",
        lambda: (lambda w: (w, mk(w)))(World("chrome")),
        lambda r: r.steps[0].unverified and r.steps[0].attempts == 1)

    def s25():
        w = World()
        return w, mk(w, replanner=lambda *a: ["点击 Google Chrome"], on_step_fail="replan", max_retries=1)
    add("S25 re-plan after a failed step and finish the task another way", "点击 不存在的东西", True, "chrome", s25,
        lambda r: r.summary.get("replans") == 1)

    def s26():
        w = World()
        return w, mk(w, replanner=lambda *a: None, on_step_fail="replan", max_retries=1)
    add("S26 re-planning returns nothing: handled as abort", "点击 不存在的东西 然后 点击 Google Chrome", False, "desktop", s26,
        lambda r: [x.reason for x in r.steps] == ["no_candidate", "skipped"])
    return S


def run_scenarios():
    os.makedirs(LOGDIR, exist_ok=True)
    for f in ("agent.log", "trace.jsonl"):
        p = os.path.join(LOGDIR, f)
        if os.path.exists(p):
            os.remove(p)
    lg = logging.getLogger("gui_agent")
    for h in list(lg.handlers):
        lg.removeHandler(h)
    setup_logging(LOGDIR, console_level=logging.CRITICAL)
    out = []
    for sc in scenarios():
        if sc["build"]() is None:
            continue                      # S19 is handled separately
        world, (ex, perc, ctrl) = sc["build"]()
        ex.monitor.trace_path = os.path.join(LOGDIR, "trace.jsonl")
        lg.info("===== %s =====", sc["name"])
        t0 = time.time()
        r = ex.run(sc["instruction"])
        extra = sc["check"](r) if sc["check"] else True
        passed = (r.ok == sc["expect_ok"]) and (world.state == sc["expect_state"]) and bool(extra)
        out.append({"name": sc["name"], "instruction": sc["instruction"], "expect_ok": sc["expect_ok"],
                    "ok": r.ok, "final_state": world.state, "expect_state": sc["expect_state"],
                    "attempts": r.summary["attempts"], "retries": r.summary["retries"],
                    "reasons": r.summary["failure_reasons"], "extra_check": bool(extra), "passed": passed,
                    "steps": [{"text": s.text, "ok": s.ok, "attempts": s.attempts, "reason": s.reason, "route": s.route}
                              for s in r.steps]})
    # S19
    w = World()
    ex, perc, ctrl = mk(w, launcher=lambda c: (w.goto("chrome") or True))
    r = ex.run("搜索 chrome 教程")
    launched = w.state == "chrome"
    out.append({"name": "S19 \"search chrome tutorial\" must not launch Chrome", "instruction": "搜索 chrome 教程",
                "expect_ok": "no app launched", "ok": r.ok, "final_state": w.state, "expect_state": "desktop",
                "attempts": r.summary["attempts"], "retries": r.summary["retries"], "reasons": r.summary["failure_reasons"],
                "extra_check": not launched, "passed": not launched, "steps": []})
    # S20: FailSafe must propagate all the way up
    w = World()
    ctrl = FakeController(w)
    def boom(*a, **k): raise PG.FailSafeException()
    ctrl.click = boom
    ex, _, _ = mk(w, ctrl=ctrl)
    raised = False
    try:
        ex.run("点击 Google Chrome")
    except PG.FailSafeException:
        raised = True
    out.append({"name": "S20 FAILSAFE emergency stop must abort the task, not be retried", "instruction": "点击 Google Chrome",
                "expect_ok": "exception raised", "ok": not raised, "final_state": w.state, "expect_state": "desktop",
                "attempts": 0, "retries": 0, "reasons": {}, "extra_check": raised, "passed": raised, "steps": []})
    return out


# ----------------------------------------------------------------------
def campaign(n_trials=3000):
    logging.getLogger("gui_agent").disabled = True
    rows = []
    for (pm, pf, pn) in [(0.10, 0.10, 0.10), (0.20, 0.15, 0.15), (0.30, 0.20, 0.20)]:
        q1 = (1 - pm) * (1 - pf) * (1 - pn)
        q2 = (1 - pm) * (1 - pf)
        for n in (1, 2, 3, 5):
            rng = random.Random(1000 * n + int(pm * 100))
            ok = att = 0
            for _ in range(n_trials):
                w = World()
                perc = FakePerceiver(w, rng=rng, p_miss=pm, miss_texts={"Google Chrome", "Search Google or type a URL"})
                ctrl = FakeController(w, rng=rng, p_fail=pf, p_noeffect=pn)
                ex = TaskExecutor(perc, ctrl, None, TaskPlanner(), TaskMonitor(trace_path=None),
                                  ExecConfig(max_retries=n, launch_enabled=False, use_llm=False,
                                             verify_text=False),   # the post-typing check scans once more and draws extra random numbers; off to keep the week-6 conditions
                                  sleep=lambda s: None)
                r = ex.run("点击 Google Chrome 然后 搜索 McMaster 选课")
                ok += r.ok
                att += r.summary["attempts"]
            # Theory: at most n attempts per step, each succeeding independently with probability q
            #   step 1 (click): q1 = (1-pm)(1-pf)(1-pn); step 2 (type, cannot "click with no effect"): q2 = (1-pm)(1-pf)
            #   P(step ok) = 1-(1-q)^n; P(task ok) = P1*P2 (step 2 only runs after step 1 succeeds)
            #   mean attempts: for one step with at most n tries = (1-(1-q)^n)/q; whole task = E1 + P1*E2
            P1, P2 = 1 - (1 - q1) ** n, 1 - (1 - q2) ** n
            theory = P1 * P2
            theory_att = P1 / q1 + P1 * (P2 / q2)
            se = (theory * (1 - theory) / n_trials) ** 0.5
            rows.append({"p_miss": pm, "p_click_fail": pf, "p_no_effect": pn, "max_retries": n, "trials": n_trials,
                         "success_rate": round(100 * ok / n_trials, 1), "theory": round(100 * theory, 1),
                         "avg_attempts": round(att / n_trials, 2), "theory_attempts": round(theory_att, 2),
                         "z": round((ok / n_trials - theory) / se, 2)})
    logging.getLogger("gui_agent").disabled = False
    return rows


if __name__ == "__main__":
    sc = run_scenarios()
    for s in sc:
        print("PASS" if s["passed"] else "FAIL", s["name"], "| ok=", s["ok"], "state=", s["final_state"],
              "attempts=", s["attempts"], s["reasons"])
    camp = campaign()
    for r in camp:
        print(r)
    json.dump({"scenarios": sc, "campaign": camp}, open(os.path.join(RES, "fault_injection.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nscenarios passed {sum(s['passed'] for s in sc)}/{len(sc)}")
    if not all(s["passed"] for s in sc):
        raise SystemExit(1)                 # make CI fail when a scenario fails
