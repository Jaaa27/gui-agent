"""Desktop GUI agent - command-line entry point.

    python main.py --dry-run            log only, no mouse or keyboard actions
    python main.py                      normal run (LLaVA fallback)
    python main.py --agent lora --adapter D:\\week5_lora
    python main.py --agent none         rules only, no LLM
    python main.py --calibrate          first calibrate screen regions that change on their own (video, animations)

Type "calibrate" (or "校准") at the prompt to recalibrate.
"""
import argparse
import os
import sys

# DPI awareness must be set before any screenshot or mouse object is created
from gui_agent.perceiver import ScreenPerceiver, enable_dpi_awareness
enable_dpi_awareness()

from gui_agent.executor import ExecConfig, TaskExecutor
from gui_agent.monitor import TaskMonitor, setup_logging
from gui_agent.planner import TaskPlanner, make_llm_decomposer, make_llm_replanner


def build_agent(args, log):
    if args.agent == "none":
        return None
    try:
        if args.agent == "lora":
            from gui_agent.agent_lora import GUIAgentLoRA
            return GUIAgentLoRA(adapter_path=args.adapter)
        from gui_agent.agent import GUIAgent
        return GUIAgent(model_name="llava")
    except Exception as e:
        log.error("LLM initialization failed: %s; using the rule path only", e)
        return None


def build_executor(args, log):
    perceiver = ScreenPerceiver(max_side=args.max_side, engine=getattr(args, "ocr", "easyocr"))
    if getattr(args, "calibrate", False):
        n = perceiver.calibrate()
        log.info("calibration done: %d signature cells change on their own and will be ignored", n)
    if args.dry_run:
        from gui_agent.controller import DryRunController
        controller = DryRunController()
    else:
        from gui_agent.controller import DesktopController
        controller = DesktopController()
    agent = build_agent(args, log)

    planner = TaskPlanner(llm_decompose=make_llm_decomposer(agent) if (args.llm_plan and agent) else None)
    on_fail = getattr(args, "on_fail", "abort")
    replanner = make_llm_replanner(agent) if (on_fail == "replan" and agent is not None) else None
    if on_fail == "replan" and replanner is None:
        log.warning("--on-fail replan needs an LLM and none is available; falling back to abort")
        on_fail = "abort"
    cfg = ExecConfig(max_retries=args.retries, verify=not args.no_verify and not args.dry_run,
                     llm_timeout=args.llm_timeout, use_llm=agent is not None,
                     launch_enabled=(os.name == "nt") and not args.dry_run and not args.no_launch,
                     on_step_fail=on_fail,
                     verify_text=not getattr(args, "no_text_check", False) and not args.dry_run)
    monitor = TaskMonitor(log, trace_path=os.path.join(args.log_dir, "trace.jsonl"))
    return TaskExecutor(perceiver, controller, agent, planner, monitor, cfg, replanner=replanner)


def max_side_arg(v):
    return v if v == "auto" else int(v)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["llava", "lora", "none"], default="llava")
    ap.add_argument("--adapter", default="outputs/lora_gui")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true", help="do not check whether the screen changed after an action")
    ap.add_argument("--dry-run", action="store_true", help="log only; do not move the mouse or press keys")
    ap.add_argument("--no-launch", action="store_true", help="do not launch apps with system commands; force the visual path")
    ap.add_argument("--max-side", type=max_side_arg, default="auto",
                    help="max long side of the screenshot before OCR; 0 = no downscaling, auto = from the display scale (default)")
    ap.add_argument("--ocr", choices=["easyocr", "paddle"], default="easyocr")
    ap.add_argument("--calibrate", action="store_true", help="at startup, calibrate screen regions that change on their own")
    ap.add_argument("--on-fail", choices=["abort", "continue", "replan"], default="abort",
                    help="when a step finally fails: stop / continue / let the LLM re-plan")
    ap.add_argument("--no-text-check", action="store_true", help="do not rescan the screen for the typed text after typing")
    ap.add_argument("--llm-timeout", type=float, default=60.0)
    ap.add_argument("--llm-plan", action="store_true", help="let the LLM split long instructions the rules cannot split")
    ap.add_argument("--log-dir", default="logs")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    log = setup_logging(args.log_dir)
    log.info("start agent=%s retries=%d verify=%s dry_run=%s",
             args.agent, args.retries, not args.no_verify and not args.dry_run, args.dry_run)
    executor = build_executor(args, log)

    while True:
        try:
            instruction = input("\nInstruction (exit to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if instruction.lower() == "exit":
            break
        if not instruction:
            continue
        if instruction in ("校准", "calibrate"):
            n = executor.perceiver.calibrate()
            print(f"calibration done: {n} cells will be ignored when comparing screens")
            continue
        try:
            result = executor.run(instruction)
        except Exception as e:
            if type(e).__name__ == "FailSafeException":
                log.error("FAILSAFE triggered (mouse moved into a screen corner); task stopped")
                continue
            log.exception("unexpected error while running the task: %s", e)
            continue
        s = result.summary
        print(f"\nResult: {'SUCCESS' if result.ok else 'FAILED'} | steps {s['steps_ok']}/{s['steps_total']} | "
              f"{s['attempts']} attempt(s) | {s['elapsed_s']}s")
        for r in result.steps:
            mark = "FAIL" if not r.ok else ("OK? " if r.unverified else "OK  ")
            print(f"  step {r.index}: {mark} {r.text}" + ("" if r.ok else f"  -> {r.reason}"))
        if s.get("unverified_steps"):
            print("  OK? means the action ran but nothing on screen confirmed it worked - worth a quick look")
    log.info("exit")


if __name__ == "__main__":
    sys.exit(main())
