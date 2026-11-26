"""Real-machine evaluation: run the 20 tasks one by one on your own Windows PC and append the results to a CSV.

    python -m evaluation.runner --agent llava --repeat 3 --resolution-label 1920x1080-125%
    python -m evaluation.runner --agent llava --no-launch --out results/eval_vision_only.csv   # visual path only
    python -m evaluation.runner --tasks T01,T05,T09 --repeat 1                                  # just a few tasks

Before running:
  * Arrange the desktop as usual and close unrelated windows. To test launching without system commands,
    the desktop needs the matching icons.
  * Don't touch the mouse or keyboard during the run. Moving the mouse into a screen corner stops it.
  * To compare resolutions / scaling, change the display settings and run again; --resolution-label tells
    the runs apart, so make sure it matches the actual settings.

Flow per task: clean up -> prepare the start state -> run -> scan the final screen for the automatic check
-> (tasks with auto_reliable=False pause and ask whether the task really succeeded) -> write a row -> clean up.
If interrupted, rerunning the same command skips the (condition, task, repeat) rows already recorded.

The condition string keeps its original Chinese values ("允许系统启动" = system launch allowed,
"仅视觉" = vision only) so that resuming matches rows recorded earlier.
"""
import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from .record import COLUMNS, check_expect, make_row
from .tasks import BY_ID, KILL, TASKS


def dpi_scale():
    """Display scale of the primary monitor in percent, or "" if unavailable."""
    if os.name != "nt":
        return ""
    try:
        import ctypes
        return int(ctypes.windll.shcore.GetScaleFactorForDevice(0))
    except Exception:
        return ""


def sh(cmd, wait=0.0):
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait)


def cleanup(task_apps):
    for app in set(task_apps):
        if app in KILL:
            subprocess.run(KILL[app], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)


def prepare(task):
    import pyautogui
    pyautogui.hotkey("win", "d")           # back to the desktop first
    time.sleep(0.8)
    for cmd in task.setup:
        sh(cmd, wait=3.0)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(r["condition"], r["task_id"], r["repeat"]) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def ask_manual(task, auto_ok):
    ans = input(f"  [{task.id}] automatic check={'PASS' if auto_ok else 'FAIL'}. Look at the screen - was the task really done? "
                f"(y=done / n=not done / Enter=use the automatic check) ").strip().lower()
    if ans == "y":
        return True
    if ans == "n":
        return False
    return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["llava", "lora", "none"], default="llava")
    ap.add_argument("--adapter", default="outputs/lora_gui")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-launch", action="store_true", help="do not launch apps with system commands; visual path only")
    ap.add_argument("--max-side", default="auto", help="auto / 0 / pixels")
    ap.add_argument("--repeat", type=int, default=3, help="repeats per task")
    ap.add_argument("--tasks", default="", help="run only these tasks, comma-separated, e.g. T01,T05")
    ap.add_argument("--resolution-label", default="", help="resolution/scaling label, e.g. 1920x1080-125%%")
    ap.add_argument("--manual", choices=["unreliable", "all", "none"], default="unreliable",
                    help="which tasks need manual confirmation: those with an unreliable automatic check (default) / all / none")
    ap.add_argument("--out", default="results/eval_runs.csv")
    ap.add_argument("--log-dir", default="logs")
    return ap.parse_args(argv)


def main():
    a = parse_args()

    # DPI awareness must be set before any screenshot/mouse object is created
    from gui_agent.perceiver import enable_dpi_awareness
    enable_dpi_awareness()
    from gui_agent.monitor import setup_logging
    from main import build_executor

    log = setup_logging(a.log_dir, console_level=logging.WARNING)
    ns = SimpleNamespace(agent=a.agent, adapter=a.adapter, retries=a.retries, no_verify=False, dry_run=False,
                         no_launch=a.no_launch,
                         max_side=a.max_side if a.max_side == "auto" else int(a.max_side), llm_timeout=60.0, llm_plan=False,
                         log_dir=a.log_dir)
    executor = build_executor(ns, log)

    import pyautogui
    w, h = pyautogui.size()
    resolution = a.resolution_label or f"{w}x{h}"
    condition = f"agent={a.agent}/{'仅视觉' if a.no_launch else '允许系统启动'}"
    scale = dpi_scale()
    print(f"condition: {condition} | resolution: {resolution} | scale: {scale or 'unknown'} | output: {a.out}")

    selected = [BY_ID[t] for t in a.tasks.split(",") if t] if a.tasks else TASKS
    finished = done_keys(a.out)
    os.makedirs("results/screens", exist_ok=True)

    for task in selected:
        for rep in range(1, a.repeat + 1):
            if (condition, task.id, str(rep)) in finished:
                continue
            print(f"\n== {task.id} (L{task.level}) repeat {rep}/{a.repeat}: {task.instruction}")
            cleanup(task.apps)
            prepare(task)

            t0 = time.perf_counter()
            try:
                result = executor.run(task.instruction)
            except Exception as e:
                if type(e).__name__ == "FailSafeException":
                    print("FAILSAFE triggered; evaluation stopped. Recorded rows are kept - rerun the same command to resume.")
                    return 1
                raise
            wall = time.perf_counter() - t0

            time.sleep(1.5)
            executor.perceiver.invalidate()
            shot = f"results/screens/{task.id}_{rep}_{'v' if a.no_launch else 'l'}.png"
            try:
                _, els = executor.perceiver.scan_screen(save_path=shot)
                auto_ok = check_expect([v["text"] for v in els.values()], task.expect)
            except Exception as e:
                print("  failed to scan the final screen:", e)
                auto_ok = False

            need_manual = a.manual == "all" or (a.manual == "unreliable" and not task.auto_reliable)
            manual_ok = ask_manual(task, auto_ok) if need_manual else None

            row = make_row(task, result, condition=condition, resolution=resolution, dpi_scale=scale,
                           repeat=rep, auto_ok=auto_ok, manual_ok=manual_ok, wall_s=wall,
                           run_id=f"{condition}-{resolution}-{task.id}-{rep}")
            append_row(a.out, row)
            print(f"  executor: {'ok' if result.ok else 'FAILED'} | final: {'PASS' if row['final_ok'] else 'FAIL'}"
                  f" | {row['elapsed_s']}s | {row['attempts']} attempt(s) | {row['failure_reason'] or '-'}")
            cleanup(task.apps)

    print(f"\nDone. Results are in {a.out}. Next: python -m evaluation.analyze --inputs {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
