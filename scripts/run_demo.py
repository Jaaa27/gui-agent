"""For recording the demo video: run a set of representative tasks in order, printing a title and a countdown before each.

    python -m scripts.run_demo --dry-run          rehearsal: no mouse or keyboard actions
    python -m scripts.run_demo --agent llava      recording (start the screen recorder first)
    python -m scripts.run_demo --only 1,3

The storyboard is in docs/demo_script.md. Between tasks, the apps opened by the previous task are closed.
The instructions are in Chinese; the English meaning is in each title.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gui_agent.perceiver import enable_dpi_awareness
enable_dpi_awareness()

import main as app                     # noqa: E402
from evaluation.tasks import KILL     # noqa: E402

# (title, instruction, apps used)
DEMO = [
    ("Launch an app with a system command (tool use): open Notepad", "打开记事本", ["notepad"]),
    ("Two steps: open Chrome then search 'McMaster course selection' (Chinese is pasted via the clipboard)", "打开 Chrome 然后 搜索 McMaster 选课", ["chrome"]),
    ("Three steps: open Edge, search 'weather', click Images", "打开 Edge 然后 搜索 天气 然后 点击 图片", ["edge"]),
    ("Menu: open Notepad then click File", "打开记事本 然后 点击 文件", ["notepad"]),
    ("Hotkey (new in week 8): open Notepad then press Ctrl+S", "打开记事本 然后 按 Ctrl+S", ["notepad"]),
    ("Scroll (new in week 8): open Chrome, search 'weather', scroll down", "打开 Chrome 然后 搜索 天气 然后 向下滚动", ["chrome"]),
    ("Robustness: the search text contains the app name (open Edge, search 'Edge browser tutorial')", "打开 Edge 浏览器 搜索 Edge 浏览器使用教程", ["edge"]),
]


def cleanup(apps, dry):
    if dry or os.name != "nt":
        return
    for a in apps:
        subprocess.run(KILL[a], shell=True, capture_output=True)
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["llava", "lora", "none"], default="none")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="run only these demos by number, e.g. 1,3")
    ap.add_argument("--countdown", type=int, default=3)
    a = ap.parse_args()

    args = app.parse_args(["--agent", a.agent] + (["--dry-run"] if a.dry_run else []))
    log = app.setup_logging(args.log_dir)
    executor = app.build_executor(args, log)
    picks = {int(x) for x in a.only.split(",") if x.strip()} or set(range(1, len(DEMO) + 1))

    for i, (title, instruction, apps) in enumerate(DEMO, 1):
        if i not in picks:
            continue
        print("\n" + "=" * 60 + f"\nDemo {i}: {title}\nInstruction: {instruction}\n" + "=" * 60)
        for n in range(a.countdown, 0, -1):
            print(f"  {n} ...", flush=True)
            time.sleep(1)
        r = executor.run(instruction)
        for s in r.steps:
            mark = "FAIL" if not s.ok else ("OK? " if s.unverified else "OK  ")
            print(f"  step {s.index}: {mark} {s.text}  [{s.route or '-'}]" + ("" if s.ok else f" -> {s.reason}"))
        print(f"  result: {'SUCCESS' if r.ok else 'FAILED'}, {r.summary['elapsed_s']}s")
        time.sleep(2)
        cleanup(apps, a.dry_run)


if __name__ == "__main__":
    main()
