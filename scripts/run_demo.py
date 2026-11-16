"""录演示视频用：按顺序跑一组有代表性的任务，每个任务前在控制台打一行标题、倒数几秒。

    python -m scripts.run_demo --dry-run          先试一遍，不动鼠标键盘
    python -m scripts.run_demo --agent llava      正式录（录屏软件先开好）
    python -m scripts.run_demo --only 1,3

演示脚本（分镜）见 docs/demo_script.md。每个任务之间会关掉上一个任务打开的应用，回到桌面。
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

# (标题, 指令, 用到的应用)
DEMO = [
    ("系统命令启动应用（Tool Use）", "打开记事本", ["notepad"]),
    ("两步任务：打开浏览器并搜索（中文走剪贴板）", "打开 Chrome 然后 搜索 McMaster 选课", ["chrome"]),
    ("三步任务：搜索后切到图片", "打开 Edge 然后 搜索 天气 然后 点击 图片", ["edge"]),
    ("菜单操作", "打开记事本 然后 点击 文件", ["notepad"]),
    ("快捷键（第 8 周新增）", "打开记事本 然后 按 Ctrl+S", ["notepad"]),
    ("滚动（第 8 周新增）", "打开 Chrome 然后 搜索 天气 然后 向下滚动", ["chrome"]),
    ("容错：输入内容里含应用名", "打开 Edge 浏览器 搜索 Edge 浏览器使用教程", ["edge"]),
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
    ap.add_argument("--only", default="", help="只跑这几个，按序号，比如 1,3")
    ap.add_argument("--countdown", type=int, default=3)
    a = ap.parse_args()

    args = app.parse_args(["--agent", a.agent] + (["--dry-run"] if a.dry_run else []))
    log = app.setup_logging(args.log_dir)
    executor = app.build_executor(args, log)
    picks = {int(x) for x in a.only.split(",") if x.strip()} or set(range(1, len(DEMO) + 1))

    for i, (title, instruction, apps) in enumerate(DEMO, 1):
        if i not in picks:
            continue
        print("\n" + "=" * 60 + f"\n演示 {i}：{title}\n指令：{instruction}\n" + "=" * 60)
        for n in range(a.countdown, 0, -1):
            print(f"  {n} ...", flush=True)
            time.sleep(1)
        r = executor.run(instruction)
        for s in r.steps:
            mark = "FAIL" if not s.ok else ("OK? " if s.unverified else "OK  ")
            print(f"  步骤 {s.index}: {mark} {s.text}  [{s.route or '-'}]" + ("" if s.ok else f" -> {s.reason}"))
        print(f"  结果：{'成功' if r.ok else '失败'}，耗时 {r.summary['elapsed_s']}s")
        time.sleep(2)
        cleanup(apps, a.dry_run)


if __name__ == "__main__":
    main()
