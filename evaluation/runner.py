"""真机评估：在自己的 Windows 电脑上把 20 个任务逐个跑一遍，结果追加写进 CSV。

    python -m evaluation.runner --agent llava --repeat 3 --resolution-label 1920x1080-125%
    python -m evaluation.runner --agent llava --no-launch --out results/eval_vision_only.csv   # 只走视觉路径
    python -m evaluation.runner --tasks T01,T05,T09 --repeat 1                                  # 只跑几个

跑之前：
  * 把桌面整理成平时的样子，关掉无关窗口。想测"不靠系统命令启动"的话，桌面上要有对应图标。
  * 评估期间别碰鼠标键盘。鼠标甩到屏幕角落可以紧急停止。
  * 换分辨率/缩放比例评估时，改完系统设置再重新跑，--resolution-label 用来区分，别填错。

流程：每个任务先清理环境 -> 准备起始状态 -> 执行 -> 扫一遍最终画面做自动判定
-> （auto_reliable=False 的任务会停下来问你是否真的完成）-> 记一行 -> 清理。
中断了重新运行同样的命令会自动跳过已经记过的 (条件, 任务, 轮次)。
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
    """当前主屏的缩放比例（百分比），取不到返回空。"""
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
    pyautogui.hotkey("win", "d")           # 先回到桌面
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
    ans = input(f"  [{task.id}] 自动判定={'通过' if auto_ok else '未通过'}，看一眼屏幕，任务真的完成了吗？"
                f"(y=完成 / n=没完成 / 回车=采用自动判定) ").strip().lower()
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
    ap.add_argument("--no-launch", action="store_true", help="不用系统命令启动应用，只走视觉路径")
    ap.add_argument("--max-side", default="auto", help="auto / 0 / 像素数")
    ap.add_argument("--repeat", type=int, default=3, help="每个任务重复几轮")
    ap.add_argument("--tasks", default="", help="只跑这些任务，逗号分隔，比如 T01,T05")
    ap.add_argument("--resolution-label", default="", help="分辨率/缩放标签，比如 1920x1080-125%%")
    ap.add_argument("--manual", choices=["unreliable", "all", "none"], default="unreliable",
                    help="哪些任务评估后要人工确认：自动判定不可靠的(默认)/全部/都不")
    ap.add_argument("--out", default="results/eval_runs.csv")
    ap.add_argument("--log-dir", default="logs")
    return ap.parse_args(argv)


def main():
    a = parse_args()

    # DPI 感知得赶在创建截图/鼠标对象之前
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
    print(f"条件: {condition} | 分辨率: {resolution} | 缩放: {scale or '未知'} | 输出: {a.out}")

    selected = [BY_ID[t] for t in a.tasks.split(",") if t] if a.tasks else TASKS
    finished = done_keys(a.out)
    os.makedirs("results/screens", exist_ok=True)

    for task in selected:
        for rep in range(1, a.repeat + 1):
            if (condition, task.id, str(rep)) in finished:
                continue
            print(f"\n== {task.id} (L{task.level}) 第 {rep}/{a.repeat} 轮: {task.instruction}")
            cleanup(task.apps)
            prepare(task)

            t0 = time.perf_counter()
            try:
                result = executor.run(task.instruction)
            except Exception as e:
                if type(e).__name__ == "FailSafeException":
                    print("触发 FAILSAFE，评估中止。已记录的结果不会丢，重新运行可以接着跑。")
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
                print("  最终画面识别失败:", e)
                auto_ok = False

            need_manual = a.manual == "all" or (a.manual == "unreliable" and not task.auto_reliable)
            manual_ok = ask_manual(task, auto_ok) if need_manual else None

            row = make_row(task, result, condition=condition, resolution=resolution, dpi_scale=scale,
                           repeat=rep, auto_ok=auto_ok, manual_ok=manual_ok, wall_s=wall,
                           run_id=f"{condition}-{resolution}-{task.id}-{rep}")
            append_row(a.out, row)
            print(f"  执行器: {'成功' if result.ok else '失败'} | 最终判定: {'通过' if row['final_ok'] else '未通过'}"
                  f" | {row['elapsed_s']}s | 尝试 {row['attempts']} 次 | {row['failure_reason'] or '-'}")
            cleanup(task.apps)

    print(f"\n完成，结果在 {a.out}。下一步：python -m evaluation.analyze --inputs {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
