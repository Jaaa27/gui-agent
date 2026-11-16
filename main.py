"""桌面 GUI 智能体入口。

    python main.py --dry-run            先只打日志，不动鼠标键盘
    python main.py                      正式跑（llava）
    python main.py --agent lora --adapter D:\\week5_lora
    python main.py --agent none         不用大模型，只走规则
    python main.py --calibrate          启动时先校准画面里会自己变化的区域（视频、动图）

运行中输入"校准"可以重新校准。
"""
import argparse
import os
import sys

# DPI 感知必须在创建任何截图/鼠标对象之前设置
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
        log.error("大模型初始化失败: %s，只能走规则路径", e)
        return None


def build_executor(args, log):
    perceiver = ScreenPerceiver(max_side=args.max_side, engine=getattr(args, "ocr", "easyocr"))
    if getattr(args, "calibrate", False):
        n = perceiver.calibrate()
        log.info("校准完成，画面指纹里有 %d 个格子会自己变化，比较时跳过", n)
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
        log.warning("--on-fail replan 需要大模型，现在没有，退回 abort")
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
    ap.add_argument("--no-verify", action="store_true", help="不检查点击后画面有没有变")
    ap.add_argument("--dry-run", action="store_true", help="不真正操作鼠标键盘")
    ap.add_argument("--no-launch", action="store_true", help="不用系统命令启动应用，强制走视觉路径")
    ap.add_argument("--max-side", type=max_side_arg, default="auto",
                    help="OCR 前图像长边上限，0 为不缩放，auto 按系统缩放比例计算（默认）")
    ap.add_argument("--ocr", choices=["easyocr", "paddle"], default="easyocr")
    ap.add_argument("--calibrate", action="store_true", help="启动时校准画面里会自己变化的区域")
    ap.add_argument("--on-fail", choices=["abort", "continue", "replan"], default="abort",
                    help="步骤失败后：停止 / 继续 / 让大模型重新规划")
    ap.add_argument("--no-text-check", action="store_true", help="输入后不再复查屏幕上有没有输入的内容")
    ap.add_argument("--llm-timeout", type=float, default=60.0)
    ap.add_argument("--llm-plan", action="store_true", help="规则拆不开的长指令让大模型拆")
    ap.add_argument("--log-dir", default="logs")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    log = setup_logging(args.log_dir)
    log.info("启动 agent=%s retries=%d verify=%s dry_run=%s",
             args.agent, args.retries, not args.no_verify and not args.dry_run, args.dry_run)
    executor = build_executor(args, log)

    while True:
        try:
            instruction = input("\n任务指令 (exit 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if instruction.lower() == "exit":
            break
        if not instruction:
            continue
        if instruction in ("校准", "calibrate"):
            n = executor.perceiver.calibrate()
            print(f"校准完成，{n} 个格子以后不参与画面比较")
            continue
        try:
            result = executor.run(instruction)
        except Exception as e:
            if type(e).__name__ == "FailSafeException":
                log.error("触发 FAILSAFE（鼠标被甩到屏幕角落），任务已停止")
                continue
            log.exception("任务执行出现未预期的异常: %s", e)
            continue
        s = result.summary
        print(f"\n结果: {'成功' if result.ok else '失败'} | 步骤 {s['steps_ok']}/{s['steps_total']} | "
              f"尝试 {s['attempts']} 次 | 耗时 {s['elapsed_s']}s")
        for r in result.steps:
            mark = "FAIL" if not r.ok else ("OK? " if r.unverified else "OK  ")
            print(f"  步骤 {r.index}: {mark} {r.text}" + ("" if r.ok else f"  -> {r.reason}"))
        if s.get("unverified_steps"):
            print("  OK? 表示动作做了，但屏幕上没找到能确认它做对了的证据，建议看一眼")
    log.info("退出")


if __name__ == "__main__":
    sys.exit(main())
