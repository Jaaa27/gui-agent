"""Figures for the final report, all drawn from the data in results/ (the architecture diagram and timeline are schematic).

    python -m scripts.make_final_figures        writes to results/final/

The report is written in Chinese, so the chart labels are Chinese; the architecture diagram is drawn in both
Chinese (fig_architecture.png) and English (fig_architecture_en.png, used by the README).
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "final")

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
BLUE, GREEN, ORANGE, RED, GREY = "#2f6fb0", "#2f7d4f", "#d9822b", "#c0392b", "#b8b8b8"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Architecture diagram text in Chinese and English, in the order fig_architecture draws the boxes and arrow labels
ARCH_TEXT = {
    "zh": dict(
        user="用户指令",
        boxes=[
            ("任务拆解", "planner.py\n规则拆分顺序连接词\n长指令可交大模型拆"),
            ("执行器", "executor.py\n逐步执行、失败分类\n退避重试、重新规划"),
            ("决策（双轨）", "matcher.py 规则优先\n点击打分 / 找输入框\n规则无候选才问大模型"),
            ("大模型兜底", "agent.py：LLaVA（Ollama）\nagent_lora.py：Qwen2.5-1.5B\n+ LoRA 适配器（第5周）"),
            ("屏幕感知", "perceiver.py\n截图（mss）、DPI 感知、自适应降采样\nEasyOCR / PaddleOCR\nSet-of-Mark 编号、画面指纹缓存"),
            ("动作执行", "controller.py\n点击、输入（中文走剪贴板）\n按键、滚动、等待（第8周）\nFAILSAFE 紧急停止"),
            ("系统命令（Tool Use）", "find_app_command\n“打开记事本” → notepad\n启动失败自动改走视觉"),
            ("验证", "画面指纹前后对比（格子计数，可屏蔽动态区域）\n输入后复查屏幕上有没有输入的内容（没有就标记未确认）"),
            ("监控与日志", "monitor.py：控制台、agent.log、trace.jsonl\n各阶段耗时、失败原因、重试次数、未确认步骤"),
        ],
        labels=["② 决策", "无候选", "① 感知", "③ 执行", "⓪ 能直接启动的应用先启动", "④ 前后画面"],
        sub_size=8.2, out="fig_architecture.png"),
    "en": dict(
        user="User instruction",
        boxes=[
            ("Planner", "planner.py\nsplits on sequence\nwords (LLM optional)"),
            ("Executor", "executor.py\nstep-by-step execution\nfailure types, retry, re-plan"),
            ("Decision", "matcher.py: rules first\nscore clicks / find input box\nLLM only if no match"),
            ("LLM fallback", "agent.py: LLaVA (Ollama)\nagent_lora.py: Qwen2.5-1.5B\n+ LoRA adapter (week 5)"),
            ("Screen perception", "perceiver.py: screenshot, DPI-aware\nadaptive downscaling\nEasyOCR / PaddleOCR\nSet-of-Mark IDs, fingerprint cache"),
            ("Action execution", "controller.py\nclick, type (non-ASCII via clipboard)\nhotkey, scroll, wait (week 8)\nFAILSAFE emergency stop"),
            ("System commands (Tool Use)", "find_app_command\n\"open notepad\" → notepad\nfalls back to vision on failure"),
            ("Verification", "before/after screen fingerprint (cell count, dynamic regions masked)\nafter typing, check the text appears on screen (else mark unverified)"),
            ("Monitoring & logging", "monitor.py: console, agent.log, trace.jsonl\nstage timings, failure reasons, retries, unverified steps"),
        ],
        labels=["② decide", "no match", "① perceive", "③ act", "⓪ launch known apps directly", "④ before/after"],
        sub_size=7.8, out="fig_architecture_en.png"),
}


def fig_architecture(lang="zh"):
    T = ARCH_TEXT[lang]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 71)
    ax.axis("off")

    def box(x, y, w, h, i, color):
        title, sub = T["boxes"][i]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=T["sub_size"], color=MUTED,
                linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=INK, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=1.3, linestyle=ls))

    def label(x, y, i, color=INK):
        ax.text(x, y, T["labels"][i], ha="center", va="center", fontsize=8, color=color,
                bbox=dict(fc="white", ec="none", pad=0.6))

    ax.text(11, 69.6, T["user"], ha="center", fontsize=10, color=MUTED)
    box(2, 48, 18, 17, 0, BLUE)
    box(26, 48, 22, 17, 1, BLUE)
    box(54, 48, 22, 17, 2, BLUE)
    box(82, 48, 26, 17, 3, ORANGE)

    box(2, 22, 30, 19, 4, GREEN)
    box(40, 22, 30, 19, 5, GREEN)
    box(78, 22, 30, 19, 6, GREEN)

    box(2, 2, 50, 13, 7, RED)
    box(58, 2, 50, 13, 8, MUTED)

    arrow(11, 68.8, 11, 65.6)
    arrow(20.4, 56.5, 25.6, 56.5)
    arrow(48.4, 56.5, 53.6, 56.5)
    label(51, 59.5, 0)
    arrow(76.4, 56.5, 81.6, 56.5, ORANGE)
    label(79, 59.5, 1, ORANGE)
    arrow(31, 47.6, 20, 41.4)
    label(23, 45.2, 2)
    arrow(62, 47.6, 56, 41.4)
    label(62.5, 44.5, 3)
    arrow(46, 47.6, 86, 41.4, MUTED, "--")
    label(78, 46.2, 4, MUTED)
    arrow(17, 21.6, 17, 15.4)
    label(24, 18.5, 5)
    arrow(55, 21.6, 42, 15.4)
    arrow(52.4, 8.5, 57.6, 8.5)
    _save(fig, T["out"])


def fig_timeline():
    weeks = [
        ("第1周", "技术调研与环境搭建", "调研报告、环境配置文档"),
        ("第2周", "感知与控制模块", "截图+EasyOCR+Set-of-Mark，鼠标键盘控制"),
        ("第3周", "数据集与 Agent 框架", "Mind2Web 流式抽样，LangChain+Ollama 接入 LLaVA"),
        ("第4周", "端到端系统 v1.0", "规则优先+大模型兜底，5类任务50条压测"),
        ("第5周", "LoRA 微调", "合成数据集，Qwen2.5-1.5B LoRA，5种配置对比"),
        ("第6周", "鲁棒性 v2.0", "任务拆解、失败分类重试、DPI/降采样/缓存、日志"),
        ("第7周", "系统评估", "20任务测试集、评估/统计脚本、理想感知仿真、竞品对比"),
        ("第8周", "收尾与待办落实", "按键/滚动/等待、输入复查、动态屏蔽、重规划、结项报告"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(weeks) + 0.6)
    colors = [MUTED, GREEN, GREEN, BLUE, ORANGE, BLUE, RED, BLUE]
    for i, (w, title, out) in enumerate(weeks):
        y = len(weeks) - i - 0.4
        ax.add_patch(FancyBboxPatch((1, y - 0.36), 9, 0.72, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=colors[i], ec="none"))
        ax.text(5.5, y, w, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        ax.text(12.5, y, title, va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(36, y, out, va="center", fontsize=9, color=MUTED)
    _save(fig, "fig_timeline.png")


def fig_week4():
    d = json.load(open(os.path.join(RES, "week4", "week4_results.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = range(len(d["categories"]))
    ax.bar([i - 0.2 for i in x], d["mock_llm"], 0.38, color=GREY, label="Mock 大模型（只验证路由逻辑）")
    ax.bar([i + 0.2 for i in x], d["llava"], 0.38, color=ORANGE, label="本地 LLaVA")
    for i, (a, b) in enumerate(zip(d["mock_llm"], d["llava"])):
        ax.text(i - 0.2, a + 2, f"{a}", ha="center", fontsize=8, color=MUTED)
        ax.text(i + 0.2, b + 2, f"{b}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["categories"], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("通过率 (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _clean(ax)
    _save(fig, "fig_week4.png")


def fig_threshold():
    d = json.load(open(os.path.join(RES, "week6", "threshold_study.json"), encoding="utf-8"))
    names = list(d["new"].keys())
    old = [100 * d["old"][k]["trigger_rate"] for k in names]
    new = [100 * d["new"][k]["trigger_rate"] for k in names]
    noise = [d["new"][k]["kind"] == "noise" for k in names]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], old, 0.38, color=GREY, label="旧判据：全图平均差 > 0.01")
    ax.bar([i + 0.2 for i in x], new, 0.38, color=BLUE, label="新判据：变化格子数 ≥ 3")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i + 0.2, n + 2, f"{n:.0f}", ha="center", fontsize=8, color=INK)
        ax.text(i - 0.2, o + 2, f"{o:.0f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([("（噪声）" if z else "") + k for k, z in zip(names, noise)], fontsize=8.5, rotation=18)
    ax.axvspan(-0.5, sum(noise) - 0.5, color="#f3f3f3", zorder=0)
    ax.text((sum(noise) - 1) / 2, 108, "不应触发", ha="center", fontsize=8.5, color=MUTED)
    ax.text((sum(noise) + len(names) - 1) / 2, 108, "应当检出", ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(0, 118)
    ax.set_ylabel("判为“画面变了”的比例 (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    _clean(ax)
    _save(fig, "fig_threshold.png")


def fig_downscale():
    rows = json.load(open(os.path.join(RES, "week6", "downscale_cost.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    marks = {"1920x1080": ("o", BLUE), "2560x1440": ("s", GREEN), "3840x2160": ("^", ORANGE)}
    for res, (m, c) in marks.items():
        pts = sorted((r["font_after_px"], r["recall"]) for r in rows if r["res"] == res)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c, s=34, label=res, zorder=3)
    ax.axvline(13, color=RED, lw=1, ls="--")
    ax.text(13.2, 8, "13 px：自适应降采样的下限", color=RED, fontsize=8.5)
    ax.set_xlabel("缩小后的界面字号 (px)")
    ax.set_ylabel("文字召回率 (%)")
    ax.set_ylim(-3, 105)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=8.5, title="原始分辨率", title_fontsize=8.5, loc="lower right")
    _clean(ax)
    _save(fig, "fig_downscale.png")


def fig_week8():
    rows = list(csv.DictReader(open(os.path.join(RES, "week8", "sim_ext.csv"), encoding="utf-8-sig")))
    mask = json.load(open(os.path.join(RES, "week8", "mask_study.json"), encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})

    tasks = list(dict.fromkeys(r["task_id"] for r in rows))
    old = {r["task_id"]: int(r["final_ok"]) for r in rows if "无按键" in r["condition"]}
    new = {r["task_id"]: int(r["final_ok"]) for r in rows if "无按键" not in r["condition"]}
    labels = {"E01": "E01 记事本\n按 Ctrl+S", "E02": "E02 搜索后\n向下滚动", "E03": "E03 等待 2 秒\n再搜索",
              "E04": "E04 输入网址\n后按 Ctrl+T"}
    x = range(len(tasks))
    a1.bar([i - 0.2 for i in x], [100 * old[t] for t in tasks], 0.38, color=GREY, label="第 7 周版本（当成点击）")
    a1.bar([i + 0.2 for i in x], [100 * new[t] for t in tasks], 0.38, color=BLUE, label="第 8 周版本（直接执行）")
    a1.set_xticks(list(x))
    a1.set_xticklabels([labels.get(t, t) for t in tasks], fontsize=8)
    for i, t in enumerate(tasks):
        a1.text(i - 0.2, 100 * old[t] + 2, f"{100 * old[t]}", ha="center", fontsize=8, color=MUTED)
        a1.text(i + 0.2, 100 * new[t] + 2, f"{100 * new[t]}", ha="center", fontsize=8)
    a1.set_ylim(0, 112)
    a1.set_ylabel("任务成功率 (%)（仿真）")
    a1.set_title("扩展动作：按键 / 滚动 / 等待", fontsize=10, pad=26)
    a1.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _clean(a1)

    modes = [("none", "不屏蔽"), ("auto", "自动校准"), ("manual", "手动指定区域")]
    ft = [100 * mask[m]["false_trigger"] for m, _ in modes]
    det = [100 * min(mask[m]["detect"].values()) for m, _ in modes]
    x = range(len(modes))
    a2.bar([i - 0.2 for i in x], ft, 0.38, color=RED, label="误触发率（只有视频在播）")
    a2.bar([i + 0.2 for i in x], det, 0.38, color=GREEN, label="真实变化检出率（最低一类）")
    for i, (f, d) in enumerate(zip(ft, det)):
        a2.text(i - 0.2, f + 2, f"{f:.0f}", ha="center", fontsize=8)
        a2.text(i + 0.2, d + 2, f"{d:.0f}", ha="center", fontsize=8)
    a2.set_xticks(list(x))
    a2.set_xticklabels([n for _, n in modes], fontsize=9)
    a2.text(0, 108, "什么都判成“变了”", ha="center", fontsize=8, color=MUTED)
    a2.set_ylim(0, 115)
    a2.set_title("画面里有视频时的验证", fontsize=10, pad=26)
    a2.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _clean(a2)
    plt.tight_layout()
    _save(fig, "fig_week8.png")


def main():
    fig_architecture()
    fig_architecture("en")
    fig_timeline()
    fig_week4()
    fig_threshold()
    fig_downscale()
    fig_week8()
    print("written to", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
