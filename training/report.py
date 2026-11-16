"""读评估结果，生成微调对比报告（markdown）和几张图。

读 results/summary.json、results/*.json、training_log.json，
写 week5_report.md 和 week5_*_chart.png。

    python -m training.report --results results --train_log outputs/lora_gui/training_log.json --out report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC",
                                   "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAMES = {
    "base_v1_legacy": "基线 v1\n(Week4 旧调用方式)",
    "base_v1": "基线 v1\n(原提示词)",
    "base_v2": "基线 + 提示词 v2\n(提示词工程)",
    "ft_ft": "LoRA 微调",
    "ft_v2": "LoRA + 提示词 v2",
}
COLORS = {"base_v1_legacy": "#9AA5B1", "base_v1": "#4C72B0", "base_v2": "#55A868", "ft_ft": "#DD8452", "ft_v2": "#C44E52"}


def load(results_dir):
    with open(os.path.join(results_dir, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def grouped_bar(summary, metric_path, title, fname, datasets=("test", "hard50")):
    cfgs = list(summary["configs"].keys())
    x = np.arange(len(datasets))
    w = 0.8 / len(cfgs)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, c in enumerate(cfgs):
        vals = []
        for ds in datasets:
            m = summary["configs"][c][ds]
            sec, key = metric_path
            vals.append(m[sec].get(key) or 0)
        rects = ax.bar(x + (i - (len(cfgs) - 1) / 2) * w, vals, w, label=NAMES[c].replace("\n", " "), color=COLORS[c])
        for r, v in zip(rects, vals):
            ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([{"test": "合成测试集 (test)", "hard50": "固定 50 题 (hard50)"}[d] for d in datasets], fontsize=12)
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    if summary.get("mock"):
        fig.text(0.5, 0.5, "MOCK 自测数据", fontsize=48, color="red", alpha=0.25, ha="center", va="center", rotation=25)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def loss_curve(train_log, fname):
    if not train_log or not os.path.exists(train_log):
        return False
    with open(train_log, encoding="utf-8") as f:
        log = json.load(f)["log_history"]
    tr = [(e["step"], e["loss"]) for e in log if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
    if not tr:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*tr), label="训练 loss", color="#4C72B0")
    if ev:
        ax.plot(*zip(*ev), "o-", label="验证 loss", color="#DD8452")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA 微调 Loss 曲线", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return True


def pct(v):
    return "—" if v is None else f"{v:.1f}%"


def table(summary, section, ds):
    head = "| 配置 | n | 合法JSON | 动作正确 | 目标ID正确 | ID幻觉(越界) | 输入文本正确 | **整体正确(exact)** | 平均延迟/条 |\n|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for c, v in summary["configs"].items():
        m = v[ds][section]
        if m.get("n", 0) == 0:
            continue
        rows.append(f"| {NAMES[c].replace(chr(10), ' ')} | {m['n']} | {pct(m['json_valid'])} | {pct(m['action_acc'])} | "
                    f"{pct(m['id_acc'])} | {pct(m['id_hallucination'])} | {pct(m['text_acc'])} | "
                    f"**{pct(m['exact'])}** | {m['avg_latency_s']:.2f}s |")
    return head + "\n".join(rows) + "\n"


def failures(results_dir, cfg, ds, k=8):
    p = os.path.join(results_dir, f"{cfg}__{ds}.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        recs = json.load(f)["records"]
    bad = [r for r in recs if not r["score"]["exact"]]
    return bad[:k], len(bad), len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--train_log", default="outputs/lora_gui/training_log.json")
    ap.add_argument("--out", default="report")
    ap.add_argument("--allow_mock", action="store_true", help="允许对 mock 自测结果生成报告（会加醒目警告）")
    a = ap.parse_args()

    s = load(a.results)
    if s.get("mock") and not a.allow_mock:
        raise SystemExit("summary 是 MOCK 自测数据，不能生成正式报告（只想测流水线请加 --allow_mock）")
    os.makedirs(a.out, exist_ok=True)
    C = s["configs"]

    grouped_bar(s, ("overall", "exact"), "整体正确率对比（动作 + 目标ID + 输入文本全对）",
                os.path.join(a.out, "week5_compare_chart.png"))
    grouped_bar(s, ("type", "exact"), "打字类任务正确率对比（Week4 真实模型 0% 的薄弱环节）",
                os.path.join(a.out, "week5_type_chart.png"))
    has_loss = loss_curve(a.train_log, os.path.join(a.out, "week5_loss_curve.png"))

    def g(cfg, ds, sec, key):
        try:
            return C[cfg][ds][sec][key]
        except KeyError:
            return None

    lines = []
    if s.get("mock"):
        lines.append("> **注意：本报告由 MOCK 自测数据生成，只用来验证流水线，数字没有实际意义。**\n")
    lines += [
        "# Week 5：多模态大模型 LoRA 微调与能力提升 —— 微调效果对比分析报告\n",
        "## 1. 实验设置\n",
        f"- 基座模型：`{s['base']}`",
        f"- LoRA 适配器：`{s['adapter']}`",
        f"- 评测集规模：{s['dataset_sizes']}",
        "- 解码：贪心（do_sample=False），保证可复现",
        "- 打分口径：`exact` = 动作类型正确 且 target_id 正确 且（有标准答案时）输入文本正确；"
        "`ID幻觉` = 输出的 target_id 不在合法 ID 列表内（即 Week4 观察到的模型捏造 ID 问题）\n",
        "## 2. 数据集\n",
        "- 训练/验证/测试集均按 `perceiver.py` 的真实输出格式合成（OCR 元素字典 + 中英文指令 + 干扰元素 + 跳号 ID），"
        "覆盖：打开应用、打开文件、点击按钮、搜索、发送消息、输入网址。",
        "- 防泄漏：hard50 的 50 条指令原文从训练集剔除；输入内容词库与 hard50 不重叠；"
        "train/val/test 之间同一条指令文本不重复。",
        "- 局限：测试集与训练集同属合成分布，hard50 的元素字典也是 Week4 手写的模拟数据，"
        "结果代表\"在 OCR 元素字典上做决策\"的能力，不等同于真实屏幕端到端成功率（Week7 评估）。\n",
        "## 3. 训练\n",
    ]
    if has_loss:
        lines.append("![loss](week5_loss_curve.png)\n")
    else:
        lines.append("（未找到 training_log.json，已跳过 loss 曲线。）\n")
    lines += [
        "## 4. 结果\n",
        "![compare](week5_compare_chart.png)\n",
        "### 4.1 hard50（固定 50 题，整体）\n", table(s, "overall", "hard50"),
        "### 4.2 hard50 — 仅打字类任务（Week4 真实模型全部失败的部分）\n", table(s, "type", "hard50"),
        "![type](week5_type_chart.png)\n",
        "### 4.3 合成测试集 test（整体）\n", table(s, "overall", "test"),
        "### 4.4 合成测试集 test — 仅打字类任务\n", table(s, "type", "test"),
    ]

    # 自动结论
    b1, bl, b2, ft = (g(c, "hard50", "overall", "exact") for c in ("base_v1", "base_v1_legacy", "base_v2", "ft_ft"))
    t1, t2, tf = (g(c, "hard50", "type", "exact") for c in ("base_v1", "base_v2", "ft_ft"))
    lines.append("## 5. 结论（由数据自动生成）\n")
    def d(a_, b_):
        return None if a_ is None or b_ is None else b_ - a_
    if b1 is not None and bl is not None:
        lines.append(f"- **调用方式修复**：Week4 的 `test_edge_cases.py` / `test_agent.py` 调用 `agent.think()` 时没有传 `valid_ids`，"
                     f"提示词里写的是\"无合法ID可用\"，与 `main.py` 实际运行时不一致。补上后 hard50 整体正确率 {bl:.1f}% → {b1:.1f}%。"
                     + ("这部分提升与微调无关，说明 Week4 的\"打字类 0%\"有一部分是测试方法造成的。" if b1 > bl
                        else "两者差异不明显，说明该调用差异不是 Week4 失败的主因。"))
    if b1 is not None and b2 is not None:
        lines.append(f"- **提示词工程（v1 → v2）**：hard50 整体 {b1:.1f}% → {b2:.1f}%（{d(b1, b2):+.1f} 个百分点）"
                     + (f"；打字类 {t1:.1f}% → {t2:.1f}%。" if t1 is not None and t2 is not None else "。"))
    if b2 is not None and ft is not None:
        lines.append(f"- **LoRA 微调**：相对优化提示词的基座，hard50 整体 {b2:.1f}% → {ft:.1f}%（{d(b2, ft):+.1f} 个百分点）"
                     + (f"；打字类 {t2:.1f}% → {tf:.1f}%。" if t2 is not None and tf is not None else "。"))
    h_b, h_f = g("base_v1", "hard50", "overall", "id_hallucination"), g("ft_ft", "hard50", "overall", "id_hallucination")
    if h_b is not None and h_f is not None:
        lines.append(f"- **ID 幻觉率**：{h_b:.1f}% → {h_f:.1f}%。")
    lat_b, lat_f = g("base_v2", "test", "overall", "avg_latency_s"), g("ft_ft", "test", "overall", "avg_latency_s")
    if lat_b is not None and lat_f is not None:
        lines.append(f"- **推理延迟**：优化提示词（含 few-shot）{lat_b:.2f}s/条，微调模型（极简提示词）{lat_f:.2f}s/条。")
    lines.append("")

    # 错误分析
    lines.append("## 6. 错误分析\n")
    for cfg in ("ft_ft", "base_v2"):
        if cfg not in C:
            continue
        res = failures(a.results, cfg, "hard50")
        if not res:
            continue
        bad, nbad, n = res
        lines.append(f"### {NAMES[cfg].replace(chr(10), ' ')}：hard50 中 {nbad}/{n} 条未完全正确（展示前 {len(bad)} 条）\n")
        lines.append("| 指令 | 期望 | 模型输出 |\n|---|---|---|")
        for r in bad:
            e = r["expected"]
            exp = f"{e['action']} #{e['target_id']}" + (f" text={e.get('text')!r}" if e.get("text") else "")
            raw = (r["raw"] or "").replace("\n", " ").replace("|", "\\|")[:90]
            lines.append(f"| {r['instruction']} | {exp} | `{raw}` |")
        lines.append("")

    lines += [
        "## 7. 局限与下一步（Week 6 衔接）\n",
        "- 模型只看 OCR 元素字典（文字 + 坐标），对**纯图标**元素无能为力；这类应用目前由 `main.py` 的 Tool Use 拦截器处理。",
        "- 本次微调为纯文本 LoRA；如需让模型直接利用截图信息，可在同一流水线上替换为 Qwen2-VL 等视觉语言模型。",
        "- 训练数据是合成的，真实屏幕上的 OCR 噪声（错字、断词）需要在 Week 6 通过错误检测与重试机制兜底，并在 Week 7 的真实任务集上验证。",
        "- 若仍有失败，优先检查上表中的错误样本，针对性补充训练模板后重训。",
    ]

    path = os.path.join(a.out, "week5_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已生成：{path}")


if __name__ == "__main__":
    main()
