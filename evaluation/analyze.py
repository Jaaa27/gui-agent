"""Read evaluation CSVs, compute metrics, draw charts and export Excel.

    python -m evaluation.analyze --inputs results/eval_runs.csv --out results/week7
    python -m evaluation.analyze --inputs results/sim_runs.csv --out results/week7_sim

Metric definitions:
  success rate     share of runs with final_ok = 1 (manual confirmation wins over the automatic check),
                   with a Wilson 95% interval
  avg. exec time   executor start-to-end time, successful runs only (failed runs mostly spend their time
                   retrying and would distort the mean)
  error rate       1 - success rate; failures are further split by stage: perception / decision / action /
                   verification / other
  retry rate       share of runs with at least one retry
  false pos./neg.  executor reported success but the task was not done (false positive), or reported
                   failure although the task was done (false negative)

Condition and resolution values in the CSVs are stored as recorded (some in Chinese, e.g.
"允许系统启动" = system launch allowed, "仅视觉" = vision only, "仿真" = simulation);
cond_label() translates them for charts.
"""
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC",
                                   "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STAGE_OF = {"perception_error": "perception", "no_elements": "perception",
            "no_candidate": "decision", "llm_no_output": "decision", "llm_invalid": "decision", "llm_timeout": "decision",
            "action_failed": "action", "no_screen_change": "verification",
            "budget_exceeded": "other", "unexpected": "other", "check_failed": "other"}
STAGES = ["perception", "decision", "action", "verification", "other"]
LEVEL_NAME = {1: "L1 single step", 2: "L2 2-3 steps", 3: "L3 hard"}
APP_NAME = {"notepad": "Notepad", "calculator": "Calculator", "chrome": "Chrome", "edge": "Edge", "explorer": "File Explorer"}
_COND_EN = [("sim-理想感知", "sim (ideal perception)"), ("允许系统启动", "system launch allowed"),
            ("仅视觉", "vision only"), ("无按键滚动等待", "no hotkey/scroll/wait"),
            ("仿真", "simulation")]


def cond_label(c):
    """English label for a condition/resolution value recorded in Chinese."""
    for zh, en in _COND_EN:
        c = c.replace(zh, en)
    return c


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        for k in ("level", "repeat", "exec_ok", "auto_ok", "final_ok", "attempts", "retries", "steps_total", "steps_ok"):
            r[k] = int(float(r[k]))
        for k in ("elapsed_s", "wall_s", "t_perceive", "t_decide", "t_act", "t_verify"):
            r[k] = float(r[k] or 0)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if r["final_ok"]]
    lo, hi = wilson(len(ok), n)
    return {
        "n": n, "success": len(ok),
        "success_rate": round(100 * len(ok) / n, 1) if n else None,
        "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1),
        "avg_time_success": round(sum(r["elapsed_s"] for r in ok) / len(ok), 2) if ok else None,
        "error_rate": round(100 * (n - len(ok)) / n, 1) if n else None,
        "retry_rate": round(100 * sum(1 for r in rows if r["retries"] > 0) / n, 1) if n else None,
        "avg_attempts": round(sum(r["attempts"] for r in rows) / n, 2) if n else None,
        "false_positive": sum(1 for r in rows if r["exec_ok"] and not r["final_ok"]),
        "false_negative": sum(1 for r in rows if not r["exec_ok"] and r["final_ok"]),
    }


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        keys = key(r)
        for k in (keys if isinstance(keys, list) else [keys]):
            g[k].append(r)
    return g


def failure_stages(rows):
    c = Counter(STAGE_OF.get(r["failure_reason"], "other") for r in rows if not r["final_ok"])
    return {s: c.get(s, 0) for s in STAGES}


def analyze(rows):
    out = {"overall": summarize(rows)}
    out["by_condition"] = {k: summarize(v) for k, v in group(rows, lambda r: r["condition"]).items()}
    out["by_level"] = {k: summarize(v) for k, v in sorted(group(rows, lambda r: (r["condition"], r["level"])).items())}
    out["by_app"] = {k: summarize(v) for k, v in sorted(
        group(rows, lambda r: [(r["condition"], a) for a in r["apps"].split("+")]).items())}
    out["by_resolution"] = {k: summarize(v) for k, v in sorted(group(rows, lambda r: (r["condition"], r["resolution"])).items())}
    out["by_task"] = {k: summarize(v) for k, v in sorted(group(rows, lambda r: (r["condition"], r["task_id"])).items())}
    out["failure_stage"] = {k: failure_stages(v) for k, v in group(rows, lambda r: r["condition"]).items()}
    out["failure_reason"] = {k: dict(Counter(r["failure_reason"] for r in v if not r["final_ok"]))
                             for k, v in group(rows, lambda r: r["condition"]).items()}
    return out


# ---------------------------------------------------------------- charts
def _watermark(fig, simulated):
    if simulated:
        fig.text(0.5, 0.5, "SIMULATION - ideal perception - not real-machine results", fontsize=30, color="red",
                 alpha=0.18, ha="center", va="center", rotation=20)


def _bars(ax, labels, rates, cis, color, offset=0.0, width=0.8, label=None):
    xs = [i + offset for i in range(len(labels))]
    err = [[r - lo for r, (lo, hi) in zip(rates, cis)], [hi - r for r, (lo, hi) in zip(rates, cis)]]
    ax.bar(xs, rates, width, color=color, label=label, yerr=err, capsize=3, error_kw={"lw": 1})
    for x, r in zip(xs, rates):
        ax.text(x, r + 2, f"{r:.0f}", ha="center", fontsize=8)


def chart_grouped(rows, a, key_fn, labels_fn, order, title, xlabel, path, simulated, key_name):
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    if not order:
        return False
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(order) + 3), 4.8))
    w = 0.8 / len(conds)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, c in enumerate(conds):
        rates, cis = [], []
        for o in order:
            s = a[key_name].get((c, o))
            rates.append(s["success_rate"] if s else 0)
            cis.append((s["ci_low"], s["ci_high"]) if s else (0, 0))
        _bars(ax, order, rates, cis, palette[i % 4], offset=(i - (len(conds) - 1) / 2) * w, width=w, label=cond_label(c))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels_fn(o) for o in order])
    ax.set_ylim(0, 118)
    ax.set_ylabel("success rate (%), error bars = Wilson 95% CI")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _watermark(fig, simulated)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def chart_per_task(rows, a, path, simulated, with_time):
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    tasks = sorted({r["task_id"] for r in rows})
    fig, axes = plt.subplots(2 if with_time else 1, 1, figsize=(11, 7 if with_time else 4.2), sharex=True)
    axes = axes if with_time else [axes]
    w = 0.8 / len(conds)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, c in enumerate(conds):
        xs = [j + (i - (len(conds) - 1) / 2) * w for j in range(len(tasks))]
        axes[0].bar(xs, [a["by_task"].get((c, t), {}).get("success_rate") or 0 for t in tasks], w,
                    color=palette[i % 4], label=cond_label(c))
        if with_time:
            axes[1].bar(xs, [a["by_task"].get((c, t), {}).get("avg_time_success") or 0 for t in tasks], w,
                        color=palette[i % 4])
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Success rate per task" + (" and mean execution time (successful runs)" if with_time else ""), fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=len(conds), frameon=False)
    if with_time:
        axes[1].set_ylabel("mean execution time (s)")
    axes[-1].set_xticks(range(len(tasks)))
    axes[-1].set_xticklabels(tasks, rotation=45)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _watermark(fig, simulated)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_failure(rows, a, path, simulated):
    conds = list(a["failure_stage"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9AA5B1"]
    bottoms = [0] * len(conds)
    for s, col in zip(STAGES, colors):
        vals = [a["failure_stage"][c][s] for c in conds]
        ax.bar(range(len(conds)), vals, bottom=bottoms, color=col, label=s)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([cond_label(c) for c in conds], fontsize=8)
    ax.set_ylabel("failures")
    ax.set_title("Where failures happen", fontweight="bold")
    ax.legend(title="stage", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _watermark(fig, simulated)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_stage_time(rows, path):
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    names = [("t_perceive", "perceive"), ("t_decide", "decide"), ("t_act", "act"), ("t_verify", "verify")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bottoms = [0] * len(conds)
    for (k, nm), col in zip(names, colors):
        vals = []
        for c in conds:
            rs = [r for r in rows if r["condition"] == c]
            vals.append(sum(r[k] for r in rs) / len(rs) if rs else 0)
        ax.bar(range(len(conds)), vals, bottom=bottoms, color=col, label=nm)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([cond_label(c) for c in conds], fontsize=8)
    ax.set_ylabel("mean time per run (s)")
    ax.set_title("Where the time goes (including retries in failed runs)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Excel
def _sheet(wb, title, header, data, widths=None):
    ws = wb.create_sheet(title)
    ws.append(header)
    for r in data:
        ws.append(r)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4C72B0")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, h in enumerate(header, 1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(h, max(10, min(40, len(str(h)) * 2 + 2)))
    ws.freeze_panes = "A2"
    return ws


def write_excel(rows, a, path, simulated):
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "README"
    notes = ["System evaluation data (week 7)", "",
             "Source: " + ("simulation (ideal perception, NOT real-machine)" if simulated else "real-machine evaluation"),
             "Success rate = share of final_ok; intervals are Wilson 95% confidence intervals.",
             "Mean execution time counts successful runs only. Error rate = 1 - success rate. "
             "Retry rate = share of runs with at least one retry.",
             "Raw records are in the 'raw' sheet and can be filtered or pivoted directly in Excel."]
    for line in notes:
        ws0.append([line])
    ws0["A1"].font = Font(bold=True, size=14)
    ws0.column_dimensions["A"].width = 90

    from .record import COLUMNS
    _sheet(wb, "raw", COLUMNS, [[r.get(c, "") for c in COLUMNS] for r in rows], {"instruction": 40, "run_id": 36})

    hdr = ["condition", "runs", "success", "success %", "CI low %", "CI high %", "mean time (success) s", "error %",
           "retry %", "mean attempts", "false pos.", "false neg."]

    def line(key, s):
        key = list(key) if isinstance(key, tuple) else [key]
        return key + [s["n"], s["success"], s["success_rate"], s["ci_low"], s["ci_high"], s["avg_time_success"],
                      s["error_rate"], s["retry_rate"], s["avg_attempts"], s["false_positive"], s["false_negative"]]

    _sheet(wb, "by condition", hdr, [line(k, s) for k, s in a["by_condition"].items()])
    _sheet(wb, "by level", ["condition", "level"] + hdr[1:],
           [line((k[0], LEVEL_NAME[k[1]]), s) for k, s in a["by_level"].items()])
    _sheet(wb, "by app", ["condition", "app"] + hdr[1:],
           [line((k[0], APP_NAME.get(k[1], k[1])), s) for k, s in a["by_app"].items()])
    _sheet(wb, "by resolution", ["condition", "resolution"] + hdr[1:], [line(k, s) for k, s in a["by_resolution"].items()])
    _sheet(wb, "by task", ["condition", "task"] + hdr[1:], [line(k, s) for k, s in a["by_task"].items()])
    _sheet(wb, "failure stage", ["condition"] + STAGES,
           [[c] + [v[s] for s in STAGES] for c, v in a["failure_stage"].items()])
    reasons = sorted({x for v in a["failure_reason"].values() for x in v})
    _sheet(wb, "failure reason", ["condition"] + reasons, [[c] + [v.get(x, 0) for x in reasons] for c, v in a["failure_reason"].items()])
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="results/week7")
    args = ap.parse_args()

    rows = load(args.inputs)
    simulated = all(r["resolution"] == "仿真" for r in rows)       # "仿真" = simulation (value written by sim.py)
    a = analyze(rows)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"simulated": simulated, "analysis": {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) and k not in ("overall",) else v
                                                        for k, v in a.items()}}, f, ensure_ascii=False, indent=2)
    write_excel(rows, a, os.path.join(args.out, "evaluation_results.xlsx"), simulated)

    levels = sorted({r["level"] for r in rows})
    chart_grouped(rows, a, None, lambda l: LEVEL_NAME[l], levels, "Success rate by task level", "task level",
                  os.path.join(args.out, "fig1_success_by_level.png"), simulated, "by_level")
    chart_per_task(rows, a, os.path.join(args.out, "fig2_per_task.png"), simulated, with_time=not simulated)
    apps = sorted({x for r in rows for x in r["apps"].split("+")})
    chart_grouped(rows, a, None, lambda x: APP_NAME.get(x, x), apps, "Success rate by app (tasks involving the app)", "app",
                  os.path.join(args.out, "fig3_success_by_app.png"), simulated, "by_app")
    resolutions = sorted({r["resolution"] for r in rows})
    if len(resolutions) > 1:
        chart_grouped(rows, a, None, cond_label, resolutions, "Success rate by resolution", "resolution / scaling",
                      os.path.join(args.out, "fig4_success_by_resolution.png"), simulated, "by_resolution")
    chart_failure(rows, a, os.path.join(args.out, "fig5_failure_stage.png"), simulated)
    if not simulated:
        chart_stage_time(rows, os.path.join(args.out, "fig6_stage_time.png"))

    o = a["overall"]
    print(f"{o['n']} runs, success rate {o['success_rate']}% (95% CI {o['ci_low']}-{o['ci_high']})")
    for c, s in a["by_condition"].items():
        print(f"  {c}: {s['success']}/{s['n']} = {s['success_rate']}%")
    print("output folder:", args.out)


if __name__ == "__main__":
    main()
