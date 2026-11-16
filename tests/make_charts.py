import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]; plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results", "week6")
rt = json.load(open(os.path.join(R, "routing_eval.json"), encoding="utf-8"))
fi = json.load(open(os.path.join(R, "fault_injection.json"), encoding="utf-8"))
C = {"correct": "#2f7d4f", "wrong": "#c0392b", "llm": "#b8b8b8"}
L = {"correct": "规则直接做对", "wrong": "规则做错", "llm": "规则无结果，交给大模型"}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=False)
for ax, key, title in zip(axes, ("hard50", "adversarial"), ("第4周固定的50条指令", "针对v1缺陷设计的18条对抗用例")):
    n = rt[key]["n"]
    for i, ver in enumerate(("v1", "v2")):
        left = 0
        for k in ("correct", "wrong", "llm"):
            v = rt[key][ver][k]
            ax.barh(i, v, left=left, color=C[k], label=L[k] if i == 0 else None)
            if v: ax.text(left + v / 2, i, str(v), ha="center", va="center", color="white" if k != "llm" else "black", fontsize=11)
            left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["v1", "v2"]); ax.set_xlim(0, n); ax.set_title(title, fontsize=11)
    ax.invert_yaxis(); [ax.spines[s].set_visible(False) for s in ("top", "right")]
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,frameon=False,fontsize=9)
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig(os.path.join(R, "chart_routing.png"), dpi=160, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
groups = {}
for r in fi["campaign"]: groups.setdefault((r["p_miss"], r["p_click_fail"], r["p_no_effect"]), []).append(r)
for ax, (k, rows) in zip(axes, groups.items()):
    xs = [r["max_retries"] for r in rows]
    ax.plot(xs, [r["theory"] for r in rows], "--", color="#888", label="理论值 1-(1-q)^n")
    ax.plot(xs, [r["success_rate"] for r in rows], "o-", color="#2f6fb0", label="模拟结果")
    ax.set_title(f"识别漏检{int(k[0]*100)}% / 点击失败{int(k[1]*100)}% / 点击无效{int(k[2]*100)}%", fontsize=9)
    ax.set_xticks(xs); ax.set_xlabel("每步最多尝试次数"); ax.grid(alpha=.25); [ax.spines[s].set_visible(False) for s in ("top", "right")]
axes[0].set_ylabel("两步任务成功率 (%)"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(R, "chart_retry.png"), dpi=160, bbox_inches="tight")
