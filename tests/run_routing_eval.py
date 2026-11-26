"""Rule-routing comparison: v1 (copy of the original main.py) vs v2 (matcher + the decision part of the executor). No LLM."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tests.fakes import install_fake_gui_modules
install_fake_gui_modules()
from tests.cases import HARD50, HARD50_ELEMENTS, APP_IDS, ADVERSARIAL, ADV_ELEMENTS
from tests.legacy_router import v1_route
from gui_agent.matcher import detect_intent, extract_payload, find_input_boxes, rank_candidates
from gui_agent.executor import find_app_command


def v2_route(ins, els):
    app = find_app_command(ins)
    if app:
        return {"kind": "tool_use"}
    intent = detect_intent(ins)
    if intent == "type":
        p = extract_payload(ins)
        c = find_input_boxes(ins, els, p)
        if c:
            return {"kind": "rule", "action": "type", "target_id": c[0]["id"], "text": p}
    else:
        c = rank_candidates(ins, els)
        if c:
            return {"kind": "rule", "action": "click", "target_id": c[0]["id"]}
    return {"kind": "llm"}


def judge(r, act, ids, text, app_ok):
    """Return correct / wrong / llm"""
    if r["kind"] == "llm":
        return "llm"
    if r["kind"] == "tool_use":
        return "correct" if app_ok else "wrong"
    if r["action"] != act or r["target_id"] not in ids:
        return "wrong"
    if text is not None and r["action"] == "type" and r.get("text") != text:
        return "wrong"
    return "correct"


def run(cases, els):
    rows = []
    for ins, act, ids, text, app_ok, note in cases:
        r1, r2 = v1_route(ins, els), v2_route(ins, els)
        # v1's rule path is always a click, never with text
        j1 = judge(r1, act, ids, None if r1["kind"] != "rule" else None, app_ok)
        if r1["kind"] == "rule" and act == "type":
            j1 = "wrong"
        j2 = judge(r2, act, ids, text, app_ok)
        rows.append({"instruction": ins, "note": note, "expected": f"{act}:{sorted(ids) or 'launch'}",
                     "v1": r1, "v1_result": j1, "v2": r2, "v2_result": j2})
    return rows


def tally(rows, k):
    t = {"correct": 0, "wrong": 0, "llm": 0}
    for r in rows:
        t[r[k]] += 1
    return t


if __name__ == "__main__":
    h50 = [(i, a, {t}, x, t in APP_IDS, "hard50") for i, a, t, x in HARD50]
    rows50 = run(h50, HARD50_ELEMENTS)
    rowsadv = run(ADVERSARIAL, ADV_ELEMENTS)
    out = {"hard50": {"n": len(rows50), "v1": tally(rows50, "v1_result"), "v2": tally(rows50, "v2_result"), "rows": rows50},
           "adversarial": {"n": len(rowsadv), "v1": tally(rowsadv, "v1_result"), "v2": tally(rowsadv, "v2_result"), "rows": rowsadv}}
    os.makedirs(os.path.join(ROOT, "results", "week6"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "week6", "routing_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k in ("hard50", "adversarial"):
        print(k, "n=", out[k]["n"], "v1", out[k]["v1"], "v2", out[k]["v2"])
    print("\n--- v2 not correct ---")
    for r in rows50 + rowsadv:
        if r["v2_result"] != "correct":
            print(r["v2_result"], r["instruction"], "| expected", r["expected"], "| v2", r["v2"])
    print("\n--- v1 errors on the adversarial set ---")
    for r in rowsadv:
        if r["v1_result"] != "correct":
            print(r["v1_result"], r["instruction"], r["v1"])
