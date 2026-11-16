"""评估记录的字段定义，以及真机评估和仿真共用的几个小函数。"""
import time

COLUMNS = ["run_id", "timestamp", "condition", "resolution", "dpi_scale", "task_id", "level", "category",
           "apps", "instruction", "repeat", "exec_ok", "auto_ok", "manual_ok", "final_ok",
           "elapsed_s", "wall_s", "attempts", "retries", "steps_total", "steps_ok", "failure_reason",
           "routes", "t_perceive", "t_decide", "t_act", "t_verify", "unverified_steps"]


def check_expect(texts, expect):
    """texts 是最后一屏识别出的文字列表。expect 是 [[词, 词], ...]，组内任意、组间全部。"""
    blob = "\n".join(str(t) for t in texts).lower()
    return all(any(w.lower() in blob for w in group) for group in expect)


def make_row(task, result, *, condition, resolution, dpi_scale, repeat, auto_ok, manual_ok, wall_s, run_id):
    """把一次任务执行的结果压成一行记录。"""
    s = result.summary
    stage = s.get("stage_s", {})
    first_fail = next((r.reason for r in result.steps if not r.ok and r.reason), "")
    routes = "|".join(r.route for r in result.steps if r.route)
    final_ok = manual_ok if manual_ok is not None else auto_ok
    return {
        "run_id": run_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "condition": condition, "resolution": resolution, "dpi_scale": dpi_scale,
        "task_id": task.id, "level": task.level, "category": task.category,
        "apps": "+".join(task.apps), "instruction": task.instruction, "repeat": repeat,
        "exec_ok": int(result.ok), "auto_ok": int(auto_ok),
        "manual_ok": "" if manual_ok is None else int(manual_ok), "final_ok": int(final_ok),
        "elapsed_s": round(s["elapsed_s"], 2), "wall_s": round(wall_s, 2),
        "attempts": s["attempts"], "retries": s["retries"],
        "steps_total": len(task.steps), "steps_ok": s["steps_ok"],
        "failure_reason": "" if final_ok else (first_fail or "check_failed"),
        "routes": routes,
        "t_perceive": stage.get("perceive", 0), "t_decide": stage.get("decide", 0),
        "t_act": stage.get("act", 0), "t_verify": stage.get("verify", 0),
        "unverified_steps": s.get("unverified_steps", 0),
    }
