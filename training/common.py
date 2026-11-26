"""Shared by training and evaluation: three prompt versions, message construction, output parsing, scoring,
and the fixed 50-instruction evaluation set (hard50).

Prompt v1 is the original one from gui_agent/agent.py (baseline), v2 is the revised one, and ft is the minimal
prompt used for fine-tuning. The prompts, instructions and screen texts are Chinese and functional - they are
exactly what the models were trained and evaluated on.
"""
import json
import re

def _valid_id_str(valid_ids):
    return ", ".join(str(i) for i in valid_ids) if valid_ids else "无合法ID可用"


def system_prompt_v1(valid_ids):
    """Same as the system_prompt in gui_agent/agent.py; the baseline."""
    valid_id_str = _valid_id_str(valid_ids)
    return f"""你是一个极其精确的 GUI 自动化控制中心。
        请仔细观察屏幕截图，并严格对照提供的元素字典。

        【极其严格的空间约束】：
        "target_id" 的值必须且只能从以下合法的 ID 列表中选择：[{valid_id_str}]。
        绝对禁止捏造、猜测列表中不存在的数字！如果列表中没有合适的，请妥协选择一个最接近相关元素的合法 ID。

        你的唯一任务是找到与用户指令相匹配的元素 ID，并决定操作类型。

        【输出格式要求】：
        你的回答必须以 ```json 开头，以 ``` 结尾。

        情况 1：如果是点击任务（如：打开、点击、寻找等），请输出：
        ```json
        {{"action": "click", "target_id": 填入合法ID列表中的数字, "reason": "说明"}}
        ```

        情况 2：如果是输入任务（如：搜索、打字、发送、回复等），请提取要打字的文本：
        ```json
        {{"action": "type", "target_id": 填入合法ID列表中的数字, "text": "提取出的具体要输入的纯文本", "reason": "说明"}}
        ```
        """


def system_prompt_v2(valid_ids):
    """Revised: explicit rules, a few examples, JSON only."""
    valid_id_str = _valid_id_str(valid_ids)
    return f"""你是桌面 GUI 自动化决策器。
输入：屏幕 OCR 得到的元素字典（id -> 文字与坐标）和一条用户指令。
输出：一个 JSON 对象。只输出 JSON，不要任何解释，不要 markdown 代码块。

规则：
1. target_id 必须是整数，且只能取自合法 ID 列表：[{valid_id_str}]。
2. 点击类指令（打开/点击/进入/寻找/关闭/退出/双击）：选文字与指令目标最匹配的元素，action 为 "click"。
3. 输入类指令（搜索/输入/打字/发送/回复/发消息）：target_id 选择"输入框"元素
   （文字通常含 搜索 / Search / 输入 / 网址 / URL / 消息 / message / 聊天 等），action 为 "type"。
   搜索 -> 搜索框；发送/回复/发消息 -> 聊天输入框；输入网址 -> 地址栏。
4. text 只保留用户真正要输入的内容，必须去掉"搜索""输入""发送""打字""在...框"等动作词和位置描述。
5. 指令没有给出具体内容时，text 为空字符串。
6. click 动作不要包含 text 字段。

示例 1
指令：搜索 红烧肉做法
输出：{{"action": "type", "target_id": 7, "text": "红烧肉做法", "reason": "搜索框"}}
示例 2
指令：在输入聊天信息处发送 我马上到
输出：{{"action": "type", "target_id": 12, "text": "我马上到", "reason": "聊天输入框"}}
示例 3
指令：点击取消按钮
输出：{{"action": "click", "target_id": 3, "reason": "匹配文字 取消"}}
"""


def system_prompt_ft(valid_ids):
    """Minimal prompt used for fine-tuning; training and inference must use exactly the same text."""
    return ("你是 GUI 自动化决策器。根据元素字典和用户指令，只输出一个 JSON 动作。"
            f"target_id 必须来自合法 ID：[{_valid_id_str(valid_ids)}]。")


def user_message(elements, instruction):
    """Kept identical to the text part that agent.py sends to the model."""
    return f"元素字典：\n{json.dumps(elements, ensure_ascii=False)}\n\n指令：{instruction}"


def answer_json(target):
    d = {"action": target["action"], "target_id": int(target["target_id"])}
    if target["action"] == "type":
        d["text"] = target.get("text", "")
    d["reason"] = target.get("reason", "")
    return json.dumps(d, ensure_ascii=False)


def build_messages(rec, prompt="ft", with_answer=False, include_valid_ids=True):
    """rec is one sample; returns chat messages."""
    valid = rec["valid_ids"] if include_valid_ids else None
    sys_fn = {"v1": system_prompt_v1, "v2": system_prompt_v2, "ft": system_prompt_ft}[prompt]
    msgs = [
        {"role": "system", "content": sys_fn(valid)},
        {"role": "user", "content": user_message(rec["elements"], rec["instruction"])},
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": answer_json(rec["target"])})
    return msgs


def parse_decision(raw):
    """Lenient parsing: accepts a ```json block, bare JSON, or JSON embedded in text; returns None if nothing parses."""
    if raw is None:
        return None
    s = str(raw).strip()
    cands = []
    m = re.search(r"```json\s*(.*?)```", s, re.S)
    if m:
        cands.append(m.group(1))
    cands.append(s)
    m2 = re.search(r"\{.*\}", s, re.S)
    if m2:
        cands.append(m2.group(0))
    for c in cands:
        try:
            d = json.loads(c.strip())
        except Exception:
            continue
        if isinstance(d, dict):
            if "target_id" in d:
                try:
                    d["target_id"] = int(d["target_id"])
                except Exception:
                    pass
            return d
    return None


def _norm_text(s):
    return re.sub(r"\s+", "", str(s)).strip("，。,.!！?？").casefold()


def score_case(decision, rec):
    """Score one sample; rec["target"] is the reference answer."""
    exp = rec["target"]
    valid = set(int(i) for i in rec["valid_ids"])
    out = {"json_valid": False, "action_ok": False, "id_ok": False,
           "id_in_valid": False, "text_ok": None, "exact": False}
    if not isinstance(decision, dict) or "action" not in decision or "target_id" not in decision:
        return out
    out["json_valid"] = True
    out["action_ok"] = decision.get("action") == exp["action"]
    tid = decision.get("target_id")
    out["id_ok"] = isinstance(tid, int) and tid == int(exp["target_id"])
    out["id_in_valid"] = isinstance(tid, int) and tid in valid
    if exp["action"] == "type" and exp.get("text") is not None:
        out["text_ok"] = _norm_text(decision.get("text", "")) == _norm_text(exp["text"])
    out["exact"] = out["action_ok"] and out["id_ok"] and (out["text_ok"] is not False)
    return out


def aggregate(rows):
    """rows: [{"expected_action":..., "score":{...}, "latency":float}, ...]"""
    def agg(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        def rate(key):
            return round(100.0 * sum(1 for r in rs if r["score"][key]) / n, 2)
        tr = [r for r in rs if r["score"]["text_ok"] is not None]
        return {
            "n": n,
            "json_valid": rate("json_valid"),
            "action_acc": rate("action_ok"),
            "id_acc": rate("id_ok"),
            "id_hallucination": round(100.0 * sum(1 for r in rs if r["score"]["json_valid"] and not r["score"]["id_in_valid"]) / n, 2),
            "text_acc": round(100.0 * sum(1 for r in tr if r["score"]["text_ok"]) / len(tr), 2) if tr else None,
            "exact": rate("exact"),
            "avg_latency_s": round(sum(r["latency"] for r in rs) / n, 3),
        }
    return {
        "overall": agg(rows),
        "click": agg([r for r in rows if r["expected_action"] == "click"]),
        "type": agg([r for r in rows if r["expected_action"] == "type"]),
    }


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Fixed evaluation set hard50: the 50 instructions from the week-4 stress test.
# These exact instructions are excluded from the training data, so the model has never seen them.

MOCK_ELEMENTS = {
    10: {"text": "Google Chrome", "center": [100, 100]},
    11: {"text": "Microsoft Edge", "center": [100, 150]},
    20: {"text": "Search Google or type a URL", "center": [500, 300]},
    30: {"text": "agent.py", "center": [200, 400]},
    31: {"text": "GUI智能体项目计划制定.docx", "center": [200, 450]},
    40: {"text": "输入聊天信息...", "center": [800, 900]},
    50: {"text": "关闭", "center": [1900, 10]},
    51: {"text": "Exit", "center": [1900, 40]},
}

# (instruction, action, target id, expected text; None = text not checked)
HARD50 = [
    ("点击 Google Chrome", "click", 10, None), ("打开 Edge 浏览器", "click", 11, None),
    ("进入 Google Chrome", "click", 10, None), ("寻找 Microsoft Edge", "click", 11, None),
    ("点击 Chrome", "click", 10, None), ("打开 Microsoft Edge 标签页", "click", 11, None),
    ("Google Chrome", "click", 10, None), ("点击图标 Microsoft Edge", "click", 11, None),
    ("进入 Edge", "click", 11, None), ("寻找浏览器 Google Chrome", "click", 10, None),

    ("在浏览器搜索 McMaster 选课", "type", 20, "McMaster 选课"),
    ("点击 Search Google 框并输入 Roush Mustang T56", "type", 20, "Roush Mustang T56"),
    ("搜索 Python 教程", "type", 20, "Python 教程"),
    ("寻找搜索框输入 McMaster", "type", 20, "McMaster"),
    ("在搜索框输入 T56 变速箱", "type", 20, "T56 变速箱"),
    ("搜索天气", "type", 20, "天气"),
    ("输入网址 www.google.com", "type", 20, "www.google.com"),
    ("点击搜索并输入 T56", "type", 20, "T56"),
    ("搜索 GUI Agent", "type", 20, "GUI Agent"),
    ("搜一下周杰伦的歌", "type", 20, "周杰伦的歌"),

    ("打开 agent.py", "click", 30, None), ("点击 GUI智能体项目计划制定.docx", "click", 31, None),
    ("进入 agent.py", "click", 30, None), ("双击 GUI智能体项目计划制定", "click", 31, None),
    ("agent.py", "click", 30, None), ("点击文本 agent.py", "click", 30, None),
    ("寻找 GUI智能体项目计划", "click", 31, None), ("打开链接 agent.py", "click", 30, None),
    ("点击 GUI智能体项目", "click", 31, None), ("agent.py 标签页", "click", 30, None),

    ("发送测试消息内容", "type", 40, None),
    ("在输入聊天信息处发送 爷爷我这周末回国", "type", 40, "爷爷我这周末回国"),
    ("点击输入聊天信息框，打字说 下周赛道见", "type", 40, "下周赛道见"),
    ("发送消息 你好", "type", 40, "你好"),
    ("打字发送 123", "type", 40, "123"),
    ("发消息给 Tyler", "type", 40, None),
    ("在输入框打字", "type", 40, None),
    ("发送语音转文字消息", "type", 40, None),
    ("打字回复 好的", "type", 40, "好的"),
    ("发送 表情包", "type", 40, "表情包"),

    ("关闭", "click", 50, None), ("Exit", "click", 51, None),
    ("点击关闭", "click", 50, None), ("点击 Exit", "click", 51, None),
    ("点击屏幕右上角的关闭", "click", 50, None), ("按下 Exit 按钮", "click", 51, None),
    ("寻找关闭", "click", 50, None), ("点击关闭按钮", "click", 50, None),
    ("退出 Exit", "click", 51, None), ("点击文本关闭", "click", 50, None),
]

HARD50_INSTRUCTIONS = {r[0] for r in HARD50}


def hard50_records():
    recs = []
    for i, (cmd, act, tid, text) in enumerate(HARD50):
        tgt = {"action": act, "target_id": tid}
        if act == "type":
            tgt["text"] = text
        recs.append({
            "id": f"hard50_{i}",
            "kind": "hard50",
            "instruction": cmd,
            "elements": {str(k): v for k, v in MOCK_ELEMENTS.items()},
            "valid_ids": sorted(MOCK_ELEMENTS.keys()),
            "target": tgt,
        })
    return recs
