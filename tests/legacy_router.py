"""
按 main.py（v1）里的代码原样复刻"规则路由"这一段，用来和 v2 做对比。
对应 main.py 的 Tool Use 拦截器 + 关键词清洗 + 两轮匹配。不含大模型调用。
"""
V1_TOOL = {"chrome": "start chrome", "edge": "start msedge", "记事本": "notepad", "计算器": "calc"}
V1_STOP = ["点击", "打开", "进入", "搜索", "寻找", "标签页", "图标", "按钮", "链接", "文本", "文字",
           "click", "open", "find", "search", "the", "tab", "button", "icon", "link", "text"]


def v1_route(instruction, elements):
    for app in V1_TOOL:
        if app in instruction.lower():
            return {"kind": "tool_use", "app": app}
    keyword = instruction.lower()
    for w in V1_STOP:
        keyword = keyword.replace(w, "")
    clean_keyword = keyword.strip().replace(" ", "").replace(".", "").replace("_", "")
    if clean_keyword:
        for el_id, d in elements.items():
            if clean_keyword == d["text"].lower().replace(" ", "").replace(".", "").replace("_", ""):
                return {"kind": "rule", "action": "click", "target_id": el_id}
        best, shortest = None, 9999
        for el_id, d in elements.items():
            t = d["text"].lower().replace(" ", "").replace(".", "").replace("_", "")
            if clean_keyword in t and len(d["text"]) < shortest:
                shortest, best = len(d["text"]), el_id
        if best is not None:
            return {"kind": "rule", "action": "click", "target_id": best}
    return {"kind": "llm"}
