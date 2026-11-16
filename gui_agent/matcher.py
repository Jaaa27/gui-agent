"""指令理解和元素匹配，纯规则，只用标准库。

大致流程：先判断指令是"点击"还是"输入"，点击就拿关键词去给屏幕上的文字打分排序，
输入就先抠出要打的内容、再挑最像输入框的元素。规则找不到才轮到大模型。
"""
import difflib
import re

# 开头的动词，以及"按钮/图标"这类不属于目标名字的词
_LEAD_CN = ["点一下", "点击", "打开", "进入", "双击", "按下", "寻找", "找到", "启动", "运行"]
_LEAD_EN = ["double click", "click", "open", "find", "press", "launch", "run"]
_NOISE_CN = ["标签页", "图标", "按钮", "链接", "文本", "文字"]
_NOISE_EN = ["tab", "button", "icon", "link", "text", "the"]
# 第二档：第一档匹配不上才去掉。有些字（比如"退出"）本身可能就是目标名字的一部分
_TIER2_CN = ["屏幕右上角的", "屏幕左上角的", "屏幕右下角的", "屏幕左下角的", "右上角的", "左上角的",
             "屏幕", "浏览器", "退出"]


def normalize(s):
    """转小写，去掉空白和常见标点。"""
    return re.sub(r"[\s._\-…,，:：!！'\"“”‘’()（）\[\]【】。]", "", str(s).lower())


def _strip_lead(s):
    s = re.sub(r"^(?:请|帮我|麻烦你?)\s*", "", s.strip())
    for _ in range(2):                      # 最多剥两层，比如"请 点击 打开"这种
        low, hit = s.lower(), False
        for v in _LEAD_CN:
            if low.startswith(v):
                s, hit = s[len(v):].lstrip(), True
                break
        if not hit:
            for v in _LEAD_EN:
                # 英文要看词边界，不然 openai 会被剥成 open + ai
                if low.startswith(v) and (len(low) == len(v) or not low[len(v)].isalnum()):
                    s, hit = s[len(v):].lstrip(), True
                    break
        if not hit:
            break
    return s


def _remove(s, cn, en):
    for w in sorted(cn, key=len, reverse=True):
        s = s.replace(w, " ")
    for w in en:
        # 英文词按单词边界删。v1 直接 replace，"Weather" 里的 the 会被删掉
        s = re.sub(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", " ", s, flags=re.I)
    return s


def candidate_keywords(instruction):
    """返回关键词列表，从严到宽，已经 normalize 并去重。"""
    stripped = _strip_lead(instruction)
    t1 = normalize(_remove(stripped, _NOISE_CN, _NOISE_EN))
    t2 = normalize(_remove(stripped, _NOISE_CN + _TIER2_CN, _NOISE_EN))
    kws = []
    for k in (t1, t2):
        if k and k not in kws:
            kws.append(k)
    if not kws:
        k = normalize(stripped) or normalize(instruction)
        if k:
            kws.append(k)
    return kws


def _partial_ratio(a, b):
    """短的那个在长的里面滑一遍，取最高相似度。"""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for i in range(len(b) - len(a) + 1):
        best = max(best, difflib.SequenceMatcher(None, a, b[i:i + len(a)]).ratio())
        if best == 1.0:
            break
    return best


def rank_candidates(instruction, elements, fuzzy=True, min_fuzzy=0.82):
    """给点击类指令排候选，返回 [{"id","score","how","text"}]，分高的在前。"""
    kws = candidate_keywords(instruction)
    items = []
    for k, v in elements.items():
        try:
            eid = int(k)
        except (TypeError, ValueError):
            continue
        txt = str(v.get("text", ""))
        n = normalize(txt)
        if n:
            items.append((eid, txt, n))

    found = {}
    for tier, kw in enumerate(kws):
        pen = 0.02 * tier
        for eid, txt, n in items:
            if n == kw:
                found[eid] = (1.0 - pen, "exact")
        if len(kw) >= 2:
            for eid, txt, n in items:
                if eid not in found and kw in n:
                    # 包含匹配时文字越短越像目标
                    found[eid] = (0.9 - pen - 0.002 * min(len(txt), 100), "contains")
        if found:
            break

    # 精确和包含都没中，再试模糊匹配，扛 OCR 的个别错字（太短的词不做，误伤太多）
    if not found and fuzzy:
        for kw in kws:
            if len(kw) < 4:
                continue
            for eid, txt, n in items:
                r = _partial_ratio(kw, n)
                if r >= min_fuzzy:
                    found[eid] = (0.7 * r, "fuzzy")
            if found:
                break

    text_of = {eid: txt for eid, txt, _ in items}
    out = [{"id": eid, "score": round(s, 4), "how": how, "text": text_of[eid]}
           for eid, (s, how) in found.items()]
    out.sort(key=lambda c: (-c["score"], c["id"]))
    return out


# "框/栏/处/里/中/并/逗号"后面紧跟输入动词，基本就是"在某处输入"
_RULE1 = re.compile(
    r"(?:框|栏|处|里|中|并|，|,|；|;)\s*(?:并且|并)?\s*"
    r"(?P<verb>输入|打字说|打字发送|打字回复|打字|发送|回复|type|send|enter)\s*(?P<rest>.*)$",
    re.I | re.S)

_VERBS2 = ["输入网址", "访问网址", "打开网址", "打字发送", "打字回复", "打字说", "发送消息", "发消息给",
           "发消息", "发送", "回复", "搜索一下", "搜一下", "搜索", "输入", "打字",
           "search for", "search", "send message", "send", "reply", "type",
           "go to", "visit", "enter url"]
# 单独一个"搜"只有后面跟空格才算动词，不然"搜狐""搜狗"会被当成搜索指令
_RULE2 = re.compile(
    "(?P<verb>" + "|".join(re.escape(v) for v in sorted(_VERBS2, key=len, reverse=True)) + r"|搜(?=\s))"
    r"\s*(?P<rest>.*)$", re.I | re.S)

_CLICK_START = re.compile(
    r"^\s*(?:请|帮我)?\s*(?:点击|点一下|打开|进入|双击|按下|寻找|找到|启动|运行|关闭|退出|"
    r"click|double click|open|find|press|launch|run|close|exit)(?![A-Za-z])", re.I)
_CLICK_SUFFIX = re.compile(r"(按钮|按键|图标|链接|标签页|button|icon|link|tab)\s*$", re.I)
_EN_TYPE_IN = re.compile(
    r"^(?:type|enter|write)\s+(?P<p>.+?)\s+(?:in|into|on)\s+(?:the\s+)?(?:\w+\s+){0,3}(?:box|field|bar|input)\s*$",
    re.I)


# ----------------------------------------------------------------------
# 不需要找目标的动作：按键、滚动、等待（第 8 周加的）
# 这三类直接执行，不走 OCR 匹配。必须在 detect_intent 的其他判断之前识别，
# 否则"按下 回车"会被当成点击一个叫"回车"的按钮。
# ----------------------------------------------------------------------
_KEY_ALIASES = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift", "win": "win", "windows": "win",
    "enter": "enter", "回车": "enter", "esc": "esc", "escape": "esc", "tab": "tab",
    "space": "space", "空格": "space", "delete": "delete", "del": "delete", "删除": "delete",
    "backspace": "backspace", "退格": "backspace", "home": "home", "end": "end",
    "pageup": "pageup", "pgup": "pageup", "pagedown": "pagedown", "pgdn": "pagedown",
    "up": "up", "down": "down", "left": "left", "right": "right",
}
_HOTKEY = re.compile(r"^\s*(?:请|帮我)?\s*(?:按下|按|press|快捷键|hotkey)\s*(?P<keys>.+?)\s*键?\s*$", re.I)


def parse_hotkey(instruction):
    """"按 Ctrl+S" -> ["ctrl", "s"]。只要有一个词不是键名就返回 None（"按下 搜索按钮"是点击）。"""
    m = _HOTKEY.match(instruction)
    if not m:
        return None
    keys = []
    for tok in re.split(r"\s*[+＋]\s*|\s+", m.group("keys").strip()):
        t = tok.lower().removesuffix("键")
        k = _KEY_ALIASES.get(t)
        if k is None and re.fullmatch(r"[a-z0-9]|f(?:[1-9]|1[0-2])", t):
            k = t
        if k is None:
            return None
        keys.append(k)
    return keys or None


_SCROLL = re.compile(
    r"^\s*(?:请|帮我)?\s*(?:(?:向|往)\s*)?(?P<d1>上|下)?\s*(?:滚动|滚|滑动|翻页|scroll)\s*(?:到|to)?\s*"
    r"(?P<d2>顶部|底部|最上面|最下面|上|下|top|bottom|up|down)?\s*(?P<n>\d+)?\s*(?:次|下|格|页|行)?\s*$", re.I)
_SCROLL_TO_END = 50          # "滚动到底部"用一个足够大的格数代替


def parse_scroll(instruction):
    """"向下滚动 3 次" -> ("down", 3)；"滚动到顶部" -> ("up", 50)。没写方向默认向下 5 格。"""
    m = _SCROLL.match(instruction)
    if not m:
        return None
    d = (m.group("d1") or m.group("d2") or "下").lower()
    up = d in ("上", "顶部", "最上面", "top", "up")
    n = int(m.group("n")) if m.group("n") else (_SCROLL_TO_END if d in ("顶部", "底部", "最上面", "最下面", "top", "bottom") else 5)
    return ("up" if up else "down"), n


_WAIT = re.compile(r"^\s*(?:请)?\s*(?:等待|等一下|等一会儿?|稍等|wait)\s*"
                   r"(?:(?P<n>\d+(?:\.\d+)?)\s*(?:秒钟?|s|sec|seconds?)?)?\s*$", re.I)


def parse_wait(instruction, default=2.0, cap=30.0):
    m = _WAIT.match(instruction)
    if not m:
        return None
    return min(float(m.group("n")), cap) if m.group("n") else default


def parse_direct(instruction):
    """返回 ("hotkey", [键...]) / ("scroll", (方向, 格数)) / ("wait", 秒)，都不是返回 None。"""
    for kind, fn in (("wait", parse_wait), ("scroll", parse_scroll), ("hotkey", parse_hotkey)):
        v = fn(instruction)
        if v is not None:
            return kind, v
    return None


def text_visible(payload, texts, min_ratio=0.8):
    """输入的内容有没有出现在屏幕上。OCR 可能错一两个字，长一点的内容允许模糊匹配。"""
    p = normalize(payload)
    if not p:
        return True
    for t in texts:
        n = normalize(t)
        if not n:
            continue
        if p in n or (len(n) >= 2 and n in p and len(n) >= 0.6 * len(p)):
            return True
        if len(p) >= 4 and _partial_ratio(p, n) >= min_ratio:
            return True
    return False


def detect_intent(instruction):
    """返回 "click" / "type"，或者不用找目标的 "hotkey" / "scroll" / "wait"。"""
    s = instruction.strip()
    direct = parse_direct(s)
    if direct:
        return direct[0]
    if _EN_TYPE_IN.match(s) or _RULE1.search(s):
        return "type"
    if _CLICK_START.match(s):
        return "click"
    if _RULE2.search(s) and not _CLICK_SUFFIX.search(s):
        return "type"
    return "click"


def _clean_payload(p):
    p = p.strip()
    p = re.sub(r"^[：:]\s*", "", p)
    return p.strip(" \t\r\n「」\"'“”‘’")


def extract_payload(instruction):
    """从输入类指令里抠出要打的内容，抠不到返回空串。"""
    s = instruction.strip()
    m = _EN_TYPE_IN.match(s)
    if m:
        return _clean_payload(m.group("p"))
    m = _RULE1.search(s)
    if m:
        return _clean_payload(m.group("rest"))
    m = _RULE2.search(s)
    if m:
        return _clean_payload(m.group("rest"))
    return ""


_H_SEARCH = ["search", "搜索", "搜寻", "url", "网址", "地址", "address"]
_H_CHAT = ["消息", "聊天", "message", "说点什么", "回复", "reply", "chat", "write"]
_H_ANY = ["输入", "请输入", "type", "enter", "input", "文本框", "在此", "说点什么"]


def input_category(instruction, payload=""):
    """判断是往搜索框、聊天框还是随便哪个输入框里打字。"""
    head = instruction
    if payload and instruction.rstrip().endswith(payload):
        head = instruction.rstrip()[:-len(payload)]      # 内容里的词不能影响分类
    h = head.lower()
    if re.search(r"搜|search|google|网址|url|访问|go to|visit", h):
        return "search"
    if re.search(r"发送|发消息|回复|消息|聊天|send|reply|message|chat", h):
        return "chat"
    return "generic"


def find_input_boxes(instruction, elements, payload=None):
    """按"像不像输入框"给元素打分，返回候选列表，分高的在前。"""
    if payload is None:
        payload = extract_payload(instruction)
    cat = input_category(instruction, payload)
    primary = {"search": _H_SEARCH, "chat": _H_CHAT, "generic": _H_SEARCH + _H_CHAT}[cat]
    out = []
    for k, v in elements.items():
        try:
            eid = int(k)
        except (TypeError, ValueError):
            continue
        raw = str(v.get("text", "")).strip()
        n = normalize(raw)
        if not n:
            continue
        score = 3 * sum(1 for h in primary if normalize(h) in n)
        score += 1 if any(normalize(h) in n for h in _H_ANY) else 0
        score += 1 if ("..." in raw or "…" in raw) else 0     # 占位符一般以省略号结尾
        if score == 0:
            continue
        if len(raw) <= 3:                  # "搜索"这种短词更像按钮，不像输入框
            score -= 1
        if score >= 1:
            out.append({"id": eid, "score": float(score), "how": "input_box", "text": raw})
    out.sort(key=lambda c: (-c["score"], c["id"]))
    return out
