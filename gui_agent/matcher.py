"""Instruction understanding and element matching. Pure rules, standard library only.

Flow: first decide whether the instruction is a "click" or a "type". For a click, its keywords are used to score
and rank the text on screen. For a type, the text to type is extracted and the most input-box-like element is chosen.
Only when the rules find nothing is the LLM consulted.

The Chinese word lists and regexes below are functional: they are what the rules match against.
"""
import difflib
import re

# Leading verbs, and words like "button"/"icon" that are not part of the target name
_LEAD_CN = ["点一下", "点击", "打开", "进入", "双击", "按下", "寻找", "找到", "启动", "运行"]
_LEAD_EN = ["double click", "click", "open", "find", "press", "launch", "run"]
_NOISE_CN = ["标签页", "图标", "按钮", "链接", "文本", "文字"]
_NOISE_EN = ["tab", "button", "icon", "link", "text", "the"]
# Tier 2: only stripped if tier 1 finds no match. Some words (e.g. "退出" = exit) can be part of the target name itself
_TIER2_CN = ["屏幕右上角的", "屏幕左上角的", "屏幕右下角的", "屏幕左下角的", "右上角的", "左上角的",
             "屏幕", "浏览器", "退出"]


def normalize(s):
    """Lower-case and strip whitespace and common punctuation."""
    return re.sub(r"[\s._\-…,，:：!！'\"“”‘’()（）\[\]【】。]", "", str(s).lower())


def _strip_lead(s):
    s = re.sub(r"^(?:请|帮我|麻烦你?)\s*", "", s.strip())
    for _ in range(2):                      # strip at most two layers, e.g. "请 点击 打开" (please click open)
        low, hit = s.lower(), False
        for v in _LEAD_CN:
            if low.startswith(v):
                s, hit = s[len(v):].lstrip(), True
                break
        if not hit:
            for v in _LEAD_EN:
                # English needs word boundaries, otherwise "openai" would be stripped to open + ai
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
        # English words are removed on word boundaries. v1 used plain replace, which removed "the" from "Weather"
        s = re.sub(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", " ", s, flags=re.I)
    return s


def candidate_keywords(instruction):
    """Return keywords from strictest to loosest, normalized and de-duplicated."""
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
    """Slide the shorter string over the longer one and return the best similarity."""
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
    """Rank candidates for a click instruction. Returns [{"id","score","how","text"}], best first."""
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
                    # For substring matches, shorter texts are more likely to be the target
                    found[eid] = (0.9 - pen - 0.002 * min(len(txt), 100), "contains")
        if found:
            break

    # Neither exact nor substring matched: try fuzzy matching to tolerate the odd OCR error (skipped for very short words - too many false hits)
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


# A typing verb right after "box/bar/at/in/and/comma" almost always means "type into <somewhere>"
_RULE1 = re.compile(
    r"(?:框|栏|处|里|中|并|，|,|；|;)\s*(?:并且|并)?\s*"
    r"(?P<verb>输入|打字说|打字发送|打字回复|打字|发送|回复|type|send|enter)\s*(?P<rest>.*)$",
    re.I | re.S)

_VERBS2 = ["输入网址", "访问网址", "打开网址", "打字发送", "打字回复", "打字说", "发送消息", "发消息给",
           "发消息", "发送", "回复", "搜索一下", "搜一下", "搜索", "输入", "打字",
           "search for", "search", "send message", "send", "reply", "type",
           "go to", "visit", "enter url"]
# A lone "搜" (search) only counts as a verb when followed by a space; otherwise site names like 搜狐 / 搜狗 would match
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
# Actions that need no target: hotkey, scroll, wait (added in week 8).
# They run directly without OCR matching. They must be detected before the other checks in detect_intent,
# otherwise "按下 回车" (press Enter) would be treated as clicking a button called "Enter".
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
    """"按 Ctrl+S" -> ["ctrl", "s"]. Returns None if any token is not a key name ("按下 搜索按钮" = press the search button, is a click)."""
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
_SCROLL_TO_END = 50          # "scroll to the bottom/top" uses a large enough number of notches


def parse_scroll(instruction):
    """"向下滚动 3 次" (scroll down 3 times) -> ("down", 3); "滚动到顶部" (scroll to top) -> ("up", 50). Default: down 5 notches."""
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


# "go to github.com/x", "visit www.wikipedia.org", "访问 www.baidu.com" (访问 = visit), or a bare URL: open it in the current
# browser through the address bar (Ctrl+L), instead of looking for an input box on the page. Typing into the
# page's own search box sends the URL to the search engine rather than opening it.
_URL = r"(?:https?://)?[\w-]+(?:\.[\w-]+)+(?::\d+)?(?:/\S*)?"
_NAVIGATE = re.compile(r"^\s*(?:go\s+to|visit|navigate\s+to|open|访问|打开网址|打开网站)\s*(?P<url>" + _URL + r")\s*$", re.I)
# A bare URL needs a scheme, "www." or a path, so file names like "agent.py" or "report.docx" stay clicks
_BARE_URL = re.compile(r"^\s*(?P<url>https?://\S+|www\.[\w-]+(?:\.[\w-]+)+\S*|[\w-]+(?:\.[\w-]+)+/\S*)\s*$", re.I)


def parse_navigate(instruction):
    """"go to github.com/zhanz565/gui-agent" -> "github.com/zhanz565/gui-agent"; None if it is not a URL."""
    m = _NAVIGATE.match(instruction) or _BARE_URL.match(instruction)
    return m.group("url") if m else None


def parse_direct(instruction):
    """Return ("hotkey", [keys...]) / ("scroll", (direction, notches)) / ("wait", seconds) / ("navigate", url), or None."""
    for kind, fn in (("wait", parse_wait), ("scroll", parse_scroll), ("hotkey", parse_hotkey),
                     ("navigate", parse_navigate)):
        v = fn(instruction)
        if v is not None:
            return kind, v
    return None


def text_visible(payload, texts, min_ratio=0.8):
    """Does the typed text appear on screen? OCR may get a character or two wrong, so longer texts may match fuzzily."""
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
    """Return "click" / "type", or one of the target-free intents "hotkey" / "scroll" / "wait"."""
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
    """Extract the text to type from a type instruction; returns "" if none is found."""
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
    """Decide whether the text goes into a search box, a chat box, or any input box."""
    head = instruction
    if payload and instruction.rstrip().endswith(payload):
        head = instruction.rstrip()[:-len(payload)]      # words inside the payload must not affect the classification
    h = head.lower()
    if re.search(r"搜|search|google|网址|url|访问|go to|visit", h):
        return "search"
    if re.search(r"发送|发消息|回复|消息|聊天|send|reply|message|chat", h):
        return "chat"
    return "generic"


def find_input_boxes(instruction, elements, payload=None):
    """Score elements by how much they look like an input box. Returns candidates, best first."""
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
        score += 1 if ("..." in raw or "…" in raw) else 0     # placeholders usually end with an ellipsis
        if score == 0:
            continue
        if len(raw) <= 3:                  # short words like "搜索" (Search) look more like buttons than input boxes
            score -= 1
        if score >= 1:
            out.append({"id": eid, "score": float(score), "how": "input_box", "text": raw})
    out.sort(key=lambda c: (-c["score"], c["id"]))
    return out
