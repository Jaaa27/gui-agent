"""把一条长指令拆成按顺序执行的单步指令。

默认纯规则：按"然后/接着/再"这类连接词拆，另外"打开 X 搜索 Y"这种没有连接词的
也拆成两步。规则拆不开、指令又比较长的时候，可以选择让大模型试着拆一下，
返回的东西不合格就老老实实用规则的结果。
"""
import json
import re
from dataclasses import dataclass

_V = (r"(?:点击|点一下|打开|进入|双击|按下|寻找|找到|启动|运行|搜索一下|搜一下|搜索|输入网址|输入|打字|"
      r"发送|发消息|回复|关闭|退出|访问|按下|按|向下|向上|往下|往上|滚动|等待|稍等|"
      r"click|open|find|press|launch|search|type|send|reply|close|go to|visit|scroll|wait)")
# 连接词后面必须跟着动作动词才拆，否则"发送 我们再见"会被从"再"这里拆开
_SEQ = re.compile(rf"\s*(?:[，,;；。]\s*)?(?:然后|接着|随后|之后|并且|再|and then|then|after that)\s*(?={_V})", re.I)
_PUNCT = re.compile(rf"\s*[;；。]\s*(?={_V})", re.I)
# "打开 X 搜索 Y"。搜索后面可以直接接中文；输入/发送必须隔一个空格，
# 免得把文件名"发送记录.xlsx"拆开
_OPEN_THEN_TYPE = re.compile(
    r"^(?P<open>(?:打开|启动|运行|open|launch)\s*.+?)(?:\s+|并且|并)"
    r"(?P<rest>(?:(?:搜索一下|搜一下|搜索)|(?:输入|发送|发消息|search|type|send)(?=\s)).*)$",
    re.I | re.S)


@dataclass
class Step:
    index: int
    text: str


class TaskPlanner:
    def __init__(self, llm_decompose=None, max_steps=8, llm_min_len=30):
        self.llm_decompose = llm_decompose
        self.max_steps = max_steps
        self.llm_min_len = llm_min_len

    def _rule_split(self, text):
        text = re.sub(r"^先\s*", "", text.strip())
        parts = []
        for p in _SEQ.split(text):
            parts.extend(_PUNCT.split(p))
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = _OPEN_THEN_TYPE.match(p)
            if m:
                out.extend([m.group("open").strip(), m.group("rest").strip()])
            else:
                out.append(p)
        return out

    def decompose(self, instruction):
        texts = self._rule_split(instruction)
        # 不按逗号拆："点击输入聊天信息框，打字说 xxx" 本来就是一步
        if len(texts) == 1 and self.llm_decompose and len(instruction) >= self.llm_min_len:
            try:
                cand = self.llm_decompose(instruction)
                if (isinstance(cand, list) and 1 <= len(cand) <= self.max_steps
                        and all(isinstance(c, str) and c.strip() for c in cand)):
                    texts = [c.strip() for c in cand]
            except Exception:
                pass                       # 大模型拆不出来也没关系，用规则结果
        texts = texts[:self.max_steps] or [instruction.strip()]
        return [Step(i, t) for i, t in enumerate(texts, 1)]


def make_llm_replanner(agent):
    """步骤失败后重新规划：把原指令、做完的步骤、失败的步骤和当前屏幕上的文字给大模型，要剩下的步骤。"""
    def _replan(instruction, done, failed, reason, screen_texts):
        from langchain_core.messages import HumanMessage, SystemMessage
        sys_msg = ("你在帮一个桌面操作程序重新规划。它按步骤执行用户任务，其中一步失败了。"
                   "根据当前屏幕上能看到的文字，给出从现在开始还需要做的步骤，每步只做一个动作"
                   "（点击某个文字、在输入框输入、按键、滚动、等待）。"
                   "只输出 JSON 字符串数组，不要解释；如果做不了，输出 []。")
        texts = [t for t in screen_texts if 0 < len(t) < 30][:60]
        user = (f"用户任务：{instruction}\n已完成：{done}\n失败的步骤：{failed}（原因：{reason}）\n"
                f"当前屏幕上的文字：{texts}")
        raw = str(agent.llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user)]).content)
        m = re.search(r"\[.*\]", raw, re.S)
        return json.loads(m.group(0)) if m else None
    return _replan


def make_llm_decomposer(agent):
    """用 GUIAgent 里现成的 ChatOllama 做纯文本拆分，不传图片。"""
    def _decompose(instruction):
        from langchain_core.messages import HumanMessage, SystemMessage
        sys_msg = ("你把用户的桌面操作任务拆成按顺序执行的单步指令。每一步只做一个动作"
                   "（点击某个文字，或在某个输入框输入某段文字）。"
                   "只输出 JSON 字符串数组，例如 [\"打开 Chrome\", \"搜索 天气\"]，不要解释。")
        raw = str(agent.llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=instruction)]).content)
        m = re.search(r"\[.*\]", raw, re.S)
        return json.loads(m.group(0)) if m else None
    return _decompose
