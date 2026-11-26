"""Test cases: the 50 fixed week-4 instructions (defined in training/common.py) plus a set of adversarial cases.

The instructions and screen texts are Chinese; each adversarial case ends with an English note on what it probes.
"""
from training.common import HARD50, MOCK_ELEMENTS as HARD50_ELEMENTS  # noqa: F401


APP_IDS = {10, 11}          # the target is an app icon: launching via a system command also counts

# Screen used by the adversarial cases
ADV_ELEMENTS = {
    3: {"text": "Search Google or type a URL", "center": [500, 300]},
    4: {"text": "天气 25°C", "center": [1700, 1040]},
    5: {"text": "关闭", "center": [1900, 10]},
    6: {"text": "关闭", "center": [1900, 60]},
    7: {"text": "输入聊天信息...", "center": [800, 900]},
    8: {"text": "发送", "center": [1000, 900]},
    9: {"text": "Weather", "center": [300, 200]},
    10: {"text": "搜狐", "center": [400, 200]},
    11: {"text": "搜索", "center": [900, 300]},
    12: {"text": "退出登录", "center": [1800, 100]},
    13: {"text": "report_fina1.docx", "center": [300, 500]},      # OCR read "l" as "1"
}

# (instruction, expected action, acceptable ids, expected text or None, whether a system launch counts, note)
ADVERSARIAL = [
    ("搜索 天气", "type", {3}, "天气", False, "a widget on screen contains '天气' (weather)"),
    ("搜索 chrome 教程", "type", {3}, "chrome 教程", False, "the text to type contains an app name"),
    ("搜索 Edge 怎么用", "type", {3}, "Edge 怎么用", False, "the text to type contains an app name"),
    ("点击 Weather", "click", {9}, None, False, "the English word contains the stop word 'the'"),
    ("打开 report_final.docx", "click", {13}, None, False, "OCR error l->1"),
    ("点击发送按钮", "click", {8}, None, False, "contains a typing verb (发送 send) but is a click"),
    ("发送 你好", "type", {7}, "你好", False, "typing instruction without a location word"),
    ("打开 搜狐", "click", {10}, None, False, "word starting with '搜' (search)"),
    ("搜狐", "click", {10}, None, False, "word starting with '搜', no verb"),
    ("点击退出登录", "click", {12}, None, False, "'退出' (exit) is part of the target name"),
    ("关闭", "click", {5, 6}, None, False, "two elements with the same name"),
    ("搜索 如何发送邮件", "type", {3}, "如何发送邮件", False, "the text to type contains '发送' (send)"),
    ("点击搜索", "click", {11}, None, False, "clicking the '搜索' (Search) button"),
    ("在输入框输入 你好", "type", {3, 7}, "你好", False, "two input boxes; either is acceptable"),
    ("type hello in the search box", "type", {3}, "hello", False, "English typing instruction"),
    ("send hi", "type", {7}, "hi", False, "English send instruction"),
    ("点击 天气", "click", {4}, None, False, "control: a plain click the rules already handle"),
    ("打开 Microsoft Edge", "click", set(), None, True, "control: launch an app"),
]
