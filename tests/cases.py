"""测试用例：第 4 周的 50 条固定指令（定义在 training/common.py）加上一组对抗用例。"""
from training.common import HARD50, MOCK_ELEMENTS as HARD50_ELEMENTS  # noqa: F401


APP_IDS = {10, 11}          # 目标是应用图标：用系统 API 直接启动也算达成目的

# 对抗用例专用屏幕
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
    13: {"text": "report_fina1.docx", "center": [300, 500]},      # OCR 把 l 识成了 1
}

# (指令, 期望动作, 可接受的 id 集合, 期望文本或 None, 用系统 API 启动是否算对, 说明)
ADVERSARIAL = [
    ("搜索 天气", "type", {3}, "天气", False, "屏幕上有含'天气'的小组件"),
    ("搜索 chrome 教程", "type", {3}, "chrome 教程", False, "输入内容里含应用名"),
    ("搜索 Edge 怎么用", "type", {3}, "Edge 怎么用", False, "输入内容里含应用名"),
    ("点击 Weather", "click", {9}, None, False, "英文词里含停用词 the"),
    ("打开 report_final.docx", "click", {13}, None, False, "OCR 错字 l->1"),
    ("点击发送按钮", "click", {8}, None, False, "含输入动词但其实是点击"),
    ("发送 你好", "type", {7}, "你好", False, "没有位置词的输入指令"),
    ("打开 搜狐", "click", {10}, None, False, "'搜'开头的词"),
    ("搜狐", "click", {10}, None, False, "'搜'开头的词，没有动词"),
    ("点击退出登录", "click", {12}, None, False, "'退出'在目标名字里"),
    ("关闭", "click", {5, 6}, None, False, "两个同名元素"),
    ("搜索 如何发送邮件", "type", {3}, "如何发送邮件", False, "输入内容里含'发送'"),
    ("点击搜索", "click", {11}, None, False, "点的是'搜索'按钮"),
    ("在输入框输入 你好", "type", {3, 7}, "你好", False, "有两个输入框，任意一个都可以接受"),
    ("type hello in the search box", "type", {3}, "hello", False, "英文输入指令"),
    ("send hi", "type", {7}, "hi", False, "英文发送指令"),
    ("点击 天气", "click", {4}, None, False, "对照：规则点击本来就能处理"),
    ("打开 Microsoft Edge", "click", set(), None, True, "对照：启动应用"),
]
