"""The 20 desktop tasks used in the week-7 evaluation (plus 4 extended-action tasks from week 8).

Three difficulty levels:
  L1 single step: one action (launch an app, click one text target)
  L2 2-3 steps:   a short chain (open an app then click a menu, search, or enter a URL)
  L3 hard:        three or more steps, several apps, or a case the system deliberately cannot handle yet
                  (e.g. typing into a blank text area with no text on it)

Success is judged by whether the expected keywords appear in the OCR text of the final screen. `expect`
is a list of groups: a group is satisfied if ANY of its words appears, and the task succeeds if ALL
groups are satisfied. Some tasks (e.g. clicking "Downloads" in the sidebar, where that word is always
visible) cannot be judged reliably by OCR; they are marked auto_reliable=False and the real-machine
runner asks for manual confirmation.

The instructions are in Chinese because the agent was developed and evaluated on a Chinese Windows UI;
the English meaning is given in a comment next to each task. Expected keywords include both the
Chinese and the English UI text, so the tasks also work on an English Windows UI.
"""
from dataclasses import dataclass, field

# Clean-up command per app, run after each task.
# File Explorer cannot simply be taskkill-ed (the taskbar would go with it), so its windows are closed one by one via COM
KILL = {
    "notepad": "taskkill /IM notepad.exe /F",
    "calculator": "taskkill /IM CalculatorApp.exe /F",
    "chrome": "taskkill /IM chrome.exe /F",
    "edge": "taskkill /IM msedge.exe /F",
    "explorer": 'powershell -NoProfile -Command "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Quit() }"',
}


@dataclass
class Task:
    id: str
    level: int
    category: str
    apps: list
    instruction: str
    steps: list                         # the steps the planner is expected to produce
    expect: list                        # [[word, word], [word]]: any within a group, all groups
    start_state: str = "desktop"        # start state; only used by the simulation
    setup: list = field(default_factory=list)     # shell commands to run before the task
    auto_reliable: bool = True
    logic_limit: bool = False           # True: expected to fail by design (probes the limits of the system)
    note: str = ""


_NOTEPAD = ["记事本", "Notepad", "无标题", "Untitled"]          # Notepad / Untitled
_CALC = ["计算器", "Calculator"]                               # Calculator

TASKS = [
    # ---- L1: single step ----
    # T01: open Notepad
    Task("T01", 1, "launch app", ["notepad"], "打开记事本", ["打开记事本"], [_NOTEPAD]),
    # T02: open Calculator
    Task("T02", 1, "launch app", ["calculator"], "打开计算器", ["打开计算器"], [_CALC]),
    # T03: open Chrome
    Task("T03", 1, "launch app", ["chrome"], "打开 Chrome", ["打开 Chrome"],
         [["Chrome", "Google", "新标签页", "New Tab"]]),
    # T04: open Edge
    Task("T04", 1, "launch app", ["edge"], "打开 Edge", ["打开 Edge"],
         [["Edge", "新建标签页", "New tab", "Microsoft"]]),
    # T05: open File Explorer
    Task("T05", 1, "launch app", ["explorer"], "打开文件资源管理器", ["打开文件资源管理器"],
         [["主页", "此电脑", "快速访问", "Home", "This PC", "Quick access"]]),
    # T06: click Downloads
    Task("T06", 1, "UI click", ["explorer"], "点击 下载", ["点击 下载"], [["下载", "Downloads"]],
         start_state="explorer", setup=["start explorer"], auto_reliable=False,
         note="the sidebar always shows \"Downloads\", so OCR cannot tell whether the click landed"),

    # ---- L2: 2-3 steps ----
    # T07: open Notepad then click File
    Task("T07", 2, "menu", ["notepad"], "打开记事本 然后 点击 文件",
         ["打开记事本", "点击 文件"], [["另存为", "Save as", "新窗口", "New window"]]),
    # T08: open File Explorer then click Documents
    Task("T08", 2, "UI click", ["explorer"], "打开文件资源管理器 然后 点击 文档",
         ["打开文件资源管理器", "点击 文档"], [["文档", "Documents"]], auto_reliable=False,
         note="same as T06: the sidebar always shows \"Documents\""),
    # T09: open Chrome then search "McMaster course selection"
    Task("T09", 2, "search", ["chrome"], "打开 Chrome 然后 搜索 McMaster 选课",
         ["打开 Chrome", "搜索 McMaster 选课"], [["McMaster"]]),
    # T10: open Edge then search "Python tutorial"
    Task("T10", 2, "search", ["edge"], "打开 Edge 然后 搜索 Python 教程",
         ["打开 Edge", "搜索 Python 教程"], [["Python"]]),
    # T11: open Chrome then enter URL www.wikipedia.org
    Task("T11", 2, "URL entry", ["chrome"], "打开 Chrome 然后 输入网址 www.wikipedia.org",
         ["打开 Chrome", "输入网址 www.wikipedia.org"], [["Wikipedia", "维基百科"]]),
    # T12: open Notepad then click Edit
    Task("T12", 2, "menu", ["notepad"], "打开记事本 然后 点击 编辑",
         ["打开记事本", "点击 编辑"], [["撤消", "Undo", "剪切", "Cut"]]),
    # T13: open Notepad then click View
    Task("T13", 2, "menu", ["notepad"], "打开记事本 然后 点击 查看",
         ["打开记事本", "点击 查看"], [["缩放", "Zoom", "状态栏", "Status bar", "自动换行", "Word wrap"]]),
    # T14: open Edge then enter URL www.baidu.com
    Task("T14", 2, "URL entry", ["edge"], "打开 Edge 然后 输入网址 www.baidu.com",
         ["打开 Edge", "输入网址 www.baidu.com"], [["百度", "baidu"]]),

    # ---- L3: hard ----
    # T15: open Chrome then search "weather" then click Images
    Task("T15", 3, "multi-step navigation", ["chrome"], "打开 Chrome 然后 搜索 天气 然后 点击 图片",
         ["打开 Chrome", "搜索 天气", "点击 图片"], [["图片", "Images"]], auto_reliable=False,
         note="the results page always has an \"Images\" tab"),
    # T16: open Edge then search "weather" then click Images
    Task("T16", 3, "multi-step navigation", ["edge"], "打开 Edge 然后 搜索 天气 然后 点击 图片",
         ["打开 Edge", "搜索 天气", "点击 图片"], [["图片", "Images"]], auto_reliable=False,
         note="same as T15"),
    # T17: open Notepad then type "hello"
    Task("T17", 3, "text entry", ["notepad"], "打开记事本 然后 输入 你好",
         ["打开记事本", "输入 你好"], [["你好"]], logic_limit=True,
         note="Notepad's blank edit area has no text, so OCR finds no input box; the rules have no candidate and only the LLM can try"),
    # T18: open Calculator then click 7
    Task("T18", 3, "single-character target", ["calculator"], "打开计算器 然后 点击 7",
         ["打开计算器", "点击 7"], [["7"]], auto_reliable=False,
         note="the target is a single digit, which OCR reads unreliably; success needs a 7 on the display, which OCR may also miss"),
    # T19: open Notepad then open Calculator
    Task("T19", 3, "multi-app", ["notepad", "calculator"], "打开记事本 然后 打开计算器",
         ["打开记事本", "打开计算器"], [_CALC], auto_reliable=False,
         note="Calculator covers Notepad, so only Calculator can be confirmed"),
    # T20: open the Edge browser, search "Edge browser tutorial"
    Task("T20", 3, "robustness", ["edge"], "打开 Edge 浏览器 搜索 Edge 浏览器使用教程",
         ["打开 Edge 浏览器", "搜索 Edge 浏览器使用教程"], [["使用教程"]],
         note="the search text contains the app name; the old launch interceptor treated it as \"launch Edge\""),
]

# Week-8 extended actions (hotkey, scroll, wait). Not part of the 20 tasks; reported separately to compare
# behaviour before (direct_actions=False) and after these actions were added
EXT_TASKS = [
    # E01: open Notepad then press Ctrl+S
    Task("E01", 2, "hotkey", ["notepad"], "打开记事本 然后 按 Ctrl+S",
         ["打开记事本", "按 Ctrl+S"], [["另存为", "Save as"]]),
    # E02: open Chrome then search "weather" then scroll down
    Task("E02", 3, "scroll", ["chrome"], "打开 Chrome 然后 搜索 天气 然后 向下滚动",
         ["打开 Chrome", "搜索 天气", "向下滚动"], [["相关搜索", "下一页", "Related"]]),
    # E03: open Edge then wait 2 seconds then search "Python tutorial"
    Task("E03", 2, "wait", ["edge"], "打开 Edge 然后 等待 2 秒 然后 搜索 Python 教程",
         ["打开 Edge", "等待 2 秒", "搜索 Python 教程"], [["Python"]]),
    # E04: open Chrome then enter URL www.wikipedia.org then press Ctrl+T
    Task("E04", 2, "hotkey", ["chrome"], "打开 Chrome 然后 输入网址 www.wikipedia.org 然后 按 Ctrl+T",
         ["打开 Chrome", "输入网址 www.wikipedia.org", "按 Ctrl+T"], [["新标签页", "New Tab"]]),
]

BY_ID = {t.id: t for t in TASKS + EXT_TASKS}
