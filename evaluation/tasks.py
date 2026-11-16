"""第 7 周评估用的 20 个桌面任务。

难度分三档：
  L1 单步：一个动作就能完成（启动应用、点一个文字目标）
  L2 两三步：两步以内的串联（打开应用后点菜单、搜索、输入网址）
  L3 困难：三步以上、跨应用，或者故意放进了系统现在做不到的情况（比如往没有文字提示的空白文本区输入）

每个任务的成功判定是"最后一屏 OCR 出来的文字里有没有期望的关键词"，expect 是若干组，
每组里命中任意一个词就算这组满足，所有组都满足才算成功。
有的任务（比如点完侧边栏的"下载"，侧边栏本来就一直有这两个字）靠 OCR 判断不准，
标了 auto_reliable=False，真机评估时建议人工确认。

关键词中英文界面都写了。想换成别的应用，改这个文件就行。
"""
from dataclasses import dataclass, field

# 收尾时用的清理命令，一个应用一条。
# 资源管理器不能直接 taskkill，不然任务栏会一起没了，所以用 COM 把窗口一个个关掉
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
    steps: list                         # 期望 planner 拆出来的步骤
    expect: list                        # [[词, 词], [词]]：组内任意、组间全部
    start_state: str = "desktop"        # 起始状态，只有仿真用得上
    setup: list = field(default_factory=list)     # 开始前要先执行的 shell 命令
    auto_reliable: bool = True
    logic_limit: bool = False           # True：按现有设计预计做不了（用来摸系统边界）
    note: str = ""


_NOTEPAD = ["记事本", "Notepad", "无标题", "Untitled"]
_CALC = ["计算器", "Calculator"]

TASKS = [
    # ---- L1：单步 ----
    Task("T01", 1, "启动应用", ["notepad"], "打开记事本", ["打开记事本"], [_NOTEPAD]),
    Task("T02", 1, "启动应用", ["calculator"], "打开计算器", ["打开计算器"], [_CALC]),
    Task("T03", 1, "启动应用", ["chrome"], "打开 Chrome", ["打开 Chrome"],
         [["Chrome", "Google", "新标签页", "New Tab"]]),
    Task("T04", 1, "启动应用", ["edge"], "打开 Edge", ["打开 Edge"],
         [["Edge", "新建标签页", "New tab", "Microsoft"]]),
    Task("T05", 1, "启动应用", ["explorer"], "打开文件资源管理器", ["打开文件资源管理器"],
         [["主页", "此电脑", "快速访问", "Home", "This PC", "Quick access"]]),
    Task("T06", 1, "界面点击", ["explorer"], "点击 下载", ["点击 下载"], [["下载", "Downloads"]],
         start_state="explorer", setup=["start explorer"], auto_reliable=False,
         note="侧边栏本来就有“下载”，OCR 判断不出有没有点中"),

    # ---- L2：两三步 ----
    Task("T07", 2, "菜单操作", ["notepad"], "打开记事本 然后 点击 文件",
         ["打开记事本", "点击 文件"], [["另存为", "Save as", "新窗口", "New window"]]),
    Task("T08", 2, "界面点击", ["explorer"], "打开文件资源管理器 然后 点击 文档",
         ["打开文件资源管理器", "点击 文档"], [["文档", "Documents"]], auto_reliable=False,
         note="同 T06，侧边栏一直有“文档”"),
    Task("T09", 2, "搜索输入", ["chrome"], "打开 Chrome 然后 搜索 McMaster 选课",
         ["打开 Chrome", "搜索 McMaster 选课"], [["McMaster"]]),
    Task("T10", 2, "搜索输入", ["edge"], "打开 Edge 然后 搜索 Python 教程",
         ["打开 Edge", "搜索 Python 教程"], [["Python"]]),
    Task("T11", 2, "网址输入", ["chrome"], "打开 Chrome 然后 输入网址 www.wikipedia.org",
         ["打开 Chrome", "输入网址 www.wikipedia.org"], [["Wikipedia", "维基百科"]]),
    Task("T12", 2, "菜单操作", ["notepad"], "打开记事本 然后 点击 编辑",
         ["打开记事本", "点击 编辑"], [["撤消", "Undo", "剪切", "Cut"]]),
    Task("T13", 2, "菜单操作", ["notepad"], "打开记事本 然后 点击 查看",
         ["打开记事本", "点击 查看"], [["缩放", "Zoom", "状态栏", "Status bar", "自动换行", "Word wrap"]]),
    Task("T14", 2, "网址输入", ["edge"], "打开 Edge 然后 输入网址 www.baidu.com",
         ["打开 Edge", "输入网址 www.baidu.com"], [["百度", "baidu"]]),

    # ---- L3：困难 ----
    Task("T15", 3, "多步导航", ["chrome"], "打开 Chrome 然后 搜索 天气 然后 点击 图片",
         ["打开 Chrome", "搜索 天气", "点击 图片"], [["图片", "Images"]], auto_reliable=False,
         note="搜索结果页本来就有“图片”标签"),
    Task("T16", 3, "多步导航", ["edge"], "打开 Edge 然后 搜索 天气 然后 点击 图片",
         ["打开 Edge", "搜索 天气", "点击 图片"], [["图片", "Images"]], auto_reliable=False,
         note="同 T15"),
    Task("T17", 3, "文本输入", ["notepad"], "打开记事本 然后 输入 你好",
         ["打开记事本", "输入 你好"], [["你好"]], logic_limit=True,
         note="记事本的空白编辑区没有文字，OCR 找不到输入框，规则路径无候选，只能靠大模型"),
    Task("T18", 3, "单字符目标", ["calculator"], "打开计算器 然后 点击 7",
         ["打开计算器", "点击 7"], [["7"]], auto_reliable=False,
         note="目标是单个数字，OCR 对单字符不稳；显示屏上有 7 才算成功，但 OCR 也可能漏"),
    Task("T19", 3, "多应用", ["notepad", "calculator"], "打开记事本 然后 打开计算器",
         ["打开记事本", "打开计算器"], [_CALC], auto_reliable=False,
         note="计算器会盖在记事本上面，只能确认计算器开了"),
    Task("T20", 3, "容错", ["edge"], "打开 Edge 浏览器 搜索 Edge 浏览器使用教程",
         ["打开 Edge 浏览器", "搜索 Edge 浏览器使用教程"], [["使用教程"]],
         note="输入内容里含应用名，旧版的启动拦截器会把它当成启动 Edge"),
]

# 第 8 周扩展的动作（按键、滚动、等待）。不算进 20 个任务里，单独统计，
# 用来对比加这三种动作之前（direct_actions=False）和之后的差别
EXT_TASKS = [
    Task("E01", 2, "快捷键", ["notepad"], "打开记事本 然后 按 Ctrl+S",
         ["打开记事本", "按 Ctrl+S"], [["另存为", "Save as"]]),
    Task("E02", 3, "滚动", ["chrome"], "打开 Chrome 然后 搜索 天气 然后 向下滚动",
         ["打开 Chrome", "搜索 天气", "向下滚动"], [["相关搜索", "下一页", "Related"]]),
    Task("E03", 2, "等待", ["edge"], "打开 Edge 然后 等待 2 秒 然后 搜索 Python 教程",
         ["打开 Edge", "等待 2 秒", "搜索 Python 教程"], [["Python"]]),
    Task("E04", 2, "快捷键", ["chrome"], "打开 Chrome 然后 输入网址 www.wikipedia.org 然后 按 Ctrl+T",
         ["打开 Chrome", "输入网址 www.wikipedia.org", "按 Ctrl+T"], [["新标签页", "New Tab"]]),
]

BY_ID = {t.id: t for t in TASKS + EXT_TASKS}
