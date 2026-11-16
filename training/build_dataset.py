"""生成 LoRA 微调用的训练/验证/测试集。

没直接用 Mind2Web：data/dataset_sample_100.json 里只有指令和动作数，没有"屏幕上有哪些元素、
目标是哪一个"的标注，监督不了 agent 真正要做的事。所以这里照着 perceiver 的输出格式合成样本
（带干扰元素、跳号的 id、中英文指令）。

防泄漏：hard50 的 50 条指令不会出现在任何划分里，输入内容的词库也和 hard50 不重叠，
同一条指令文本按哈希只会落进一个划分。

    python -m training.build_dataset --out data --train 2400 --val 200 --test 300
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter

from training.common import HARD50_INSTRUCTIONS, hard50_records, save_jsonl

APPS = ["Google Chrome", "Microsoft Edge", "Firefox", "Safari", "记事本", "Notepad", "计算器", "Calculator",
        "微信", "WeChat", "Visual Studio Code", "Word", "Excel", "PowerPoint", "Spotify", "Discord", "Steam",
        "Telegram", "QQ", "Slack", "Zoom", "文件资源管理器", "控制面板", "回收站", "此电脑", "Photoshop",
        "OBS Studio", "Brave Browser", "Opera", "网易云音乐"]

FILES = ["report.docx", "notes.txt", "budget_2024.xlsx", "简历.pdf", "项目计划.docx", "main.py", "data.csv",
         "readme.md", "毕业论文.docx", "screenshot.png", "train_model.py", "课程表.xlsx", "meeting_minutes.docx",
         "requirements.txt", "实验记录.txt", "dataset.json", "presentation.pptx", "合同扫描件.pdf",
         "utils.py", "config.yaml", "周报.docx", "logo.png"]

BUTTONS = ["确认", "取消", "保存", "提交", "返回", "下一步", "删除", "登录", "确定", "Confirm", "Cancel",
           "Close", "Save", "Submit", "Back", "Next", "Delete", "Login", "OK", "Quit", "Apply", "Continue",
           "Skip", "Done", "Accept", "Reject", "关闭", "Exit", "搜索", "Search"]

NOISE = ["文件", "编辑", "视图", "帮助", "File", "Edit", "View", "Help", "Tools", "Home", "新建标签页", "历史记录",
         "书签", "下载", "Downloads", "Documents", "桌面", "最近", "收藏夹", "主页", "Logout", "Profile",
         "Notifications", "Tab 1", "12:30", "2024/05/01", "100%", "Wi-Fi", "Battery", "Untitled", "新建文件夹",
         "Properties", "Layout", "插入", "格式", "Window", "共享", "预览", "排序方式", "名称", "修改日期", "大小",
         "3 个项目", "Total 25 files", "Page 1 of 3", "已连接", "最近使用的文件", "快速访问", "System", "Add-ons"]

SEARCH_BOXES = ["Search Google or type a URL", "搜索", "搜索框", "Search", "在此搜索", "Search or enter address",
                "输入搜索内容", "Search the web", "搜索或输入网址", "Search Google"]
ADDR_BOXES = ["Search Google or type a URL", "搜索或输入网址", "输入网址", "Enter URL", "地址栏", "Address bar",
              "Search or enter address", "在此输入网址"]
CHAT_BOXES = ["输入聊天信息...", "Type a message", "输入消息", "Message", "说点什么...", "Write a message...",
              "请输入内容", "输入聊天内容", "Send a message", "输入消息..."]

# 要输入的内容（和 hard50 里的完全不重叠）
SEARCH_PAYLOADS = ["Python 基础", "今天股价", "附近的咖啡店", "机器学习入门", "深度学习 课程", "北京到上海高铁",
                   "汇率换算", "PyTorch 安装教程", "LoRA 微调", "最新电影", "周末天气预报", "红烧肉做法",
                   "Linux 命令大全", "多伦多 租房", "英语四级真题", "VS Code 快捷键", "GitHub 加速",
                   "best pizza near me", "how to learn rust", "weather tomorrow", "python list comprehension",
                   "cheap flights to tokyo", "react tutorial", "lofi study music", "stock market today",
                   "docker compose example", "transformers library docs", "laptop deals"]
CHAT_PAYLOADS = ["你好啊", "在吗", "晚上一起吃饭吗", "收到，谢谢", "我马上到", "明天开会改到三点", "好的没问题",
                 "稍等一下", "see you tomorrow", "thanks a lot", "running 10 minutes late", "sounds good",
                 "happy birthday!", "已发送文件请查收", "哈哈哈", "周五见", "我晚点回你", "ok 没问题"]
URL_PAYLOADS = ["www.baidu.com", "github.com", "https://huggingface.co", "www.bilibili.com", "mail.google.com",
                "www.wikipedia.org", "news.ycombinator.com", "arxiv.org", "colab.research.google.com"]

APP_TPL = ["点击 {t}", "点击{t}", "打开 {t}", "打开{t}", "进入 {t}", "寻找 {t}", "启动 {t}", "双击 {t}", "{t}",
           "点击图标 {t}", "打开 {t} 应用", "click {t}", "open {t}", "launch {t}", "find {t}", "double click {t}"]
FILE_TPL = ["打开 {t}", "点击 {t}", "进入 {t}", "双击 {t}", "{t}", "点击文本 {t}", "寻找 {t}", "打开链接 {t}",
            "{t} 标签页", "open {t}", "click {t}", "find {t}"]
BTN_TPL = ["点击 {t}", "点击{t}", "点击{t}按钮", "按下 {t} 按钮", "{t}", "点击屏幕右上角的{t}", "寻找{t}",
           "点击文本{t}", "click {t}", "press {t}", "click the {t} button"]

SEARCH_TPL = ["搜索 {p}", "在浏览器搜索 {p}", "搜一下{p}", "在搜索框输入 {p}", "点击搜索框并输入 {p}",
              "帮我搜索 {p}", "搜索一下 {p}", "查找 {p}", "search for {p}", "search {p}", "google {p}",
              "type {p} in the search box", "look up {p}"]
CHAT_TPL = ["发送 {p}", "发消息 {p}", "发送消息 {p}", "给对方发送 {p}", "回复 {p}", "打字回复 {p}",
            "在输入聊天信息处发送 {p}", "点击聊天输入框，打字说 {p}", "send {p}", "send message {p}",
            "reply {p}", "type {p} in the message box"]
URL_TPL = ["输入网址 {p}", "在地址栏输入 {p}", "访问 {p}", "打开网址 {p}", "go to {p}", "visit {p}",
           "enter URL {p}"]
GENERIC_TPL = ["输入 {p}", "打字 {p}", "type {p}"]

KIND_WEIGHTS = [("click_app", 0.22), ("click_file", 0.18), ("click_button", 0.20),
                ("type_search", 0.15), ("type_chat", 0.15), ("type_url", 0.05), ("type_generic", 0.05)]


def split_of(instruction):
    """按指令文本的哈希分到 train/val/test。"""
    h = int(hashlib.md5(instruction.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 88 else "test")


def _alias(rng, text, kind):
    """用户有时只说简称：Google Chrome -> Chrome；report.docx -> report。"""
    if kind == "app" and " " in text and rng.random() < 0.25:
        return rng.choice([text.split(" ")[0], text.split(" ")[-1]])
    if kind == "file" and "." in text and rng.random() < 0.3:
        return text.rsplit(".", 1)[0]
    return text


def _build_scene(rng, target_text, forbidden, extras=()):
    """围绕目标元素凑一屏 OCR 结果（带干扰项），返回 (elements, id_of_text)。"""
    n = rng.randint(8, 22)
    texts = [target_text]
    seen = {target_text.lower()}
    for e in extras:
        el = e.lower()
        if el in seen or any(f in el or el in f for f in forbidden):
            continue
        texts.append(e)
        seen.add(el)
    tries = 0
    while len(texts) < n and tries < 800:
        tries += 1
        pool = rng.choices([NOISE, APPS, FILES, BUTTONS], weights=[.45, .2, .15, .2])[0]
        d = rng.choice(pool)
        dl = d.lower()
        if dl in seen or any(f in dl or dl in f for f in forbidden):
            continue
        texts.append(d)
        seen.add(dl)
    rng.shuffle(texts)
    ids = sorted(rng.sample(range(0, 60 + n * 2), len(texts)))  # 故意跳号，模拟低置信度的文字被过滤掉
    elements = {}
    id_of_text = {}
    for i, t in zip(ids, texts):
        elements[str(i)] = {"text": t, "center": [rng.randint(40, 1880), rng.randint(15, 1060)]}
        id_of_text[t] = i
    return elements, id_of_text


def make_sample(rng):
    kind = rng.choices([k for k, _ in KIND_WEIGHTS], weights=[w for _, w in KIND_WEIGHTS])[0]

    if kind in ("click_app", "click_file", "click_button"):
        if kind == "click_app":
            t, tpl, akind = rng.choice(APPS), APP_TPL, "app"
        elif kind == "click_file":
            t, tpl, akind = rng.choice(FILES), FILE_TPL, "file"
        else:
            t, tpl, akind = rng.choice(BUTTONS), BTN_TPL, "btn"
        name = _alias(rng, t, akind)
        instr = rng.choice(tpl).format(t=name)
        forbidden = [t.lower(), name.lower()]
        elements, idmap = _build_scene(rng, t, forbidden)
        tid = idmap[t]
        target = {"action": "click", "target_id": tid, "reason": f"匹配目标文字「{t}」"}
    else:
        if kind == "type_search":
            box, p, tpl, dist = rng.choice(SEARCH_BOXES), rng.choice(SEARCH_PAYLOADS), SEARCH_TPL, CHAT_BOXES
        elif kind == "type_chat":
            box, p, tpl, dist = rng.choice(CHAT_BOXES), rng.choice(CHAT_PAYLOADS), CHAT_TPL, SEARCH_BOXES
        elif kind == "type_url":
            box, p, tpl, dist = rng.choice(ADDR_BOXES), rng.choice(URL_PAYLOADS), URL_TPL, CHAT_BOXES
        else:  # type_generic：场景里只放一个输入框，避免歧义
            pool = rng.choice([SEARCH_BOXES, CHAT_BOXES, ADDR_BOXES])
            box = rng.choice(pool)
            p = rng.choice(SEARCH_PAYLOADS + CHAT_PAYLOADS + URL_PAYLOADS)
            tpl, dist = GENERIC_TPL, []
        instr = rng.choice(tpl).format(p=p)
        extras = []
        if dist and rng.random() < 0.35:
            extras.append(rng.choice(dist))          # 干扰输入框
        forbidden = [box.lower()]
        if kind in ("type_search", "type_url"):
            forbidden += ["搜索", "search"]          # 避免出现语义上同样像"搜索入口"的干扰项
        elements, idmap = _build_scene(rng, box, forbidden, extras)
        tid = idmap[box]
        target = {"action": "type", "target_id": tid, "text": p, "reason": f"输入框「{box}」，提取输入内容"}

    return {
        "kind": kind,
        "instruction": instr,
        "elements": elements,
        "valid_ids": sorted(int(k) for k in elements),
        "target": target,
    }


def validate(rec):
    """标签得和场景对得上，对不上直接报错。"""
    tid = str(rec["target"]["target_id"])
    assert tid in rec["elements"], "target id 不在元素字典中"
    assert int(tid) in rec["valid_ids"]
    if rec["target"]["action"] == "type":
        assert rec["target"]["text"] in rec["instruction"], "输入内容未出现在指令中"
    assert all(len(v["text"]) < 30 for v in rec["elements"].values()), "元素文本过长（agent 会剪枝掉>=30字）"


def build(out_dir, n_train, n_val, n_test, seed):
    rng = random.Random(seed)
    quota = {"train": n_train, "val": n_val, "test": n_test}
    data = {k: [] for k in quota}
    skipped_hard = 0
    attempts = 0
    while any(len(data[k]) < quota[k] for k in quota) and attempts < 2_000_000:
        attempts += 1
        rec = make_sample(rng)
        if rec["instruction"] in HARD50_INSTRUCTIONS:
            skipped_hard += 1
            continue
        sp = split_of(rec["instruction"])
        if len(data[sp]) >= quota[sp]:
            continue
        validate(rec)
        rec["id"] = f"{sp}_{len(data[sp])}"
        data[sp].append(rec)

    # 泄漏检查
    instr_sets = {k: {r["instruction"] for r in v} for k, v in data.items()}
    assert not (instr_sets["train"] & instr_sets["val"]), "train/val 指令泄漏"
    assert not (instr_sets["train"] & instr_sets["test"]), "train/test 指令泄漏"
    assert not (instr_sets["val"] & instr_sets["test"]), "val/test 指令泄漏"
    for k, s in instr_sets.items():
        assert not (s & HARD50_INSTRUCTIONS), f"{k} 含 hard50 指令"

    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        save_jsonl(os.path.join(out_dir, f"{k}.jsonl"), v)
    save_jsonl(os.path.join(out_dir, "hard50.jsonl"), hard50_records())

    meta = {
        "seed": seed,
        "sizes": {k: len(v) for k, v in data.items()},
        "kinds": {k: dict(Counter(r["kind"] for r in v)) for k, v in data.items()},
        "unique_instructions": {k: len(s) for k, s in instr_sets.items()},
        "skipped_hard50_collisions": skipped_hard,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    m = build(a.out, a.train, a.val, a.test, a.seed)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"数据集已生成到 {a.out}/")
