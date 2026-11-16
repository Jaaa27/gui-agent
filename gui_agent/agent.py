"""多模态大模型这一路（llava，走本地 Ollama）：规则匹配不了的时候才会调到这里。"""
import base64
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

log = logging.getLogger("gui_agent")


class GUIAgent:
    def __init__(self, model_name="llava"):
        # 不开 format="json"，开了模型会死板地照抄示例。num_ctx 调大是怕长字典被截断
        self.llm = ChatOllama(model=model_name, temperature=0, num_ctx=8192)

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def think(self, instruction, image_path, elements_dict, valid_ids=None):
        """返回 {"action", "target_id", ...}，模型没按格式输出就返回 None。

        valid_ids 是剪枝之后允许选的 id，写进提示词里约束模型别瞎编数字。
        """
        img_base64 = self._encode_image(image_path)
        valid_id_str = ", ".join(valid_ids) if valid_ids else "无合法ID可用"

        system_prompt = f"""你是一个极其精确的 GUI 自动化控制中心。
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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": f"元素字典：\n{json.dumps(elements_dict, ensure_ascii=False)}\n\n指令：{instruction}"},
            ]),
        ]

        try:
            raw = str(self.llm.invoke(messages).content)
            log.debug("llava 原始输出: %s", raw)
            if "```json" in raw:
                return json.loads(raw.split("```json")[1].split("```")[0].strip())
            stripped = raw.strip()
            if stripped.startswith("{") and stripped.endswith("}"):    # 忘了加代码块也认
                return json.loads(stripped)
            log.warning("模型没有输出标准 JSON")
            return None
        except Exception as e:
            log.warning("推理或解析失败: %s", e)
            return None
