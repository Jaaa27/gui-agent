"""第 5 周微调出来的 LoRA 版"大脑"，接口和 GUIAgent 一样，main.py 里用 --agent lora 切换。

微调的是纯文本模型，不看截图，image_path 只是为了接口一致才留着。
"""
import logging

from training.common import build_messages, parse_decision

log = logging.getLogger("gui_agent")


class GUIAgentLoRA:
    def __init__(self, adapter_path="outputs/lora_gui", base_model="Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens=128):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
            torch.float16 if use_cuda else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True,
                                                     device_map={"": 0} if use_cuda else None)
        self.model = PeftModel.from_pretrained(model, adapter_path).eval()

    def think(self, instruction, image_path, elements_dict, valid_ids=None):
        if valid_ids is None:
            valid_ids = [str(k) for k in elements_dict.keys()]
        rec = {
            "instruction": instruction,
            "elements": {str(k): v for k, v in elements_dict.items()},
            "valid_ids": [int(i) for i in valid_ids],
        }
        msgs = build_messages(rec, prompt="ft")
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        try:
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            raw = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            log.debug("lora 原始输出: %s", raw)
            decision = parse_decision(raw)
            if decision is None:
                log.warning("模型输出解析不出 JSON")
            return decision
        except Exception as e:
            log.warning("推理或解析失败: %s", e)
            return None
