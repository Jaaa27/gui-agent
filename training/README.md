# 训练（第 5 周）

LoRA 微调一个纯文本模型（Qwen2.5-1.5B-Instruct），让它在"OCR 元素字典 + 指令"的输入下输出 JSON 动作。
在项目根目录下运行，Colab T4 或 8GB 以上显存的本地 GPU。

    pip install -r training/requirements.txt
    python -m training.build_dataset --out data
    python -m training.train_lora --data data --out outputs/lora_gui
    python -m training.evaluate --data data --adapter outputs/lora_gui --out results
    python -m training.report --results results --train_log outputs/lora_gui/training_log.json --out report

训练每 25 步存一次检查点，中途断了重新运行同一条命令会自动接着跑。
`evaluate` 加 `--backend mock` 可以不用 GPU 测流水线，但这种结果不能写进报告。

训练好的适配器用 `python main.py --agent lora --adapter outputs/lora_gui` 接进主程序。
