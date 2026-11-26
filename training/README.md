# Training (week 5)

LoRA fine-tuning of a text-only model (Qwen2.5-1.5B-Instruct) so that, given "OCR element dictionary + instruction", it outputs one JSON action.
Run from the project root on a Colab T4 or a local GPU with 8 GB+ VRAM.

    pip install -r training/requirements.txt
    python -m training.build_dataset --out data
    python -m training.train_lora --data data --out outputs/lora_gui
    python -m training.evaluate --data data --adapter outputs/lora_gui --out results/week5
    python -m training.report --results results/week5 --train_log outputs/lora_gui/training_log.json --out report

Training saves a checkpoint every 25 steps; if it is interrupted, rerunning the same command resumes from the last checkpoint.
`evaluate --backend mock` tests the pipeline without a GPU, but its numbers are random and must not be reported.

Use the trained adapter in the main program with `python main.py --agent lora --adapter outputs/lora_gui`.

## Why synthetic data

The Mind2Web sample (`data/dataset_sample_100.json`) only contains instructions and action counts; it has no labels for which elements are on screen and which one is the target, so it cannot supervise the "element dictionary + instruction -> action" mapping. `build_dataset.py` therefore synthesizes samples in the same format the perceiver produces: distractor elements, non-contiguous IDs, and Chinese and English instructions for opening apps and files, clicking buttons, searching, sending messages and entering URLs.

Leakage checks: the 50 hard50 instructions are excluded from every split, the payload vocabulary does not overlap with hard50, and each instruction text lands in exactly one split (by hash). The split and hard50 checks are asserted every time the data is built.

## Evaluation configurations

| Config | Model | Prompt |
|---|---|---|
| `base_v1_legacy` | base | original prompt, without the valid-ID list (how the week-4 test script called it) |
| `base_v1` | base | original prompt, with the valid-ID list (how `agent.py` calls it) |
| `base_v2` | base | revised prompt with rules and few-shot examples |
| `ft_ft` | base + LoRA | minimal fine-tuning prompt |
| `ft_v2` | base + LoRA | revised prompt (robustness to a prompt change) |

Metrics: valid JSON, action accuracy, target-ID accuracy, ID hallucination (ID outside the valid list), typed-text accuracy, and exact match (all of them correct). Two evaluation sets: `test` (synthetic, no instruction overlap with training) and `hard50` (the fixed week-4 set). Both measure decisions on an element dictionary, not end-to-end success on a real screen.

The prompts, instructions and generated reports are in Chinese, matching the rest of the internship work.
