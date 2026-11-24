# Desktop GUI Agent

AUthor： Zhenxin Zhang

A desktop GUI agent for Windows. You tell it what to do in one sentence (Chinese or English). It takes a screenshot, uses OCR to find and number the text on screen, decides where to click and what to type into which input box, drives the mouse and keyboard, and then checks whether the action actually took effect.

This is an 8-week internship project. The weekly reports are in `docs/`; the final report is `docs/final_report.md` (Word version: `docs/final_report.docx`). The reports are written in Chinese.

![Architecture](results/final/fig_architecture.png)

## How it works

1. **Decompose**: a long instruction such as "open Chrome then search weather then scroll down" is split into single steps.
2. **System commands**: "open Notepad / Calculator / File Explorer / Chrome / Edge" is first tried with a system command. If that fails, the agent falls back to the visual path.
3. **Perceive**: take a screenshot, recognize text with EasyOCR (or PaddleOCR), and give every text block an ID together with its center coordinates (Set-of-Mark).
4. **Decide**: rules come first. Click instructions are scored against on-screen text by keyword; type instructions first look for an input box. Only when the rules find no target are the screenshot and element list passed to a large model (local LLaVA, or the LoRA model fine-tuned in week 5), and the model may only choose from the listed IDs.
5. **Act**: click, type (non-ASCII text is pasted through the clipboard), press keys, scroll, wait.
6. **Verify**: compare the screen before and after the action. For type steps, the agent also rescans the screen to check that the typed text actually appears.
7. **Retry**: failures are handled according to their cause, e.g. try the next candidate, tell the model which elements have already been tried, or wait for the page to finish loading before scanning again. Optionally, the model can re-plan the remaining steps.

## Requirements

- Windows 10 / 11, Python 3.10+ (adaptive downscaling reads the display scale with `GetDpiForSystem`; older systems fall back to estimating it from the screenshot size)
- No GPU needed to run (rule-only path). LLaVA requires [Ollama](https://ollama.com). LoRA training needs a GPU with 8 GB+ VRAM or Colab.
- The UI language can be Chinese or English; the evaluation tasks list keywords for both.

## Installation

    pip install -r requirements.txt
    ollama pull llava                 # skip if you only use the rule path
    python scripts/check_env.py       # checks screen size, mouse movement and screenshots

Optional:

    pip install paddleocr             # to use PaddleOCR (--ocr paddle)

Training dependencies are listed separately in `training/requirements.txt`; see `training/README.md`.

## Quick start

    python main.py --dry-run          # first run: log only, no mouse or keyboard actions
    python main.py                    # normal run (LLaVA as fallback by default)
    python main.py --agent none       # rules only, no large model

Then type instructions at the prompt, for example:

    open notepad
    open Chrome then search McMaster courses
    open Edge then search weather then click Images
    open notepad then press ctrl+s
    open Chrome then search weather then scroll down
    wait 3 seconds

Chinese works the same way, e.g. `打开 Chrome 然后 搜索 天气`.

Type `calibrate` (or `校准`) to recalibrate the regions of the screen that change on their own (videos, animations), and `exit` to quit. Moving the mouse into any corner of the screen triggers an emergency stop.

After each task the agent prints the result of every step: `OK` means success, `OK?` means the action was performed but nothing on screen confirmed it (e.g. the typed text was not found afterwards), and `FAIL` is followed by the failure reason.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--agent llava\|lora\|none` | llava | Large model used when the rules find no target; `none` disables it |
| `--adapter <path>` | outputs/lora_gui | Location of the LoRA adapter for `--agent lora` |
| `--dry-run` | off | Log only, no mouse or keyboard actions |
| `--no-launch` | off | Do not launch apps with system commands; force the visual path |
| `--max-side auto\|0\|<pixels>` | auto | Upper limit of the screenshot's long side before OCR. `auto` derives it from the display scale so that text stays at least 13 px after downscaling; `0` disables downscaling |
| `--ocr easyocr\|paddle` | easyocr | OCR engine |
| `--calibrate` | off | At startup, detect screen regions that change on their own and ignore them during verification |
| `--on-fail abort\|continue\|replan` | abort | When a step finally fails: stop / continue with the remaining steps / let the large model re-plan from the current screen (requires a model) |
| `--no-text-check` | off | Do not rescan the screen for the typed text after typing |
| `--no-verify` | off | Do not check whether the screen changed after an action |
| `--retries N` | 3 | Maximum attempts per step |
| `--llm-timeout <seconds>` | 60 | Timeout for a single model call |
| `--llm-plan` | off | Let the large model split long instructions the rules cannot split |
| `--log-dir` | logs | Log directory |

## Logs

- Console: INFO level, shows every step in real time.
- `logs/agent.log`: DEBUG level, rotated at 2 MB, 3 backups kept.
- `logs/trace.jsonl`: one event per line (task start, step start, each attempt, step end, task end) with per-stage timings, for analysis with scripts afterwards.
- `som_test_result.png`: the annotated image from the latest scan (a box and an ID for every text block).

## Project layout

    main.py                     command-line entry point
    gui_agent/
      perceiver.py              screenshot, DPI awareness, adaptive downscaling, OCR, numbering,
                                screen fingerprint (with dynamic-region masking)
      matcher.py                intent detection (click/type/hotkey/scroll/wait), element scoring,
                                input-box detection, text extraction
      planner.py                instruction decomposition; model-based splitting and re-planning
      executor.py               step execution, verification, cause-based retries, system-command launch
      controller.py             mouse and keyboard (click, type, hotkey, scroll), dry-run controller
      agent.py                  LLaVA (via Ollama)
      agent_lora.py             week-5 LoRA fine-tuned model, same interface as agent.py
      monitor.py                logging and task monitoring
    training/                   week 5: synthetic data, LoRA training, evaluation, report generation
    evaluation/
      tasks.py                  20 evaluation tasks + 4 extended-action tasks
      runner.py                 real-machine evaluation (environment cleanup, manual confirmation, resume)
      sim.py / run_sim.py       ideal-perception simulation
      analyze.py                statistics (Wilson intervals; by level/app/resolution/failure stage), Excel, charts
      record.py                 fields of an evaluation record
    tests/
      test_units.py             unit tests for the core modules
      test_evaluation.py        tests for the evaluation framework
      test_week8.py             tests for the week-8 features
      fakes.py                  simulated screen, perceiver, controller and model
      run_routing_eval.py       rule routing, v1 vs v2
      run_fault_injection.py    fault-injection scenarios and random-fault simulation
      legacy_router.py          v1 routing logic (for comparison)
      cases.py                  adversarial cases
      make_charts.py            the two week-6 charts
    scripts/
      check_env.py              environment check
      bench_perception.py       real-machine OCR speed, recall, and idle-screen noise check
      study_threshold.py        verification-threshold study
      study_downscale.py        downscaling-cost study (tesseract)
      study_mask.py             dynamic-region masking study
      run_demo.py               task sequence for recording the demo video
      make_final_figures.py     figures for the final report
      fetch_mind2web.py         sample Mind2Web tasks from Hugging Face
    docs/                       week 6 and 7 reports, final report, demo storyboard
    results/                    experiment data and charts for each week

## Tests

No screen or GPU required:

    python -m unittest discover -s tests -t .       # 67 unit tests
    python -m tests.run_routing_eval                # rule-routing comparison
    python -m tests.run_fault_injection             # 26 fault scenarios + random-fault simulation

## Reproducing the experiments

| Experiment | Command | Output |
|---|---|---|
| Rule-routing comparison | `python -m tests.run_routing_eval` | `results/week6/routing_eval.json` |
| Fault injection, random-fault simulation | `python -m tests.run_fault_injection` | `results/week6/fault_injection.json` |
| Verification threshold | `python -m scripts.study_threshold` | `results/week6/threshold_study.json` |
| Downscaling cost (needs tesseract) | `python -m scripts.study_downscale` | `results/week6/downscale_cost.json` |
| Dynamic-region masking | `python -m scripts.study_mask` | `results/week8/mask_study.json` |
| Simulation of the 20 tasks | `python -m evaluation.run_sim --out results/week7_sim/sim_runs.csv` | same folder |
| Extended actions, before vs after | `python -m evaluation.run_sim --ext --out results/week8/sim_ext.csv` | same folder |
| Statistics and charts | `python -m evaluation.analyze --inputs <csv...> --out <folder>` | xlsx and charts |
| Final-report figures | `python -m scripts.make_final_figures` | `results/final/` |

## Real-machine evaluation

    python -m evaluation.runner --agent llava --repeat 3 --resolution-label 1920x1080-125%
    python -m evaluation.runner --agent llava --repeat 3 --no-launch --out results/eval_vision_only.csv
    python -m evaluation.analyze --inputs results/eval_runs.csv results/eval_vision_only.csv --out results/week7

Do not touch the mouse or keyboard while the evaluation is running. The 6 tasks whose automatic check is unreliable pause for manual confirmation (`--manual all` confirms every task, `--manual none` confirms none). If interrupted, rerun the same command and recorded runs are skipped. To compare resolutions, change the display settings and run again with a different `--resolution-label`.

Before evaluating, it is worth running:

    python -m scripts.bench_perception --idle-check     # changed cells between frames on an idle screen; should be well below 3
    python -m scripts.bench_perception --repeat 5       # speed and recall under different downscaling settings

## Demo

    python -m scripts.run_demo --dry-run                # rehearsal
    python -m scripts.run_demo --agent llava            # recording

Runs 7 demo tasks in order, printing a title and a countdown before each and closing the opened apps afterwards. The storyboard and recording tips are in `docs/demo_script.md`. The demo instructions are in Chinese.

## Training (week 5)

    python -m training.build_dataset --out data
    python -m training.train_lora --data data --out outputs/lora_gui
    python -m training.evaluate --data data --adapter outputs/lora_gui --out results/week5
    python -m training.report --results results/week5 --train_log outputs/lora_gui/training_log.json --out report

Use the trained adapter with `python main.py --agent lora --adapter outputs/lora_gui`. See `training/README.md` for details.

## Troubleshooting

- **Clicks are offset**: usually caused by Windows display scaling. The agent enables DPI awareness at startup and converts coordinates; if clicks are still off, make sure you started it from `main.py` (DPI awareness must be set before any screenshot object is created).
- **Chinese text is not typed**: install `pyperclip`. Non-ASCII text is pasted through the clipboard, and the original clipboard content is restored afterwards.
- **Always reports "screen did not change"**: check the idle-screen noise with `--idle-check`; if the screen has videos or animations, use `--calibrate` or type `calibrate`.
- **Model fails to initialize**: make sure Ollama is running and `ollama pull llava` has been done. On failure the agent automatically falls back to the rule path.
- **Small text is not recognized**: on 2K/4K screens keep `--max-side auto` or set it to `0`; do not set it too low.

## Known limitations

- Only text is recognized. Icons, controls without text, and empty input areas (e.g. Notepad's blank editing area) cannot be located.
- Verification can show that "the screen changed" and "the typed text appeared", but not that a click hit the right target.
- Re-planning depends on a large model and relies only on the text on screen.
- The threshold, downscaling and masking studies were run on synthetic screens; on a new machine, re-check them with the tools above.
