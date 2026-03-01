# outfit-cropper (Phase 1 Milestone)

离线服装抠图批处理项目。当前里程碑只实现 **Phase 1 (GPU 视觉处理)**，Phase 2 的 Gemini 分类保留 TODO。

## Current Scope

- Implemented:
  - `gpu_batch_process.py` (Step1 + Step2 + Step3)
  - Step1: OCR-like text mask + inpainting fallback
  - Step2: candidate detection + segmentation fallback
  - Step3: white-background composite + 10% padding crop + `meta.json`
  - `download_weights.py` for Phase1 weight bootstrap
- TODO:
  - `gcp_batch_process.py` Gemini Structured Outputs classification
  - Final `result.json` assembly with `item_type`

## Directory Contract

```text
input_images/
  xhs_12345.jpg

gpu_output/
  xhs_12345/
    item_0.jpg
    item_1.jpg
    meta.json
  error_report.json
```

`meta.json`:

```json
{
  "outfit_id": "xhs_12345",
  "items": [
    {
      "item_image_path": "item_0.jpg",
      "bbox": [277.78, 260.42, 833.33, 470.83],
      "is_fallback": false,
      "confidence": 0.91
    }
  ]
}
```

## Quick Start

1. Install GPU dependencies:

```bash
pip install -r requirements_gpu.txt
```

2. Download weights:

```bash
python download_weights.py --checkpoints-dir checkpoints
```

3. Put `.jpg` images into `input_images/`, then run:

```bash
python gpu_batch_process.py --input-dir input_images --output-dir gpu_output --checkpoints-dir checkpoints
```

## Memory Safety Redline

The pipeline enforces explicit release between Step1 and Step2:

- `del step1 model objects`
- `gc.collect()`
- `torch.cuda.empty_cache()`

This is required to reduce OOM risk on 16/32GB V100.

## Failure Policy

- Retry each critical step up to `1 + --max-retries` attempts.
- If one image still fails, write empty `meta.json` for that outfit and continue batch.
- Final batch errors are summarized in `gpu_output/error_report.json`.

## Phase 2 Placeholder

`gcp_batch_process.py` is intentionally a TODO stub. It currently creates:

- `results/result.todo.json`

with `status=todo_phase2`.

