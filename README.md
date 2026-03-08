# outfit-cropper

当前仓库只保留一条主链路：

1. 用 Gemini API 基于原图生成一张 flatlay 图
2. 对 flatlay 图做纯 `mask-first` 单品切图
3. 生成 `review.html` 做人工检查

完整链路文档见：

- [docs/full-chain.md](/C:/Users/DELL/Desktop/outfit-cropper/docs/full-chain.md)

## Current Models

- 生图模型：`gemini-3.1-flash-image-preview`
- 当前 prompt：`prompts/flatlay_v9.txt`
- 切图阶段：不使用 YOLO，不使用其他检测模型，使用 `OpenCV + GrabCut` 的 `mask-first` 方案

## Main Entrypoints

生成 flatlay：

```bash
python api_pilot.py --input-dir normal_1068807_1070000 --output-dir pilot_output --sample-size 100 --round-id round_100_v9_debug --prompt-version v9
```

切 flatlay 单品：

```bash
python segment_flatlay_mask.py --round-dir pilot_output/round_100_v9_debug --output-dir pilot_output/round_100_v9_debug_extract_mask_padded
```

生成 review 页面：

```bash
python build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_padded --title "round_100_v9 mask-first padded review"
```

## Main Files

- `api_pilot.py`
- `segment_flatlay_mask.py`
- `build_debug_review_page.py`
- `pipeline/nano_banana_client.py`
- `pipeline/prompt_loader.py`
- `pipeline/sample_selector.py`
- `pipeline/flatlay_segmenter.py`
- `pipeline/flatlay_mask_extractor.py`
- `prompts/flatlay_v9.txt`

## Notes

- 生产切图依据是 `mask`，不是 `bbox`
- `meta.json` 中的 `bbox_xyxy` 只用于调试和排序
- 当前默认导出会给透明 PNG 加 padding：
  - `pad_ratio=0.08`
  - `min_pad_px=24`
