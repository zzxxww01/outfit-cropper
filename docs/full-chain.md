# Full Chain

## Scope

当前仓库只保留以下链路：

`原始人物图 -> Gemini flatlay 生图 -> mask-first 切图 -> review.html`

不再保留：

- YOLO / DeepFashion2 提取链路
- 本地 GPU inpainting / matting 老链路
- GCP 占位脚本
- 旧 prompt 版本和旧工作流文档

## Models And Methods

### 1. Generation

- 模型：`gemini-3.1-flash-image-preview`
- 调用方式：Gemini REST API
- 客户端文件：[pipeline/nano_banana_client.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/nano_banana_client.py)
- 入口脚本：[api_pilot.py](/C:/Users/DELL/Desktop/outfit-cropper/api_pilot.py)
- 当前 prompt：[prompts/flatlay_v9.txt](/C:/Users/DELL/Desktop/outfit-cropper/prompts/flatlay_v9.txt)

说明：

- 这里的图像生成阶段是唯一使用大模型的环节。
- 当前默认 `temperature=0.2`。

### 2. Extraction

- 不使用 YOLO
- 不使用 SAM
- 不使用 DeepFashion2
- 当前使用的是经典 CV 方案：
  - 背景估计
  - LAB 空间前景分离
  - 连通域候选框
  - 局部 GrabCut 精修
  - 基于 mask 的透明 PNG 导出

核心文件：

- [pipeline/flatlay_segmenter.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_segmenter.py)
- [pipeline/flatlay_mask_extractor.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_mask_extractor.py)
- [segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)

## Full Pipeline

### Stage 1. Sample Selection

脚本：[pipeline/sample_selector.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/sample_selector.py)

逻辑：

1. 从 `normal_1068807_1070000` 读取原图
2. 如果已有 manifest，则直接复用
3. 否则按 seed 随机抽样并写入 manifest

### Stage 2. Prompt Loading

脚本：[pipeline/prompt_loader.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/prompt_loader.py)

逻辑：

1. 根据 `--prompt-version` 解析 prompt 文件名
2. 当前默认走 `flatlay_v9.txt`

### Stage 3. Flatlay Generation

脚本：[api_pilot.py](/C:/Users/DELL/Desktop/outfit-cropper/api_pilot.py)

内部调用：

- [pipeline/nano_banana_client.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/nano_banana_client.py)
- [utils/retry.py](/C:/Users/DELL/Desktop/outfit-cropper/utils/retry.py)

逻辑：

1. 读取原图
2. 加载 `flatlay_v9` prompt
3. 调 Gemini API
4. 保存 `flatlay.png`
5. 在 debug 模式下保存 `prompt.txt / request.json / response.json / review.json`

当前默认输入：

- 原图目录：`normal_1068807_1070000`

当前常用输出：

- `pilot_output/round_100_v9_debug`

### Stage 4. Flatlay Segmentation

底层脚本：[pipeline/flatlay_segmenter.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_segmenter.py)

逻辑：

1. 估计背景色
2. 在 LAB 空间做前景差分
3. 生成 `loose_mask` 和 `strong_mask`
4. 基于连通域提取候选区域
5. 对每个候选区域运行局部 GrabCut
6. 生成局部精修 mask
7. 用 mask 导出透明 PNG

关键点：

- 这里生成的 `bbox` 只用来定位候选区域和写元数据
- 最终单品图是按 `mask` 抠出的透明 PNG，不是按矩形直接裁

### Stage 5. Mask-First Post Processing

脚本：[pipeline/flatlay_mask_extractor.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_mask_extractor.py)

逻辑：

1. 丢弃极小噪点
2. 合并一双鞋为一个商品
3. 合并明显成对的小配饰
4. 给每个单品图加透明 padding
5. 重新编号
6. 生成 review 用的 `relayout.png`

当前默认 padding：

- `pad_ratio=0.08`
- `min_pad_px=24`

### Stage 6. Export

脚本：[segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)

每个样本输出：

- `source.jpg`
- `flatlay.png`
- `meta.json`
- `items/item_0.png`
- `items/item_1.png`
- ...

debug 模式下额外输出：

- `items/item_*_white.jpg`
- `items/item_*_mask.png`
- `relayout.png`

### Stage 7. Review

脚本：[build_debug_review_page.py](/C:/Users/DELL/Desktop/outfit-cropper/build_debug_review_page.py)

页面结构：

1. 第一行左边原图，右边 flatlay
2. 第二行开始展示所有切好的单品 PNG
3. 单品过多时自动换行
4. 透明区域用棋盘底显示

当前常用 review 页面：

- [pilot_output/round_100_v9_debug_extract_mask_padded/review.html](/C:/Users/DELL/Desktop/outfit-cropper/pilot_output/round_100_v9_debug_extract_mask_padded/review.html)

## Current Commands

### Generation

```bash
python api_pilot.py --input-dir normal_1068807_1070000 --output-dir pilot_output --sample-size 100 --round-id round_100_v9_debug --prompt-version v9
```

### Extraction

```bash
python segment_flatlay_mask.py --round-dir pilot_output/round_100_v9_debug --output-dir pilot_output/round_100_v9_debug_extract_mask_padded
```

### Review

```bash
python build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_padded --title "round_100_v9 mask-first padded review"
```

## Current Output Directories

- 原图目录：
  - [normal_1068807_1070000](/C:/Users/DELL/Desktop/outfit-cropper/normal_1068807_1070000)
- 最新生图结果：
  - [pilot_output/round_100_v9_debug](/C:/Users/DELL/Desktop/outfit-cropper/pilot_output/round_100_v9_debug)
- 最新切图结果：
  - [pilot_output/round_100_v9_debug_extract_mask_padded](/C:/Users/DELL/Desktop/outfit-cropper/pilot_output/round_100_v9_debug_extract_mask_padded)

## What Is Production Truth

生产真值只有两种：

- `flatlay.png`
- `items/*.png`

说明：

- `items/*.png` 是最终单品图
- `meta.json` 只做索引和调试
- `bbox_xyxy` 只做调试，不用于最终切图
