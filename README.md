# outfit-cropper

当前项目主链路：

`原始人物图 -> Gemini flatlay 生图 -> mask-first 切图 -> OpenCLIP 初判 + SigLIP 二判 + topology 重排 -> review.html`

完整说明见 [docs/full-chain.md](/C:/Users/DELL/Desktop/outfit-cropper/docs/full-chain.md)。

## 当前模型与方法

- 生图模型：`gemini-3.1-flash-image-preview`
- 当前 prompt：`prompts/flatlay_v9.txt`
- 切图：`OpenCV + GrabCut` 的 `mask-first` 管线
- 分类：
  - 第一阶段：本地 `OpenCLIP ViT-B-32`
  - 第二阶段：本地 `SigLIP base`
  - 最终裁决：基于单品 mask 的 topology 规则

## 8 类分类

- `Outerwear`
- `Top`
- `Bottom`
- `One_piece`
- `Shoes`
- `Bag`
- `Accessories`
- `Unknown`

## 当前保留结果

当前只保留最新一版结果目录：

- `pilot_output/round_100_v9_debug_extract_mask_classified_siglip`

该目录包含：

- `source.jpg`
- `flatlay.png`
- `meta.json`
- `items/item_*.png`
- `review.html`

## 常用命令

切图并分类：

```bash
py -3.12 segment_flatlay_mask.py --round-dir pilot_output/round_100_v9_debug --output-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --minimal-output
```

生成相对路径版 review：

```bash
py -3.12 build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --title "round_100_v9_debug_extract_mask_classified_siglip"
```

如需单文件内联版 review：

```bash
py -3.12 build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --title "round_100_v9_debug_extract_mask_classified_siglip" --inline-assets
```

## 主要文件

- `api_pilot.py`
- `segment_flatlay_mask.py`
- `build_debug_review_page.py`
- `pipeline/flatlay_segmenter.py`
- `pipeline/flatlay_mask_extractor.py`
- `pipeline/item_classifier.py`
- `prompts/flatlay_v9.txt`
