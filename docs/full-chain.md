# Full Chain

## 项目概览

当前项目把人物穿搭图转成一张干净的 flatlay 图，再从这张 flatlay 图里切出每件单品，并补上 8 类分类结果。

完整链路：

`原始人物图 -> flatlay 生图 -> mask-first 切图 -> OpenCLIP 初判 -> SigLIP 二判 -> topology 规则裁决 -> review.html`

## 当前效果

- flatlay 生图使用固定 prompt 和固定温度，风格稳定
- 单品切图以 `mask` 为真值，导出透明 PNG
- 导出单品图会自动补透明 padding
- 每个单品会带 8 类分类结果
- 对 `Bottom / One_piece / Top` 的混淆，当前使用二阶段分类和 topology 规则做专门收敛
- review 页面可同时查看原图、flatlay 和所有切图结果

## 当前模型与方法

### 1. 生图阶段

- 模型：`gemini-3.1-flash-image-preview`
- 调用方式：Gemini REST API
- 当前 prompt：`prompts/flatlay_v9.txt`
- 默认温度：`0.2`

相关文件：

- [api_pilot.py](/C:/Users/DELL/Desktop/outfit-cropper/api_pilot.py)
- [pipeline/nano_banana_client.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/nano_banana_client.py)
- [pipeline/prompt_loader.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/prompt_loader.py)
- [pipeline/sample_selector.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/sample_selector.py)

### 2. 切图阶段

切图阶段使用 `mask-first` 方法：

- 背景估计
- LAB 空间前景分离
- 连通域候选提取
- 局部 GrabCut 精修
- 基于 mask 导出透明 PNG

切图后的后处理包括：

- 去除微小噪点
- 合并一双鞋
- 合并明显成对的小配饰
- 导出前补透明 padding

相关文件：

- [segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)
- [pipeline/flatlay_segmenter.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_segmenter.py)
- [pipeline/flatlay_mask_extractor.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_mask_extractor.py)

### 3. 分类阶段

分类阶段分三层：

1. `OpenCLIP ViT-B-32`
   - 全量单品初判
   - 负责 8 类快速分类

2. `SigLIP base`
   - 只处理服装类疑难样本
   - 重点收敛 `Bottom / One_piece / Top / Outerwear`

3. topology 规则裁决
   - 基于单品 mask 的轮廓特征
   - 重点解决“仅下半身被误判成 One_piece 或 Top”
   - 同时保护明显连衣裙和明显外套/上衣

相关文件：

- [pipeline/item_classifier.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/item_classifier.py)
- [segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)

## 全链路说明

### 1. 样本选择

脚本：

- [pipeline/sample_selector.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/sample_selector.py)

流程：

1. 读取原图目录中的图片列表
2. 根据 manifest 或随机种子确定处理样本

### 2. Prompt 加载

脚本：

- [pipeline/prompt_loader.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/prompt_loader.py)

流程：

1. 根据 `prompt-version` 定位 prompt 文件
2. 读取 prompt 内容
3. 交给生图阶段调用

### 3. Flatlay 生图

脚本：

- [api_pilot.py](/C:/Users/DELL/Desktop/outfit-cropper/api_pilot.py)

流程：

1. 读取原始人物图
2. 加载 `flatlay_v9` prompt
3. 调用 Gemini API 生成 `flatlay.png`

### 4. Flatlay 切图

脚本：

- [pipeline/flatlay_segmenter.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_segmenter.py)

流程：

1. 估计背景颜色
2. 构建前景 mask
3. 提取候选区域
4. 对候选区域做局部 GrabCut 精修
5. 导出单品透明 PNG

这里的 `bbox` 用于：

- 候选区域定位
- 排序
- 元数据记录

最终单品图仍然由 `mask` 决定。

### 5. 后处理

脚本：

- [pipeline/flatlay_mask_extractor.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/flatlay_mask_extractor.py)

流程：

1. 去掉微小噪点
2. 合并一双鞋
3. 合并成对小配饰
4. 给最终 PNG 加透明 padding
5. 重新编号

### 6. 单品分类

脚本：

- [pipeline/item_classifier.py](/C:/Users/DELL/Desktop/outfit-cropper/pipeline/item_classifier.py)

流程：

1. 对每个单品图做第一轮 OpenCLIP 分类
2. 对服装类疑难样本补做 SigLIP 二判
3. 用 topology 特征修正 `Bottom / One_piece / Top`
4. 输出 `class_name`、`class_confidence`、`classification_stage`
5. 额外记录：
   - `stage2_triggered`
   - `topology_reranked`
   - `decision_reason`

### 7. 结果导出

脚本：

- [segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)

当前保留的最新结果目录：

- `pilot_output/round_100_v9_debug_extract_mask_classified_siglip`

目录内容：

- `source.jpg`
- `flatlay.png`
- `meta.json`
- `items/item_*.png`
- `review.html`

## Review 页面

脚本：

- [build_debug_review_page.py](/C:/Users/DELL/Desktop/outfit-cropper/build_debug_review_page.py)

支持两种输出：

1. 相对路径版 `review.html`
   - 适合同目录查看

2. 单文件内联版
   - 使用 `--inline-assets`
   - 适合单独发送到其他设备
   - 会把图片全部嵌进 HTML，文件会明显变大

## 常用命令

### 切图并分类

```bash
py -3.12 segment_flatlay_mask.py --round-dir pilot_output/round_100_v9_debug --output-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --minimal-output
```

### 生成相对路径版 review

```bash
py -3.12 build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --title "round_100_v9_debug_extract_mask_classified_siglip"
```

### 生成单文件内联版 review

```bash
py -3.12 build_debug_review_page.py --round-dir pilot_output/round_100_v9_debug_extract_mask_classified_siglip --title "round_100_v9_debug_extract_mask_classified_siglip" --inline-assets
```
