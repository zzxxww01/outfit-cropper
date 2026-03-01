# outfit-cropper 技术规格（当前执行版：Phase 1）

## 1. 当前版本范围

本仓库当前只交付 **Phase 1（GPU 离线批处理）**：

- 输入：`input_images/` 目录中的本地图片（`.jpg/.jpeg/.png`）。
- 输出：`gpu_output/{outfit_id}/item_*.jpg` 与 `gpu_output/{outfit_id}/meta.json`。
- 不包含：FastAPI、`image_uri` 拉取、Gemini 分类、最终 `result.json` 聚合。

> `gcp_batch_process.py` 当前为 **TODO 占位**，仅输出 `todo_phase2` 结果文件，不调用 Vertex/Gemini。

## 2. 处理流程（仅 Phase 1）

### Step 1：文本噪声抹除（Inpainting）

- 目标：先抹除高饱和度贴纸/价格文字，减少后续分割“破洞”。
- 当前实现：OCR+Inpainting 采用可运行的本地回退实现（OpenCV）并预留模型接入点。
- 后续增强：替换为 Surya OCR + SD Inpainting/PowerPaint。

### Step 2：候选框检测与抠图（Matting）

- 目标：提取服装主体，去除干扰物并得到二值掩码。
- 当前实现：检测与分割采用可运行回退实现（边缘/轮廓 + GrabCut）并预留模型接入点。
- 后续增强：替换为 Florence-2 + SAM（含负向提示词）。

### Step 3：白底合成与裁剪

- 将掩码外像素填充为纯白 `(255,255,255)`。
- 基于目标框外扩 `10% padding` 后裁剪，输出 `item_{i}.jpg`。
- 生成 `meta.json`，记录相对坐标 `bbox=[ymin,xmin,ymax,xmax]`（0-1000）。

## 3. 显存安全红线（必须满足）

Step1 结束后、进入 Step2 前，必须执行显式释放：

1. `del OCR/Inpainting 模型对象`
2. `gc.collect()`
3. `torch.cuda.empty_cache()`

本项目代码已在 Step1->Step2 切换处强制执行该流程。

## 4. 批处理失败策略

- 每个关键步骤默认重试：`1 + --max-retries` 次。
- 单张图失败不终止整批：该图写入空 `meta.json` 并继续下一张。
- 全批次结束写出 `gpu_output/error_report.json`。

## 5. 目录结构（当前版）

```text
outfit-cropper/
├── gpu_batch_process.py
├── gcp_batch_process.py               # TODO(Phase2)
├── download_weights.py
├── requirements_gpu.txt
├── requirements_local.txt
├── pipeline/
│   ├── step1_inpainting.py
│   ├── step2_matting.py
│   └── step3_gemini.py                # TODO(Phase2)
├── schemas/
│   └── models.py
├── utils/
│   ├── gpu_memory.py
│   ├── image_utils.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   └── retry.py
└── README.md
```

## 6. Phase 2 计划（仅 TODO，不实现）

后续 `Phase 2` 将在本地执行：

- 读取 `gpu_output/*/meta.json + item_*.jpg + 原图上下文`
- 调用 GCP Vertex AI Gemini（Structured Outputs）
- 输出带 `item_type` 的最终 `result.json`

当前版本不实现以上能力，仅保留占位脚本和接口定义位置。

