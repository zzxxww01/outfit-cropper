# outfit-cropper

当前项目只保留一条可运行主链路：

`原始人物图 -> Gemini flatlay 生图 -> mask-first 切图 -> OpenCLIP 初判 + SigLIP 二判 + topology 重排 -> review.html`

## 保留内容

- 生图入口：[api_pilot.py](/C:/Users/DELL/Desktop/outfit-cropper/api_pilot.py)
- 切图与分类入口：[segment_flatlay_mask.py](/C:/Users/DELL/Desktop/outfit-cropper/segment_flatlay_mask.py)
- review 生成入口：[build_debug_review_page.py](/C:/Users/DELL/Desktop/outfit-cropper/build_debug_review_page.py)
- prompt 文件：[flatlay_v1.txt](/C:/Users/DELL/Desktop/outfit-cropper/prompts/flatlay_v1.txt)
- 最后一轮结果目录：`pilot_output/final_round_v1`

## 当前约定

- prompt 版本只支持 `v1`
- 生图模型默认是 `gemini-3.1-flash-image-preview`
- 切图主链路只保留 `mask-first`
- 分类主链路是 `OpenCLIP ViT-B-32 + SigLIP + topology`

## 主要目录

- `pipeline/`: 主链路依赖模块
- `utils/`: 通用 IO、日志和重试工具
- `prompts/`: 当前只保留 `flatlay_v1.txt`
- `pilot_output/final_round_v1/`: 保留的最后一轮结果
- `normal_1068807_1070000/`: 示例输入图目录

## 常用命令

生成 flatlay：

```bash
py -3.12 api_pilot.py --prompt-version v1 --round-id round_001_v1
```

切图并分类：

```bash
py -3.12 segment_flatlay_mask.py --round-dir pilot_output/round_001_v1 --output-dir pilot_output/final_round_v1 --minimal-output
```

生成 review：

```bash
py -3.12 build_debug_review_page.py --round-dir pilot_output/final_round_v1 --title "final_round_v1"
```
