from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a debug-first HTML review page for one round.")
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Debug Review")
    parser.add_argument("--inline-assets", action="store_true", help="Embed images into the HTML so the file can be viewed standalone.")
    return parser.parse_args()


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported inline asset type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def asset_src(path: Path, *, round_dir: Path, inline_assets: bool) -> str:
    if inline_assets:
        return file_to_data_uri(path)
    return rel_path(path, round_dir)


def render_usage(usage_metadata: Dict[str, Any]) -> str:
    if not usage_metadata:
        return "<span>usage: n/a</span>"
    prompt_tokens = usage_metadata.get("promptTokenCount", "n/a")
    candidate_tokens = usage_metadata.get("candidatesTokenCount", "n/a")
    total_tokens = usage_metadata.get("totalTokenCount", "n/a")
    return (
        f"<span>prompt tokens: {html.escape(str(prompt_tokens))}</span>"
        f"<span>candidate tokens: {html.escape(str(candidate_tokens))}</span>"
        f"<span>total tokens: {html.escape(str(total_tokens))}</span>"
    )


def render_links(round_dir: Path, outfit_dir: Path, *, inline_assets: bool) -> str:
    if inline_assets:
        return '<span class="links-note">standalone html: local file links omitted</span>'
    links: List[str] = []
    for name in ["prompt.txt", "request.json", "response.json", "review.json", "meta.json", "relayout.png"]:
        path = outfit_dir / name
        if path.exists():
            links.append(
                f'<a href="{html.escape(rel_path(path, round_dir))}" '
                f'target="_blank" rel="noreferrer">{html.escape(name)}</a>'
            )
    return "".join(links)


def render_top_row(round_dir: Path, outfit_dir: Path, *, inline_assets: bool) -> str:
    figures: List[str] = []
    for label, filename in [("source", "source.jpg"), ("flatlay", "flatlay.png")]:
        path = outfit_dir / filename
        if path.exists():
            figures.append(
                f"""
                <figure class="hero-figure">
                  <figcaption>{label}</figcaption>
                  <img src="{html.escape(asset_src(path, round_dir=round_dir, inline_assets=inline_assets))}" alt="{html.escape(outfit_dir.name)} {label}">
                </figure>
                """
            )
    while len(figures) < 2:
        figures.append('<figure class="hero-figure hero-placeholder"></figure>')
    return "".join(figures[:2])


def render_items_grid(round_dir: Path, outfit_dir: Path, meta: Dict[str, Any], *, inline_assets: bool) -> str:
    items = meta.get("items", [])
    if not items:
        return '<div class="items-grid empty-grid"><div class="empty">no extracted items</div></div>'

    tiles: List[str] = []
    for item in items:
        image_path = outfit_dir / item["image_path"]
        if not image_path.exists():
            continue
        source_method = item.get("source_method", "")
        method_label = "mask" if source_method == "mask_seg_flatlay" else (source_method or "item")
        info_parts = [item.get("item_id", "item")]
        if item.get("class_name"):
            info_parts.append(item["class_name"])
        class_confidence = item.get("class_confidence")
        if class_confidence is not None:
            info_parts.append(f"conf {float(class_confidence):.2f}")
        if item.get("classification_stage"):
            info_parts.append(item["classification_stage"])
        if item.get("stage2_triggered"):
            info_parts.append("stage2")
        if item.get("topology_reranked"):
            info_parts.append("topology")
        area_ratio = item.get("area_ratio")
        if area_ratio is not None:
            info_parts.append(f"area {float(area_ratio):.4f}")

        secondary_meta: List[str] = []
        if item.get("decision_reason"):
            secondary_meta.append(str(item["decision_reason"]))
        if item.get("group_type") and item.get("group_type") != "single":
            secondary_meta.append(str(item["group_type"]))

        tiles.append(
            f"""
            <div class="item-tile">
              <img src="{html.escape(asset_src(image_path, round_dir=round_dir, inline_assets=inline_assets))}" alt="{html.escape(item.get('item_id', 'item'))}">
              <div class="item-meta">
                <span>{html.escape(" | ".join(info_parts))}</span>
                <span class="method method-{method_label}">{method_label}</span>
              </div>
              {f'<div class="item-submeta">{html.escape(" | ".join(secondary_meta))}</div>' if secondary_meta else ''}
            </div>
            """
        )
    return f'<div class="items-grid">{"".join(tiles)}</div>'


def build_entry(round_dir: Path, outfit_dir: Path, *, inline_assets: bool) -> str:
    request = read_json_if_exists(outfit_dir / "request.json")
    response = read_json_if_exists(outfit_dir / "response.json")
    review = read_json_if_exists(outfit_dir / "review.json")
    meta = read_json_if_exists(outfit_dir / "meta.json")

    outfit_id = outfit_dir.name
    prompt_version = request.get("prompt_version") or review.get("prompt_version") or "n/a"
    temperature = request.get("temperature", "n/a")
    review_status = review.get("status", "pending_review")
    pipeline = meta.get("pipeline", "")
    classification_model = meta.get("classification_model", "")
    classification_device = meta.get("classification_device", "")
    classification_stage2_model = meta.get("classification_stage2_model", "")
    classification_stage2_enabled = meta.get("classification_stage2_enabled")
    item_count = meta.get("item_count", 0)
    warnings = meta.get("warnings", [])
    stage2_count = sum(1 for item in meta.get("items", []) if item.get("stage2_triggered"))
    topology_count = sum(1 for item in meta.get("items", []) if item.get("topology_reranked"))

    badges: List[str] = [
        f"<span>prompt: {html.escape(str(prompt_version))}</span>",
        f"<span>temperature: {html.escape(str(temperature))}</span>",
        f"<span>review: {html.escape(str(review_status))}</span>",
    ]
    if pipeline:
        badges.append(f"<span>pipeline: {html.escape(str(pipeline))}</span>")
    if classification_model:
        badges.append(f"<span>classifier: {html.escape(str(classification_model))}</span>")
    if classification_device:
        badges.append(f"<span>device: {html.escape(str(classification_device))}</span>")
    if classification_stage2_enabled:
        badges.append(f"<span>stage2: {html.escape(str(classification_stage2_model))}</span>")
    if item_count:
        badges.append(f"<span>items: {html.escape(str(item_count))}</span>")
    if stage2_count:
        badges.append(f'<span class="badge-stage2">stage2 {stage2_count}/{item_count}</span>')
    if topology_count:
        badges.append(f'<span class="badge-topology">topology {topology_count}/{item_count}</span>')

    warnings_html = ""
    if warnings:
        warnings_html = '<div class="warnings">' + " | ".join(html.escape(str(w)) for w in warnings) + "</div>"

    return f"""
    <section class="card">
      <div class="meta">
        <div>
          <h2>{html.escape(outfit_id)}</h2>
          <div class="badges">
            {''.join(badges)}
          </div>
        </div>
        <div class="usage">
          {render_usage(response.get("usage_metadata", {}))}
        </div>
      </div>
      {warnings_html}
      <div class="top-row">
        {render_top_row(round_dir, outfit_dir, inline_assets=inline_assets)}
      </div>
      {render_items_grid(round_dir, outfit_dir, meta, inline_assets=inline_assets)}
      <div class="links">
        {render_links(round_dir, outfit_dir, inline_assets=inline_assets)}
      </div>
    </section>
    """


def build_html(round_dir: Path, title: str, *, inline_assets: bool) -> str:
    batch_report = read_json_if_exists(round_dir / "batch_report.json")
    outfit_dirs = sorted(
        [
            path
            for path in round_dir.iterdir()
            if path.is_dir() and ((path / "flatlay.png").exists() or (path / "meta.json").exists())
        ],
        key=lambda path: path.name,
    )
    entries = "\n".join(build_entry(round_dir, outfit_dir, inline_assets=inline_assets) for outfit_dir in outfit_dirs)

    summary_bits = [
        f"processed={batch_report.get('processed', 'n/a')}",
        f"succeeded={batch_report.get('succeeded', 'n/a')}",
        f"failed={batch_report.get('failed', 'n/a')}",
    ]
    if "pipeline" in batch_report:
        summary_bits.append(f"pipeline={batch_report.get('pipeline')}")
    if "classification_model" in batch_report:
        summary_bits.append(f"classifier={batch_report.get('classification_model')}")
    if batch_report.get("classification_stage2_model"):
        summary_bits.append(f"stage2={batch_report.get('classification_stage2_model')}")
    if "stage2_triggered_items" in batch_report:
        summary_bits.append(f"stage2_items={batch_report.get('stage2_triggered_items')}")
    if "topology_reranked_items" in batch_report:
        summary_bits.append(f"topology_items={batch_report.get('topology_reranked_items')}")

    standalone_note = "当前文件为单文件内联版本，可单独发送到其他设备查看。" if inline_assets else "当前文件使用相对路径引用图片，需与结果目录一起查看。"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --card: #fffdf9;
      --line: #d6cec0;
      --text: #1f1a14;
      --muted: #6a6258;
      --accent: #1d6c5f;
      --ok-bg: #e7f4ef;
      --cool-bg: #eef2ff;
      --cool-text: #3047a5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #efe7da 0%, var(--bg) 100%);
      color: var(--text);
      font: 14px/1.45 Georgia, "Noto Serif SC", "Songti SC", serif;
    }}
    .page {{
      width: min(1800px, calc(100vw - 32px));
      margin: 24px auto 56px;
    }}
    .hero {{
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 6px 0; color: var(--muted); }}
    .cards {{ display: grid; gap: 18px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 8px 30px rgba(78, 65, 44, 0.07);
    }}
    .meta {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .meta h2 {{ margin: 0; font-size: 20px; }}
    .badges, .usage, .links {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badges span, .usage span, .links a, .links-note {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
      color: var(--muted);
      text-decoration: none;
      font-size: 12px;
    }}
    .badge-stage2 {{
      background: var(--ok-bg) !important;
      color: var(--accent) !important;
      border-color: #b9d7cb !important;
    }}
    .badge-topology {{
      background: var(--cool-bg) !important;
      color: var(--cool-text) !important;
      border-color: #c7d1ff !important;
    }}
    .warnings {{
      margin-bottom: 12px;
      color: #a6422b;
      font-size: 12px;
    }}
    .top-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .hero-figure, .item-tile {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fcfaf5;
      overflow: hidden;
    }}
    .hero-placeholder {{
      min-height: 240px;
      background: rgba(255,255,255,0.4);
    }}
    .hero-figure figcaption {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      text-transform: lowercase;
    }}
    .hero-figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 9 / 16;
      object-fit: contain;
      background: #fff;
    }}
    .items-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .empty-grid {{
      display: block;
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 14px;
      color: var(--muted);
      background: rgba(255,255,255,0.5);
    }}
    .item-tile img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background:
        linear-gradient(45deg, rgba(0,0,0,0.03) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.03) 75%, rgba(0,0,0,0.03)),
        linear-gradient(45deg, rgba(0,0,0,0.03) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.03) 75%, rgba(0,0,0,0.03));
      background-position: 0 0, 12px 12px;
      background-size: 24px 24px;
    }}
    .item-meta {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      flex-wrap: wrap;
    }}
    .item-submeta {{
      padding: 0 10px 10px;
      color: var(--muted);
      font-size: 11px;
    }}
    .method {{
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid transparent;
    }}
    .method-mask {{
      background: var(--cool-bg);
      color: var(--cool-text);
      border-color: #c7d1ff;
    }}
    @media (max-width: 900px) {{
      .page {{ width: calc(100vw - 16px); margin: 8px auto 32px; }}
      .hero {{ padding: 16px; border-radius: 14px; }}
      .card {{ padding: 14px; border-radius: 14px; }}
      .top-row {{ grid-template-columns: 1fr; }}
      .items-grid {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(" | ".join(summary_bits))}</p>
      <p>{html.escape(standalone_note)}</p>
      <p>第一行左侧为原图，右侧为 flatlay。下方展示所有切好的透明单品 PNG，并标注最终类别、置信度，以及是否触发 stage2 二判或 topology 重排。</p>
    </section>
    <section class="cards">
      {entries}
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if not args.round_dir.exists():
        raise FileNotFoundError(f"Round directory does not exist: {args.round_dir}")
    default_name = "review_inline.html" if args.inline_assets else "review.html"
    output_path = args.output or (args.round_dir / default_name)
    output_path.write_text(
        build_html(args.round_dir, args.title, inline_assets=args.inline_assets),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
