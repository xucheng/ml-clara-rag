# Image Extraction Modes

本文档说明 `extract_with_docling.py` 支持的两种图片提取模式。

## 背景问题

在数据合成过程中，我们需要将文档中的图片信息融合到文本中。Docling 在提取文档时会：
1. 识别并保存图片到文件
2. 在 markdown 输出中留下占位符 (`<!-- image -->`)

**问题**: 原始的占位符没有文件路径，无法与提取的图片建立对应关系。

## 解决方案：混合模式

`extract_with_docling.py` 现在支持两种模式：

### 模式 1: Dict Mode (默认，推荐)

**原理**: 使用 `export_to_dict()` 获取文档的结构化表示，精确定位图片在文档树中的位置。

**特点**:
- ✅ **图片位置精确**: 图片引用插入在文档结构中的原始位置
- ✅ **无占位符问题**: 不使用 `<!-- image -->` 标记
- ✅ **语义完整**: 图片与上下文文本的关系更清晰
- ⚠️ **新实现**: 基于 docling 的 `export_to_dict()` API

**输出示例**:
```
This is the first paragraph before the image.

This is text after the first image.

[IMAGE_REF: example/extracted_assets/doc_img_0.png]

Second page with another image below.

Text after the second image.
```

**使用方法**:
```bash
python scripts/extract_with_docling.py \
  --input_dir raw_data \
  --output_file example/raw_knowledge.jsonl \
  --image_output_dir example/extracted_assets \
  --extraction_mode dict  # 默认，可省略
```

### 模式 2: Markdown Mode (兼容模式)

**原理**: 使用传统的 `export_to_markdown()` 方法，所有图片引用追加到文档末尾。

**特点**:
- ✅ **兼容性好**: 与旧版行为一致
- ✅ **稳定**: 使用成熟的 markdown 导出 API
- ❌ **丢失位置信息**: 无法知道图片在文档中的原始位置
- ✅ **无占位符问题**: 已禁用 `<!-- image -->` 标记 (`image_placeholder=''`)

**输出示例**:
```
This is the first paragraph before the image.

This is text after the first image.

Second page with another image below.

Text after the second image.

--- Extracted Images ---
[IMAGE_REF: example/extracted_assets/doc_img_0.png]
```

**使用方法**:
```bash
python scripts/extract_with_docling.py \
  --input_dir raw_data \
  --output_file example/raw_knowledge.jsonl \
  --image_output_dir example/extracted_assets \
  --extraction_mode markdown
```

## 在数据管道中使用

### 默认使用 Dict Mode

`run_data_pipeline.sh` 默认使用 dict 模式:

```bash
bash scripts/run_data_pipeline.sh
```

### 切换到 Markdown Mode

通过环境变量指定:

```bash
EXTRACTION_MODE=markdown bash scripts/run_data_pipeline.sh
```

## 后续处理

无论使用哪种模式，后续的数据合成步骤都会：

1. **加载图片描述** (`load_image_descriptions()`):
   - 从 `raw_knowledge.jsonl` 中读取 vision LLM 生成的图片描述
   - 建立 `filename -> description` 的映射

2. **替换图片引用** (`replace_image_refs_with_descriptions()`):
   - 找到文档中的 `[IMAGE_REF: path]` 标记
   - 替换为对应的 vision LLM 描述
   - 移除 `<!-- image -->` 标记（如果有旧数据）
   - 移除 `--- Extracted Images ---` 分隔符

3. **生成训练数据**:
   - 合成的 QA 对会基于增强后的文档（包含图片语义）
   - 避免生成关于文件路径的问题

## 技术细节

### Dict Mode 实现原理

Docling 的 `export_to_dict()` 返回的文档结构:

```json
{
  "body": {
    "children": [
      {"$ref": "#/texts/0"},      // 文本段落
      {"$ref": "#/pictures/0"},   // 图片 (位置明确!)
      {"$ref": "#/texts/1"}       // 后续文本
    ]
  },
  "texts": [...],
  "pictures": [
    {
      "self_ref": "#/pictures/0",
      "prov": [{"page_no": 1, "bbox": {...}}],
      "image": {"uri": "data:image/png;base64,..."}
    }
  ]
}
```

通过递归遍历文档树，我们可以：
- 按正确顺序渲染所有元素
- 在图片元素处插入 `[IMAGE_REF: ...]`
- 从 base64 data URI 中提取并保存图片

### Markdown Mode 实现原理

1. 使用 `doc.export_to_markdown(image_placeholder='')` 禁用内联占位符
2. 遍历 `doc.pictures` 保存图片
3. 将 `[IMAGE_REF: ...]` 追加到文档末尾

## 选择建议

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 新项目 | **Dict** | 图片位置精确，语义更完整 |
| 生产环境 | **Dict** | 无占位符问题，输出更干净 |
| 测试/验证 | Markdown | 兼容旧流程，便于对比 |
| 遇到 dict 模式问题 | Markdown | 降级方案，确保流程可用 |

## 常见问题

**Q: 为什么有些图片没被识别？**
A: Docling 的图片检测不是 100% 准确，某些嵌入式图片可能无法识别。这是库本身的限制，两种模式都会受影响。

**Q: Dict 模式是否兼容旧的合成脚本？**
A: 完全兼容！`synthesize_data.py` 和 `synthesize_data_topk.py` 都能处理两种格式的输出。

**Q: 如何验证提取结果？**
A: 检查 `raw_knowledge.jsonl` 中的 `content` 字段，确认：
- Dict 模式: `[IMAGE_REF: ...]` 出现在文档中间
- Markdown 模式: `[IMAGE_REF: ...]` 出现在 `--- Extracted Images ---` 后

**Q: 生成的数据中还有 `<!-- image -->` 标记怎么办？**
A: 这是旧数据。重新运行提取脚本即可。两种模式都已禁用该占位符。

## 参考

- Docling 文档: https://github.com/DS4SD/docling
- 完整数据流程: `docs/DATA_SYNTHESIS_PIPELINE.md`
