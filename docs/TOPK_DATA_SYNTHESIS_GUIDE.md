# CLaRa Top-K Data Synthesis Guide

这份指南说明如何修改 data pipeline 来生成支持 `generation_top_k > 1` 的训练数据。

## 问题诊断

### 当前问题

原始的 `scripts/synthesize_data.py` 在 **Line 151** 只为每个问题生成 1 个文档：

```python
qa_entry = {
    "question": qa.get("question"),
    "docs": [chunk],  # ❌ 只有 1 个文档
    "gold_answer": qa.get("answer")
}
```

这导致：
- `generation_top_k=5` 时会报错：`RuntimeError: selected index k out of range`
- 即使设置 `generation_top_k=1`，模型也无法学习文档排序和多文档融合能力

## 解决方案

### 方案 1：使用新脚本 `synthesize_data_topk.py` (推荐)

我创建了一个增强版的数据合成脚本，支持为每个问题生成多个候选文档。

#### 特性

- ✅ **可配置 top-k 值**：通过 `--target_top_k` 参数设置（1-10）
- ✅ **两种负样本策略**：
  - **随机采样**（默认）：随机选择其他文档块作为负样本
  - **硬负样本挖掘**（使用 `--use_embeddings`）：基于语义相似度选择最具迷惑性的负样本
- ✅ **自动文档混排**：正样本和负样本随机打乱，模拟真实检索结果
- ✅ **保持向后兼容**：`--target_top_k 1` 时行为与原脚本相同

#### 使用方法

##### 基础用法（随机负样本）

```bash
# 生成 top-k=5 的训练数据（随机负样本）
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url $BASE_URL \
    --model qwen-turbo \
    --target_top_k 5
```

**优点**：
- 🚀 速度快（不需要生成 embeddings）
- 💰 成本低（只调用 LLM API）
- ✅ 适合文档数量 > 20 的场景

**缺点**：
- 负样本质量一般（可能完全无关）
- 模型可能学会简单的表面特征区分

##### 高级用法（硬负样本挖掘）

```bash
# 生成 top-k=5 的训练数据（基于嵌入的硬负样本）
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model qwen-turbo \
    --target_top_k 5 \
    --use_embeddings
```

**优点**：
- 🎯 高质量负样本（语义相似但不包含答案）
- 🧠 训练更具挑战性的检索能力
- ✅ 模型学会细粒度的相关性判断

**缺点**：
- 🐌 速度慢（需要为每个 chunk 生成 embedding）
- 💸 成本高（额外的 embedding API 调用）
- ⚠️ 需要 OpenAI API（`text-embedding-3-small` 模型）

#### 输出示例

生成的 `end_to_end_data.jsonl` 格式：

```json
{
  "question": "How do I prevent overfitting in neural networks?",
  "docs": [
    "Neural networks consist of layers...",  // Weakly relevant (negative)
    "Overfitting occurs when models memorize training data...",  // Relevant (positive)
    "Batch normalization can accelerate training...",  // Moderately relevant
    "Learning rate controls the step size...",  // Weakly relevant (negative)
    "Dropout randomly drops neurons to prevent overfitting..."  // Relevant (positive)
  ],
  "gold_answer": "Methods to prevent overfitting: 1) Use more training data, 2) Apply dropout..."
}
```

**注意**：
- `docs` 数组长度 = `target_top_k`
- 文档顺序是随机的（模拟真实检索结果）
- 至少包含 1 个正样本（能回答问题的文档）

---

### 方案 2：修改原始脚本 `synthesize_data.py`

如果不想使用新脚本，可以手动修改 `synthesize_data.py`：

#### 修改步骤

1. **在主函数开头添加参数**：

```python
def main():
    args = parse_args()

    TARGET_TOP_K = 5  # ✅ 添加这一行

    # ... rest of code
```

2. **修改 Line 145-157**：

```python
# OLD CODE (只生成 1 个文档)
for qa in data["qa_pairs"]:
    qa_entry = {
        "question": qa.get("question"),
        "docs": [chunk],  # ❌ 单文档
        "gold_answer": qa.get("answer")
    }
    f_instruct.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
    f_e2e.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
```

```python
# NEW CODE (生成多个候选文档)
for qa in data["qa_pairs"]:
    # Select negative documents
    negative_docs = []
    available_chunks = [c for j, c in enumerate(chunks) if j != i]

    if len(available_chunks) >= (TARGET_TOP_K - 1):
        negative_docs = random.sample(available_chunks, TARGET_TOP_K - 1)
    else:
        negative_docs = available_chunks  # Use all available if not enough

    # Combine positive + negatives and shuffle
    candidate_docs = [chunk] + negative_docs
    random.shuffle(candidate_docs)

    qa_entry = {
        "question": qa.get("question"),
        "docs": candidate_docs,  # ✅ 多文档
        "gold_answer": qa.get("answer")
    }
    f_instruct.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
    f_e2e.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
```

3. **添加 import**：

在文件顶部添加：
```python
import random
```

---

## 训练配置调整

生成 top-k=5 数据后，需要相应调整训练脚本参数。

### Colab Notebook 修改

在 `training_colab_complete.ipynb` 的 **Cell 30 (Stage 3 Training)** 中：

```python
# Stage 3 配置
--stage stage2 \
--generation_top_k 5 \  # ✅ 改为 5（或你的 target_top_k 值）
```

### 本地训练脚本修改

如果使用 `openrlhf/cli/train_sft.py`：

```bash
deepspeed --module openrlhf.cli.train_sft \
   --stage stage2 \
   --generation_top_k 5 \  # ✅ 改为 5
   # ... other args
```

---

## 数据质量要求

### Top-K=1（单文档模式）

**每个样本需要**：
- 1 个高质量文档（包含答案的全部信息）

**数据合成难度**：简单
**适用场景**：简单 QA、快速原型

### Top-K=5（多文档模式）

**每个样本需要**：
- 1 个核心相关文档（包含主要答案）
- 2-3 个辅助相关文档（补充信息）
- 1-2 个困难负样本（主题相关但不回答问题）

**数据合成难度**：中等
**适用场景**：复杂推理、多来源信息整合

### 推荐配置

| 文档库规模 | 推荐 Top-K | 负样本策略 | 原因 |
|-----------|-----------|-----------|------|
| < 10 chunks | 1-2 | 随机 | 文档不够，强制 top-k=5 会重复 |
| 10-50 chunks | 3-5 | 随机 | 足够多样性 |
| > 50 chunks | 5-10 | 硬负样本 | 可以挖掘高质量负样本 |

---

## 完整工作流示例

### 场景：从零开始生成 top-k=5 训练数据

```bash
# Step 1: 提取原始知识（假设你有 PowerPoint 文件）
python scripts/extract_with_docling.py \
    --input_dir /path/to/pptx/files \
    --output_file example/raw_knowledge.jsonl

# Step 2: 使用新脚本合成 top-k=5 数据（硬负样本）
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --target_top_k 5 \
    --use_embeddings

# Step 3: 检查生成的数据
head -1 example/end_to_end_data.jsonl | python -m json.tool

# 输出应该类似：
# {
#   "question": "...",
#   "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],  // 5 个文档
#   "gold_answer": "..."
# }

# Step 4: 上传到 Colab 并训练
# 在 Colab 中运行 training_colab_complete.ipynb
# 确保 Stage 3 配置中 --generation_top_k 5
```

---

## 常见问题

### Q1: 数据中已经有多个文档，但训练还是报错？

**A**: 检查两点：
1. 确认 Colab 中代码是最新的（运行 `!git pull`）
2. 确认 `--generation_top_k` 参数与数据中 `docs` 数组长度一致

### Q2: 使用 `--use_embeddings` 时报错？

**A**: 确保：
- `--base_url` 设置为 `https://api.openai.com/v1`（OpenAI 官方 API）
- API key 有访问 `text-embedding-3-small` 模型的权限
- 如果使用其他 API provider（如 DashScope），移除 `--use_embeddings`

### Q3: 负样本质量不好，怎么办？

**A**: 三种方法：
1. 使用 `--use_embeddings` 启用硬负样本挖掘
2. 增加 `--chunk_size`（例如 1500），让每个 chunk 包含更完整的语义
3. 手动构造负样本（编辑生成的 JSONL 文件）

### Q4: Top-K 设置多大合适？

**A**: 根据场景：
- **快速验证**：top-k=1（单文档）
- **一般应用**：top-k=3-5（平衡性能和成本）
- **高难度任务**：top-k=8-10（需要多文档融合）

### Q5: 可以混合使用不同 top-k 的数据吗？

**A**: 可以！CLaRa 的自动调整逻辑（commit 1b99307）会处理：
```python
actual_top_k = min(self.generation_top_k, len(docs))
```
但建议保持一致以最大化训练效果。

---

## 总结

| 修改内容 | 位置 | 修改难度 |
|---------|------|---------|
| ✅ **使用新脚本** | `scripts/synthesize_data_topk.py` | 简单（推荐） |
| ⚠️ **修改原脚本** | `scripts/synthesize_data.py` L145-157 | 中等 |
| ✅ **调整训练参数** | `training_colab_complete.ipynb` Cell 30 | 简单 |

**最佳实践**：
1. 使用 `synthesize_data_topk.py --target_top_k 5 --use_embeddings`
2. 确保文档库 > 20 chunks
3. 训练时设置 `--generation_top_k 5`
4. 验证数据格式：每个样本的 `docs` 数组长度 = 5
