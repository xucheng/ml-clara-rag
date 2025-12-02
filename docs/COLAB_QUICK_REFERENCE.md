# CLaRa Colab 快速参考

一页速查指南 - 打印出来随时查看

---

## ⚡ 5 分钟快速开始

```bash
1. 上传 training_colab_complete.ipynb 到 Colab
2. 运行时 → GPU (T4/V100/A100) + High RAM
3. 运行时 → 全部运行
4. 等待 1-2 小时
5. 下载模型到 Google Drive
```

---

## 📊 GPU 选择指南

| GPU | 显存 | 速度 | 批次 | 样本数 | 价格 |
|-----|------|------|------|--------|------|
| T4 | 16GB | 1x | 32 | 200 | 免费 |
| V100 | 32GB | 2x | 64 | 500 | Pro |
| A100 | 40GB | 4x | 128 | 1000+ | Pro+ |

**推荐：** 首次测试用 T4，正式训练用 A100

---

## 🎯 训练阶段速查

### Stage 1: 压缩预训练
```python
数据: pretrain_data.jsonl (QA 格式)
时间: T4=30min, A100=10min
输出: /content/checkpoints/clara_stage1
```

### Stage 2: 指令微调
```python
数据: instruction_data.jsonl (问答格式)
时间: T4=30min, A100=10min
输出: /content/checkpoints/clara_stage2
```

### Stage 3: 端到端
```python
数据: end_to_end_data.jsonl (同 Stage 2)
时间: T4=45min, A100=15min
输出: /content/checkpoints/clara_stage3_final ✅
```

---

## 📁 数据格式

### Stage 1 格式
```json
{
  "data_type": "qa",
  "question": ["问题"],
  "answers": ["答案"],
  "docs": ["文档"]
}
```

### Stage 2/3 格式
```json
{
  "question": "问题",
  "docs": ["文档1", "文档2"],
  "gold_answer": "答案"
}
```

---

## ❌ OOM 错误 - 快速修复

```python
# 修改配置单元格
TRAIN_BATCH_SIZE = 16      # 减小
MICRO_BATCH_SIZE = 1       # 保持
MAX_SAMPLES = 100          # 减少
MAX_LEN = 1024             # 减半
```

---

## ⚙️ 关键参数速查

```python
# 批次大小（调整以适应 GPU）
TRAIN_BATCH_SIZE = 32      # T4
TRAIN_BATCH_SIZE = 128     # A100

# 训练样本
MAX_SAMPLES = 200          # 测试
MAX_SAMPLES = 1000+        # 生产

# 学习率
LEARNING_RATE = 1e-4       # 默认
LEARNING_RATE = 5e-5       # 保守
LEARNING_RATE = 2e-4       # 激进

# 压缩率
COMPRESS_RATE = 32         # 推荐
COMPRESS_RATE = 64         # 更高压缩

# Flash Attention
USE_FLASH_ATTN = False     # 跳过（稳定）
USE_FLASH_ATTN = True      # 加速 15%
```

---

## 🚨 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| CUDA OOM | 显存不足 | 减小 BATCH_SIZE |
| Checkpoint not found | 上阶段失败 | 检查 /content/checkpoints/ |
| JSON decode error | 数据格式错误 | 验证 .jsonl 格式 |
| RuntimeError: no GPU | 未启用 GPU | 切换到 GPU 运行时 |
| Disconnected | 超时断开 | 定期点击页面 |

---

## 📤 模型导出 3 步

### 方法 1: 下载
```python
!zip -r model.zip /content/checkpoints/clara_stage3_final/
from google.colab import files
files.download('/content/checkpoints/model.zip')
```

### 方法 2: Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/checkpoints/clara_stage3_final \
  /content/drive/MyDrive/
```

### 方法 3: HuggingFace
```python
!pip install huggingface_hub
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.upload_folder(
    folder_path="/content/checkpoints/clara_stage3_final",
    repo_id="username/clara-model"
)
```

---

## 🧪 快速测试代码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/content/checkpoints/clara_stage3_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def ask(question, doc):
    prompt = f"Document: {doc}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
answer = ask("What is CLaRa?", "CLaRa is a RAG framework...")
print(answer)
```

---

## ⏱️ 预计时间

### T4 (16GB) - 免费
- Stage 1: 30-60 分钟
- Stage 2: 30-60 分钟
- Stage 3: 45-90 分钟
- **总计: 2-3 小时**

### A100 (40GB) - Pro
- Stage 1: 10-20 分钟
- Stage 2: 10-20 分钟
- Stage 3: 15-30 分钟
- **总计: 40-70 分钟**

---

## 💰 成本估算

| 配置 | GPU | 时间 | 数据 | 成本 |
|------|-----|------|------|------|
| 测试 | T4 | 2h | 200条 | 免费 |
| 小规模 | V100 | 1.5h | 500条 | $0.5 |
| 中规模 | A100 | 1h | 2K条 | $2 |
| 大规模 | A100 | 3h | 10K条 | $6 |

---

## 📞 获取帮助

1. **文档**: [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
2. **GitHub**: [ml-clara Issues](https://github.com/apple/ml-clara/issues)
3. **Flash Attn**: [训练指南 Q5](COLAB_TRAINING_GUIDE.md#q5-flash-attention-安装失败)

---

## ✅ 检查清单

**训练前**
- [ ] GPU 运行时已启用（T4/V100/A100）
- [ ] High-RAM 已选择
- [ ] 数据文件已准备（.jsonl 格式）
- [ ] 磁盘空间充足（30GB+）

**训练后**
- [ ] 3 个阶段都完成
- [ ] 检查点在 /content/checkpoints/
- [ ] 模型可以加载
- [ ] 推理测试通过
- [ ] 模型已备份

---

## 🎓 专业提示

1. **首次运行**: 用示例数据 + T4 GPU 测试流程
2. **数据准备**: 在本地生成，验证后上传
3. **正式训练**: Colab Pro + A100 + 完整数据
4. **保存模型**: 立即备份到 Google Drive
5. **监控训练**: 每 30 分钟检查一次进度

---

## 📊 性能对比

| 配置 | 每秒样本数 | 每小时样本数 |
|------|-----------|-------------|
| T4 + batch=32 | ~10 | ~36,000 |
| V100 + batch=64 | ~20 | ~72,000 |
| A100 + batch=128 | ~40 | ~144,000 |

---

## 🔗 相关链接

- 📄 [Paper](https://arxiv.org/abs/2511.18659)
- 💻 [GitHub](https://github.com/apple/ml-clara)
- 🤗 [Models](https://huggingface.co/probejie)
- 📚 [完整指南](COLAB_TRAINING_GUIDE.md)

---

**版本**: 1.0 | **日期**: 2025-12-01
**打印此页以供快速参考**
