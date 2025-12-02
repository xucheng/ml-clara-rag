# 🔒 数据安全指南

## ⚠️ 重要提醒

本仓库的代码是公开的，但**训练数据应该保持私密**。

---

## 📋 敏感数据定义

以下数据文件可能包含企业敏感信息，**不应提交到公开仓库**：

### 🚫 禁止公开的文件类型

```
# 企业内部数据
data/internal/**/*.jsonl          # 内部训练数据
data/production/**/*.jsonl        # 生产环境数据
data/customer/**/*                # 客户数据
data/proprietary/**/*             # 专有数据

# 个人或企业知识库
knowledge_base/**/*.pdf           # 内部文档
knowledge_base/**/*.pptx          # 内部演示文稿
knowledge_base/**/*.docx          # 内部文档

# 敏感配置
.env                              # API密钥和凭证
config/production.yaml            # 生产配置
secrets/**/*                      # 密钥文件
```

---

## ✅ 当前仓库状态检查

### 已在 Git 中的数据文件

以下文件**已经被 Git 跟踪**，公开仓库前需要检查：

```bash
example/clara_training_data.jsonl
example/end_to_end_data.jsonl
example/instruction_data.jsonl
example/instruction_tuning_data.jsonl
example/pretrain_data.jsonl
example/raw_knowledge.jsonl
```

### ⚠️ 操作建议

**如果这些文件包含企业敏感数据，请按照下面的"清理敏感数据"步骤操作。**

**如果这些是示例/公开数据，可以安全地保留。**

---

## 🔧 推荐的目录结构

```
ml-clara/
├── example/                    # ✅ 公开示例数据（小规模、脱敏）
│   ├── pretrain_data.jsonl     # 10-100条示例
│   ├── instruction_data.jsonl
│   └── end_to_end_data.jsonl
│
├── data/                       # 🚫 企业内部数据（.gitignore）
│   ├── internal/               # 您的企业真实数据
│   │   ├── pretrain_data.jsonl
│   │   ├── instruction_data.jsonl
│   │   └── end_to_end_data.jsonl
│   └── README.md              # 说明数据格式
│
├── scripts/                    # ✅ 公开脚本（数据处理逻辑）
│   ├── extract_raw_data.py    # 提取原始数据
│   ├── synthesize_data.py     # 合成训练数据
│   └── run_data_pipeline.sh   # 完整数据流程
│
└── training_colab_complete.ipynb  # ✅ 公开训练模板
```

---

## 🧹 清理敏感数据

### 步骤 1：从 Git 历史中移除敏感文件

```bash
# ⚠️ 警告：这将从 Git 历史中完全移除文件
# 如果文件包含敏感数据，这是必要的

# 移除单个文件
git rm --cached example/clara_training_data.jsonl
git commit -m "Remove sensitive training data"

# 或批量移除所有敏感数据
git rm --cached example/*.jsonl
git commit -m "Remove all sensitive training data"
```

### 步骤 2：更新 .gitignore

确保 `.gitignore` 包含以下内容（已自动添加）：

```gitignore
# Sensitive training data
data/internal/**/*.jsonl
data/production/**/*
data/customer/**/*
data/proprietary/**/*

# If example/ contains sensitive data, add:
# example/*_data.jsonl

# Knowledge base with sensitive documents
knowledge_base/**/*.pdf
knowledge_base/**/*.pptx
knowledge_base/**/*.docx

# Sensitive configuration
.env
.env.local
.env.production
config/production.yaml
secrets/**/*
```

### 步骤 3：创建示例数据

创建**脱敏的小规模示例数据**用于公开演示：

```python
# scripts/create_example_data.py

import json

# 创建脱敏的示例数据（10-20条）
example_data = [
    {
        "data_type": "qa",
        "question": ["What is machine learning?"],
        "answers": ["Machine learning is a subset of AI..."],
        "docs": ["Machine learning enables computers to learn from data."]
    },
    # ... 更多示例
]

# 保存到 example/ 目录
with open('example/pretrain_data.jsonl', 'w') as f:
    for item in example_data[:10]:  # 只保留10条示例
        f.write(json.dumps(item) + '\n')
```

### 步骤 4：验证清理结果

```bash
# 检查敏感文件是否已被移除
git status

# 检查 .gitignore 是否生效
git check-ignore data/internal/pretrain_data.jsonl
# 应该输出文件路径，表示被忽略

# 查看即将推送的内容
git log --oneline --all
```

---

## 📝 使用企业数据的正确方式

### 方案 1：本地保存（推荐）

```bash
# 1. 将企业数据保存在 data/internal/ 目录
mkdir -p data/internal
mv /path/to/sensitive/data/*.jsonl data/internal/

# 2. data/internal/ 已在 .gitignore 中，不会被提交
git status  # 确认 data/internal/ 不在待提交列表中

# 3. 在 Colab 中使用 Google Drive 加载
# 参见 training_colab_complete.ipynb 的 Option B
```

### 方案 2：私有数据仓库

```bash
# 1. 创建单独的私有仓库存储数据
# 2. 代码仓库（公开）：ml-clara-rag
# 3. 数据仓库（私有）：ml-clara-data

# 在 Colab 中：
# !git clone https://github.com/xucheng/ml-clara-rag.git  # 公开代码
# !git clone https://{token}@github.com/xucheng/ml-clara-data.git  # 私有数据
```

### 方案 3：环境变量 + 密钥

```python
# 在 Colab 中使用 Google Drive + 加密
from google.colab import drive
drive.mount('/content/drive')

# 数据路径通过环境变量配置
import os
DATA_PATH = os.getenv('PRIVATE_DATA_PATH',
                      '/content/drive/MyDrive/ml-clara-data')
```

---

## ✅ 公开仓库前的检查清单

在将仓库设为公开之前，请确认：

- [ ] **检查 Git 历史**
  ```bash
  git log --all --full-history -- example/*.jsonl
  # 确认没有敏感数据在历史中
  ```

- [ ] **验证 .gitignore**
  ```bash
  git status
  # 确认 data/internal/ 等目录不在待提交列表
  ```

- [ ] **检查文件内容**
  ```bash
  grep -r "CONFIDENTIAL\|INTERNAL\|PROPRIETARY" example/
  # 确认没有敏感标记
  ```

- [ ] **测试示例数据**
  ```bash
  # 确认 example/ 中的数据是脱敏的
  wc -l example/*.jsonl
  # 应该只有少量示例（10-100条）
  ```

- [ ] **删除临时文件**
  ```bash
  find . -name "*.bak" -o -name "*.tmp" -o -name ".DS_Store" | xargs rm -f
  ```

- [ ] **审查所有 README 和文档**
  ```bash
  grep -r "company\|企业\|internal" *.md
  # 确认没有泄露企业信息
  ```

---

## 🚨 数据泄露应急响应

**如果已经推送了敏感数据到公开仓库：**

### 立即行动

```bash
# 1. 立即将仓库设为私有
# GitHub → Settings → Change visibility → Make private

# 2. 使用 BFG Repo-Cleaner 清理历史
# https://rtyley.github.io/bfg-repo-cleaner/

# 下载 BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 删除敏感文件
java -jar bfg-1.14.0.jar --delete-files '*.jsonl' --no-blob-protection .git

# 清理和强制推送
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all

# 3. 通知相关人员
# 4. 评估数据泄露影响
```

---

## 📚 推荐实践

### ✅ DO（推荐做法）

1. **代码和数据分离**
   - 代码仓库：公开
   - 数据存储：私有（Google Drive / 私有仓库）

2. **使用示例数据**
   - 在 `example/` 目录提供10-100条脱敏样本
   - 足够演示格式，但不泄露信息

3. **文档化数据格式**
   - 在 README 中说明数据格式
   - 提供数据生成脚本

4. **环境变量管理密钥**
   - API密钥通过 `.env` 文件（不提交）
   - 在 Colab 中使用 Secrets

### ❌ DON'T（禁止做法）

1. **不要提交真实企业数据**
2. **不要在代码中硬编码密钥**
3. **不要提交 `.env` 文件**
4. **不要在 Git 历史中保留敏感信息**

---

## 📞 需要帮助？

如果您不确定某个文件是否应该公开，遵循"宁可保守"原则：

1. **默认设为私密**
2. **咨询法务/合规团队**
3. **进行数据脱敏处理**
4. **使用本地数据存储**

---

**创建日期:** 2025-12-02
**维护者:** CLaRa Team
**更新频率:** 根据需要
