# CLaRa 数据合成管道详细说明

本文档详细描述CLaRa从原始文档到训练数据的完整数据合成流程。

---

## 目录

1. [流程概览](#流程概览)
2. [阶段一：文档提取](#阶段一文档提取)
3. [阶段二：图片描述生成](#阶段二图片描述生成)
4. [阶段三：QA数据合成](#阶段三qa数据合成)
5. [输出数据格式](#输出数据格式)
6. [完整示例](#完整示例)

---

## 流程概览

```
原始文档 (raw_data/)
    ├── PDF/DOCX/PPTX 文件
    └── 独立图片文件 (*.jpg, *.png)
            ↓
[阶段1: 文档提取] extract_with_docling.py
            ↓
中间文件: raw_knowledge.jsonl
    ├── 文档文本 + [IMAGE_REF] 标记
    └── 提取的图片 (extracted_assets/)
            ↓
[阶段2: 图片描述] extract_images.py (Vision LLM)
            ↓
增强的 raw_knowledge.jsonl
    ├── 文档文本 + [IMAGE_REF] 标记
    ├── 嵌入图片描述 (source_type: "image", extracted_assets/)
    └── 独立图片描述 (source_type: "image", raw_data/)
            ↓
[阶段3: QA合成] synthesize_data.py / synthesize_data_topk.py
            ↓
训练数据 (example/)
    ├── pretrain_data.jsonl      (Stage 1: Compression)
    ├── instruction_data.jsonl   (Stage 2: Instruction)
    └── end_to_end_data.jsonl    (Stage 3: End-to-End)
```

---

## 阶段一：文档提取

**脚本**: `scripts/extract_with_docling.py`

**功能**: 使用IBM Docling提取文档内容和嵌入的图片

### 输入

- **输入目录**: `raw_data/` (支持递归遍历子目录)
- **支持格式**: PDF, DOCX, PPTX
- **支持内容**:
  - 文本内容（保留布局结构）
  - 表格（转换为Markdown格式）
  - 嵌入的图片（提取并保存）

### 处理流程

1. **递归扫描目录**
   ```python
   for root, dirs, files in os.walk(input_dir):
       for file in files:
           if file.endswith(('.pdf', '.docx', '.pptx')):
               process_document(file)
   ```

2. **文档转换**
   - 使用Docling的 `DocumentConverter` 进行高质量布局解析
   - 保留文档结构（标题、段落、列表）
   - 表格转换为Markdown格式

3. **图片提取**
   - 自动提取文档中的嵌入图片
   - 保存到 `extracted_assets/` 目录
   - 命名规则: `{文档名}_img_{序号}.png`
   - 在文档内容中插入占位符: `[IMAGE_REF: extracted_assets/xxx.png]`

4. **图片列表追加**
   - 文档末尾添加 `--- Extracted Images ---` 分隔符
   - 列出所有提取的图片路径

### 输出格式

**文档条目**（JSONL格式）:
```json
{
  "file_path": "raw_data/technical_architecture.docx",
  "filename": "technical_architecture.docx",
  "content": "# System Architecture\n\n## Overview\n\nThe system follows a microservices architecture pattern...\n\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_0.png]\n\n## Core Components\n\n...\n\n--- Extracted Images ---\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_0.png]\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_1.png]",
  "product_area": "Technical Documentation",
  "extracted_images": [
    "example/extracted_assets/technical_architecture_img_0.png",
    "example/extracted_assets/technical_architecture_img_1.png"
  ]
}
```

### 关键参数

```bash
python scripts/extract_with_docling.py \
    --input_dir raw_data \
    --output_file example/raw_knowledge.jsonl \
    --image_output_dir example/extracted_assets
```

### 输出文件

- `example/raw_knowledge.jsonl`: 文档文本 + IMAGE_REF标记
- `example/extracted_assets/*.png`: 提取的图片文件

---

## 阶段二：图片描述生成

**脚本**: `scripts/extract_images.py`

**功能**: 使用Vision LLM为图片生成语义描述

### 两次处理（Two-Pass Approach）

#### Pass 1: 处理独立图片
```bash
python scripts/extract_images.py \
    --input_dir raw_data \
    --output_file example/raw_knowledge.jsonl \
    --model qwen-vl-max
```

- **目标**: `raw_data/` 中的独立图片文件（*.jpg, *.png）
- **输出**: 作为独立条目追加到 `raw_knowledge.jsonl`

#### Pass 2: 处理嵌入图片
```bash
python scripts/extract_images.py \
    --input_dir example/extracted_assets \
    --output_file example/raw_knowledge.jsonl \
    --model qwen-vl-max
```

- **目标**: 从文档提取的图片 (`extracted_assets/`)
- **输出**: 作为独立条目追加到 `raw_knowledge.jsonl`

### 处理流程

1. **图片验证与预处理**
   ```python
   def validate_and_convert_image(image_path):
       # 使用Pillow验证图片有效性
       with Image.open(image_path) as img:
           # 转换为RGB（去除alpha通道）
           if img.mode in ('RGBA', 'P', 'LA'):
               img = img.convert('RGB')

           # 调整大小（如果超过2048x2048）
           if max(img.size) > 2048:
               img.thumbnail((2048, 2048))

           # 转换为JPEG base64
           return base64.b64encode(img_bytes).decode('utf-8')
   ```

2. **Vision LLM调用**
   ```python
   def analyze_image(client, model, image_path):
       base64_image = validate_and_convert_image(image_path)

       response = client.chat.completions.create(
           model=model,  # qwen-vl-max, gpt-4o
           messages=[{
               "role": "user",
               "content": [
                   {"type": "text", "text": VISION_PROMPT},
                   {"type": "image_url", "image_url": {
                       "url": f"data:image/jpeg;base64,{base64_image}"
                   }}
               ]
           }],
           max_tokens=1024
       )
       return response.choices[0].message.content
   ```

3. **提示词模板**（VISION_PROMPT）
   ```
   Please analyze this image and provide:

   1. Title/Topic: What is this image about?
   2. Text Content: Extract all visible text (labels, titles, captions)
   3. Visual Elements: Describe diagrams, charts, UI components, etc.
   4. Key Information: Main concepts, processes, or data shown

   Focus on business/technical content that would be useful for RAG.
   ```

4. **断点续传支持**
   - 读取已有的 `raw_knowledge.jsonl`
   - 记录已处理的图片文件名
   - 跳过已处理的图片，只处理新增图片

### 输出格式

**图片描述条目**（追加到 raw_knowledge.jsonl）:

```json
{
  "file_path": "example/extracted_assets/technical_architecture_img_0.png",
  "filename": "technical_architecture_img_0.png",
  "content": "[IMAGE DESCRIPTION of technical_architecture_img_0.png]\n1. **Title/Topic**:\nMicroservices Architecture Diagram\n\n2. **Text Content**:\n- API Gateway\n- Service Registry\n- Authentication Service\n- User Service\n- Order Service\n- Database Layer\n\n3. **Visual Elements**:\nThe diagram shows a microservices architecture with:\n- API Gateway as the entry point\n- Service Registry for service discovery\n- Multiple independent services (Authentication, User, Order)\n- Each service connected to its own database\n- Communication via REST APIs\n\n4. **Key Information**:\nThis architecture enables scalability, independent deployment, and fault isolation. Each service can be scaled independently based on load.",
  "product_area": "extracted_assets",
  "source_type": "image"
}
```

**独立图片条目**:
```json
{
  "file_path": "raw_data/Workflows/order_processing_flow.jpg",
  "filename": "order_processing_flow.jpg",
  "content": "[IMAGE DESCRIPTION of order_processing_flow.jpg]\n1. **Title/Topic**:\nOrder Processing Workflow\n\n2. **Text Content**:\n- Start\n- Receive Order\n- Validate Inventory\n- Process Payment\n- Ship Order\n- Send Confirmation\n- End\n\n3. **Visual Elements**:\nA flowchart showing the complete order processing pipeline with decision points for inventory availability and payment validation.\n\n4. **Key Information**:\nThe workflow includes validation checkpoints, error handling paths, and notification triggers at key stages.",
  "source_type": "image"
}
```

### 关键参数

```bash
# 环境变量
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VISION_MODEL="qwen-vl-max"

# 支持的Vision模型
# - qwen-vl-max (阿里云DashScope)
# - gpt-4o (OpenAI)
# - gpt-4o-mini (OpenAI)
```

### 错误处理

- **损坏的图片**: 自动跳过，输出警告信息
- **API限流**: 每次调用后sleep 0.5秒
- **超时/网络错误**: 记录错误，继续处理下一张图片

---

## 阶段三：QA数据合成

**脚本**:
- `scripts/synthesize_data.py` (基础版，top-k=1)
- `scripts/synthesize_data_topk.py` (高级版，支持top-k>1)

**功能**: 使用文本LLM生成双语QA对

### 核心处理流程

#### 3.1 加载图片描述

```python
def load_image_descriptions(input_file: str) -> Dict[str, str]:
    """加载所有图片描述到内存"""
    image_descs = {}

    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('source_type') == 'image':
                # 文件名 -> 完整描述
                image_descs[entry['filename']] = entry['content']

    return image_descs

# 执行
image_descriptions = load_image_descriptions("example/raw_knowledge.jsonl")
# 结果: {"technical_architecture_img_0.png": "[IMAGE DESCRIPTION] ...", ...}
```

#### 3.2 读取并分类文档

```python
all_texts = []

with open(input_file, 'r') as f:
    for line in f:
        entry = json.loads(line)

        if entry.get("source_type") == "image":
            # 嵌入图片: 跳过（会通过IMAGE_REF合并）
            if "extracted_assets" in entry.get("file_path", ""):
                continue
            # 独立图片: 保留为独立文档

        if "content" in entry:
            all_texts.append((entry["filename"], entry["content"]))

# 结果:
# all_texts = [
#     ("technical_architecture.docx", "文本内容...[IMAGE_REF: ...]..."),
#     ("order_processing_flow.jpg", "[IMAGE DESCRIPTION] ..."),  # 独立图片
# ]
```

#### 3.3 文档分块（Chunking）

```python
chunks = []

for filename, text in all_texts:
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        # 计算结束位置
        end_pos = min(current_pos + chunk_size, text_len)  # chunk_size=1000

        # 如果不是文档末尾，在换行符处切分（避免切断句子）
        if end_pos < text_len:
            search_window = text[end_pos-100:end_pos]
            last_newline = search_window.rfind('\n')
            if last_newline != -1:
                end_pos = (end_pos - 100) + last_newline + 1

        chunk_text = text[current_pos:end_pos].strip()
        if len(chunk_text) > 50:  # 过滤过短的chunk
            chunks.append(chunk_text)

        current_pos = end_pos

# 结果: 10个文档 -> 150个chunks (平均每个文档15个chunks)
```

#### 3.4 图片描述融合（核心功能）

```python
def replace_image_refs_with_descriptions(chunk: str, image_descs: Dict[str, str]) -> str:
    """将[IMAGE_REF]标记替换为实际的图片描述"""
    import re

    # 1. 删除末尾的图片索引部分
    if "--- Extracted Images ---" in chunk:
        chunk = chunk.split("--- Extracted Images ---")[0]

    # 2. 替换所有[IMAGE_REF: path]为实际描述
    def replace_ref(match):
        full_path = match.group(1)  # "example/extracted_assets/xxx.png"
        filename = os.path.basename(full_path)  # "xxx.png"

        if filename in image_descs:
            # 找到对应描述，返回完整内容
            return f"\n\n{image_descs[filename]}\n\n"
        else:
            # 未找到描述，使用占位符
            return "[图片]"

    chunk = re.sub(r'\[IMAGE_REF:\s*([^\]]+)\]', replace_ref, chunk)

    # 3. 清理多余空白
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)

    return chunk.strip()
```

**示例转换**:

```
输入chunk:
"""
# System Architecture

The system follows a microservices architecture pattern.

[IMAGE_REF: example/extracted_assets/technical_architecture_img_0.png]

## Core Components

The architecture includes the following services...
"""

输出enriched_chunk:
"""
# System Architecture

The system follows a microservices architecture pattern.

[IMAGE DESCRIPTION of technical_architecture_img_0.png]
1. **Title/Topic**: Microservices Architecture Diagram
2. **Text Content**:
- API Gateway
- Service Registry
- Authentication Service
- User Service
- Order Service
3. **Visual Elements**:
The diagram shows a microservices architecture with API Gateway as entry point...

## Core Components

The architecture includes the following services...
"""
```

#### 3.5 LLM生成QA对

```python
def generate_data(client: OpenAI, model: str, chunk: str, image_descs: Dict[str, str]) -> Dict:
    # 1. 图片描述融合
    enriched_chunk = replace_image_refs_with_descriptions(chunk, image_descs)

    # 2. 调用LLM
    response = client.chat.completions.create(
        model=model,  # qwen-turbo, gpt-4o-mini
        messages=[
            {"role": "system", "content": "You are a helpful data synthesis assistant. Always output valid JSON."},
            {"role": "user", "content": PROMPT_TEMPLATE.replace("{{TEXT_CHUNK}}", enriched_chunk)}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=2048
    )

    # 3. 解析返回的JSON
    return json.loads(response.choices[0].message.content)
```

#### 3.6 提示词模板（PROMPT_TEMPLATE）

```
# Role
You are a data synthesis expert for a bilingual (English/Chinese) RAG system called CLaRa.

# Input
[TEXT CHUNK]:
{{TEXT_CHUNK}}

# Important Instructions
- Focus on actual business content (features, processes, requirements, architecture)
- You can generate questions about images when they illustrate business concepts
  (e.g., architecture diagrams, process flows, UI mockups)
- Do NOT generate questions about technical artifacts:
  × File paths or folder names ("What is in extracted_assets?")
  × Image file names ("What is xxx.png?")
  × Document structure markers
- Image descriptions are provided inline to help you understand complete context

# Task
Generate a JSON object with:
1. "dense_summary": Rewritten paragraph with ALL key information (50-80% of original length)
2. "qa_pairs": List of 3-5 question-answer pairs
   - CRITICAL: Generate questions in BOTH English and Chinese
   - If text is Chinese, include 1-2 English questions
   - If text is English, include 1-2 Chinese questions
   - Questions must be self-contained (no "according to the text", "he said", etc.)
   - At least one fact-based question and one reasoning question

# Output Format (JSON)
{
    "dense_summary": "string",
    "qa_pairs": [
        {
            "type": "fact",
            "question": "string (in English or Chinese)",
            "answer": "string (in same language as text)"
        },
        {
            "type": "cross_lingual",
            "question": "string (in OPPOSITE language of text)",
            "answer": "string (in same language as text)"
        }
    ]
}
```

#### 3.7 LLM返回示例

```json
{
  "dense_summary": "The system implements a microservices architecture with an API Gateway serving as the entry point. Core components include Service Registry for discovery, Authentication Service for security, and domain services (User, Order) each with dedicated databases. This design enables independent scaling, fault isolation, and flexible deployment strategies.",

  "qa_pairs": [
    {
      "type": "fact",
      "question": "What are the main components in the microservices architecture?",
      "answer": "The main components are API Gateway (entry point), Service Registry (service discovery), Authentication Service, User Service, and Order Service. Each service has its own database layer."
    },
    {
      "type": "cross_lingual",
      "question": "微服务架构中的API网关有什么作用？",
      "answer": "The API Gateway serves as the single entry point for all client requests, handling routing, authentication, and load balancing to downstream services."
    },
    {
      "type": "reasoning",
      "question": "Why does each service have its own database in this architecture?",
      "answer": "Each service having its own database enables data isolation, independent scaling, and prevents tight coupling between services. This follows the database-per-service pattern in microservices design."
    }
  ]
}
```

#### 3.8 生成三种训练数据

**Stage 1: Compression Pretraining** (`pretrain_data.jsonl`)
```json
{
  "data_type": "qa",
  "question": ["Summarize the following text: # System Architecture..."],
  "answers": ["The system implements a microservices architecture with API Gateway..."],
  "docs": ["# System Architecture\n\nThe system follows...\n\n[IMAGE DESCRIPTION]...\n\n## Core Components..."]
}
```
- **用途**: 训练文档压缩器（单文档 -> 压缩表示）
- **特点**:
  - `docs` 字段**始终包含1个文档**（即使使用top-k合成）
  - question是通用的摘要提示
  - answer是dense_summary

**Stage 2: Instruction Tuning** (`instruction_data.jsonl`)
```json
{
  "question": "What are the main components in the microservices architecture?",
  "docs": [
    "# System Architecture\n\n[IMAGE DESCRIPTION]...\n\nCore components include...",
    "负面样本文档1（语义相似但不含答案）",
    "负面样本文档2",
    "负面样本文档3",
    "负面样本文档4"
  ],
  "gold_answer": "The main components are API Gateway, Service Registry, Authentication Service, User Service, and Order Service..."
}
```
- **用途**: 训练从多文档中检索和生成答案
- **特点**:
  - `docs` 字段**包含top-k个文档**（如top_k=5则有5个）
  - 第一个是正样本（包含答案）
  - 其余是负样本（困难负样本或随机负样本）
  - question是具体的用户问题
  - gold_answer是参考答案

**Stage 3: End-to-End** (`end_to_end_data.jsonl`)
```json
{
  "question": "Why does each service have its own database in this architecture?",
  "docs": [
    "# System Architecture...",
    "负面样本1",
    "负面样本2",
    "负面样本3",
    "负面样本4"
  ],
  "gold_answer": "Each service having its own database enables data isolation, independent scaling..."
}
```
- **用途**: 联合训练检索器和生成器
- **特点**: 与Stage 2格式完全相同，但训练目标不同

#### 3.9 负样本选择（仅Stage 2/3）

**如果 top_k = 1**（基础模式）:
```python
candidate_docs = [chunk]  # 只有正样本
```

**如果 top_k > 1**（多文档模式）:

1. **随机负样本**（默认）:
```python
def select_negative_documents(positive_idx, all_chunks, target_top_k):
    num_negatives = target_top_k - 1  # 需要4个负样本（如果top_k=5）

    # 排除正样本，随机采样
    available_indices = [i for i in range(len(all_chunks)) if i != positive_idx]
    negative_indices = random.sample(available_indices, num_negatives)

    return [all_chunks[i] for i in negative_indices]

negative_docs = select_negative_documents(current_idx, chunks, args.target_top_k)
candidate_docs = [positive_chunk] + negative_docs
random.shuffle(candidate_docs)  # 打乱顺序，避免位置偏差
```

2. **困难负样本**（使用 `--use_embeddings`）:
```python
def select_negative_documents_hard(positive_chunk, positive_idx, all_chunks,
                                   chunk_embeddings, positive_embedding, target_top_k):
    num_negatives = target_top_k - 1

    # 计算所有文档与正样本的相似度
    similarities = []
    for i, (chunk, emb) in enumerate(zip(all_chunks, chunk_embeddings)):
        if i != positive_idx and emb:
            sim = cosine_similarity(positive_embedding, emb)
            similarities.append((i, sim))

    # 按相似度降序排序，取top-N最相似的作为困难负样本
    similarities.sort(key=lambda x: x[1], reverse=True)
    negative_indices = [idx for idx, _ in similarities[:num_negatives]]

    return [all_chunks[i] for i in negative_indices]
```

**Embeddings生成**（如果使用困难负样本）:
```python
# 为所有chunks生成embeddings
chunk_embeddings = []
for chunk in tqdm(chunks, desc="Embeddings"):
    response = client.embeddings.create(
        model="text-embedding-v3",  # OpenAI embedding model
        input=chunk[:8000]  # 截断到8000字符
    )
    emb = response.data[0].embedding
    chunk_embeddings.append(emb)
    time.sleep(0.1)  # 速率限制
```

### 关键参数

#### 基础合成（top-k=1）
```bash
python scripts/synthesize_data.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --model qwen-turbo \
    --chunk_size 1000
```

#### 高级合成（top-k=5，随机负样本）
```bash
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --model qwen-turbo \
    --target_top_k 5 \
    --chunk_size 1000
```

#### 高级合成（top-k=5，困难负样本）
```bash
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --target_top_k 5 \
    --use_embeddings  # 启用困难负样本挖掘
```

### 性能优化

1. **进度条**: 使用 `tqdm` 显示处理进度
2. **速率限制**: 每次LLM调用后 `time.sleep(0.5)`
3. **错误容忍**: 单个chunk失败不影响后续处理
4. **断点续传**: 图片描述生成支持从中断处恢复

---

## 输出数据格式

### Stage 1: pretrain_data.jsonl

**字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `data_type` | string | 固定值 "qa" |
| `question` | list[string] | 摘要提示（长度=1） |
| `answers` | list[string] | dense_summary（长度=1） |
| `docs` | list[string] | 原始chunk（长度=1，**始终为1**） |

**示例**:
```json
{
  "data_type": "qa",
  "question": ["Summarize the following text: # System Architecture..."],
  "answers": ["The system implements a microservices architecture..."],
  "docs": ["# System Architecture\n\n[IMAGE DESCRIPTION]...\n\nCore components..."]
}
```

**训练目标**:
- 输入: `docs[0]` (原始文档)
- 输出: `answers[0]` (dense summary)
- 学习: 文档压缩到memory tokens

### Stage 2: instruction_data.jsonl

**字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 用户问题 |
| `docs` | list[string] | 候选文档（长度=top_k） |
| `gold_answer` | string | 参考答案 |

**示例（top_k=5）**:
```json
{
  "question": "What are the main components in the microservices architecture?",
  "docs": [
    "# System Architecture\n\n[IMAGE DESCRIPTION]...\n\nCore components include...",
    "负面样本1：关于数据库设计的文档...",
    "负面样本2：关于用户认证的文档...",
    "负面样本3：关于部署流程的文档...",
    "负面样本4：关于监控系统的文档..."
  ],
  "gold_answer": "The main components are API Gateway, Service Registry, Authentication Service, User Service, and Order Service..."
}
```

**训练目标**:
- 输入: `question` + `docs` (多个候选文档)
- 输出: `gold_answer`
- 学习: 从多文档中检索+生成答案

### Stage 3: end_to_end_data.jsonl

**格式**: 与Stage 2完全相同

**训练目标**: 联合训练检索器和生成器

---

## 完整示例

### 输入文件

**raw_data/technical_architecture.docx** (部分内容):
```
# System Architecture

## Overview

The system follows a microservices architecture pattern for scalability and maintainability.

[此处有一张架构图]

## Core Components

### API Gateway
The API Gateway serves as the single entry point...

### Service Registry
The Service Registry maintains...

[此处有一张组件交互图]
```

**raw_data/Workflows/order_processing_flow.jpg**:
[一张订单处理流程图]

### 阶段1输出: raw_knowledge.jsonl（提取后）

```json
{
  "file_path": "raw_data/technical_architecture.docx",
  "filename": "technical_architecture.docx",
  "content": "# System Architecture\n\n## Overview\n\nThe system follows a microservices architecture...\n\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_0.png]\n\n## Core Components\n\n### API Gateway\nThe API Gateway serves...\n\n### Service Registry\nThe Service Registry maintains...\n\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_1.png]\n\n--- Extracted Images ---\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_0.png]\n[IMAGE_REF: example/extracted_assets/technical_architecture_img_1.png]",
  "extracted_images": ["example/extracted_assets/technical_architecture_img_0.png", "example/extracted_assets/technical_architecture_img_1.png"]
}
```

### 阶段2输出: raw_knowledge.jsonl（图片描述追加）

```json
{
  "file_path": "example/extracted_assets/technical_architecture_img_0.png",
  "filename": "technical_architecture_img_0.png",
  "content": "[IMAGE DESCRIPTION of technical_architecture_img_0.png]\n1. **Title/Topic**: Microservices Architecture Overview\n\n2. **Text Content**:\n- API Gateway\n- Service Registry\n- Authentication Service\n- User Service\n- Order Service\n- Database Layer\n\n3. **Visual Elements**:\nThe diagram shows a microservices architecture with API Gateway as the entry point, Service Registry for discovery, and multiple independent services.\n\n4. **Key Information**:\nEach service has its own database, enabling independent scaling and deployment. Services communicate via REST APIs.",
  "source_type": "image"
}
```

```json
{
  "file_path": "raw_data/Workflows/order_processing_flow.jpg",
  "filename": "order_processing_flow.jpg",
  "content": "[IMAGE DESCRIPTION of order_processing_flow.jpg]\n1. **Title/Topic**: Order Processing Workflow\n\n2. **Text Content**:\n- Start\n- Receive Order\n- Validate Inventory\n- Process Payment\n- Ship Order\n- Send Confirmation\n- End\n\n3. **Visual Elements**:\nA flowchart showing the complete order processing pipeline with decision points.\n\n4. **Key Information**:\nIncludes validation checkpoints for inventory and payment, with error handling paths.",
  "source_type": "image"
}
```

### 阶段3输出: 训练数据

#### pretrain_data.jsonl（部分）

```json
{
  "data_type": "qa",
  "question": ["Summarize the following text: ## Core Components\n\n### API Gateway..."],
  "answers": ["The architecture consists of an API Gateway entry point, Service Registry for discovery, and domain services (Authentication, User, Order) each with dedicated databases. This design enables independent scaling and fault isolation."],
  "docs": ["## Core Components\n\n### API Gateway\nThe API Gateway serves as the single entry point...\n\n[IMAGE DESCRIPTION of technical_architecture_img_0.png]\n1. **Title/Topic**: Microservices Architecture Overview\n...\n\n### Service Registry\nThe Service Registry maintains..."]
}
```

#### instruction_data.jsonl（部分，top_k=5）

```json
{
  "question": "What role does the API Gateway play in the microservices architecture?",
  "docs": [
    "## Core Components\n\n### API Gateway\nThe API Gateway serves as the single entry point...\n\n[IMAGE DESCRIPTION]...",
    "## Data Layer\n\nThe database architecture uses...",
    "[IMAGE DESCRIPTION of order_processing_flow.jpg]\n...",
    "## Deployment Strategy\n\nServices are deployed using...",
    "## Monitoring and Logging\n\nThe system uses..."
  ],
  "gold_answer": "The API Gateway serves as the single entry point for all client requests, handling routing, authentication, load balancing, and API composition. It shields clients from the complexity of the underlying microservices architecture."
}
```

```json
{
  "question": "订单处理的完整流程是什么？",
  "docs": [
    "[IMAGE DESCRIPTION of order_processing_flow.jpg]\n1. **Title/Topic**: Order Processing Workflow\n...",
    "## Payment Integration\n\nThe payment system...",
    "## Core Components\n\n### API Gateway...",
    "## User Management\n\nUser profiles are stored...",
    "## Shipping and Fulfillment\n\nOnce orders are..."
  ],
  "gold_answer": "The complete order processing workflow includes: 1) Receive order from customer, 2) Validate inventory availability, 3) Process payment through payment gateway, 4) Trigger shipping process, 5) Send confirmation to customer. Each step includes validation checkpoints and error handling."
}
```

#### end_to_end_data.jsonl

格式与 `instruction_data.jsonl` 完全相同，只是训练阶段和目标不同。

---

## 数据统计示例

假设输入：
- 10个文档（5个PDF，2个DOCX，1个PPTX，2个TXT）
- 8个独立图片文件
- 从文档中提取120张嵌入图片

处理结果：
```
阶段1: extract_with_docling.py
├── 10个文档条目写入 raw_knowledge.jsonl
└── 120张图片保存到 extracted_assets/

阶段2: extract_images.py
├── Pass 1: 8个独立图片 → 8个描述条目追加到 raw_knowledge.jsonl
└── Pass 2: 120张嵌入图片 → 120个描述条目追加到 raw_knowledge.jsonl

raw_knowledge.jsonl 总计:
├── 10个文档条目（包含[IMAGE_REF]标记）
├── 120个嵌入图片描述条目
└── 8个独立图片描述条目
    总计: 138个条目

阶段3: synthesize_data_topk.py (chunk_size=1000, top_k=5)
├── 读取: 10个文档 + 8个独立图片 = 18个待处理文档
├── 加载: 128个图片描述（120个嵌入 + 8个独立）
├── 分块: 18个文档 → 约180个chunks（平均每文档10个）
├── 图片融合: 120个[IMAGE_REF]被替换为实际描述
└── LLM生成: 180个chunks × 4个QA/chunk = 720个QA对

输出文件:
├── pretrain_data.jsonl: 180条（每chunk一条，docs长度=1）
├── instruction_data.jsonl: 720条（每QA一条，docs长度=5）
└── end_to_end_data.jsonl: 720条（与instruction相同）
```

---

## 执行命令总结

### 完整管道（自动化）

```bash
# 设置环境变量
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VISION_MODEL="qwen-vl-max"
export MODEL="qwen-turbo"

# 运行完整管道
bash scripts/run_data_pipeline.sh
```

### 逐步执行（手动）

```bash
# Step 1: 文档提取
python scripts/extract_with_docling.py \
    --input_dir raw_data \
    --output_file example/raw_knowledge.jsonl \
    --image_output_dir example/extracted_assets

# Step 2.1: 独立图片描述
python scripts/extract_images.py \
    --input_dir raw_data \
    --output_file example/raw_knowledge.jsonl \
    --model qwen-vl-max

# Step 2.2: 嵌入图片描述
python scripts/extract_images.py \
    --input_dir example/extracted_assets \
    --output_file example/raw_knowledge.jsonl \
    --model qwen-vl-max

# Step 3: QA数据合成（top-k=5）
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --model qwen-turbo \
    --target_top_k 5
```

### 高级选项

```bash
# 使用困难负样本（需要OpenAI embedding API）
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --target_top_k 5 \
    --use_embeddings

# 自定义chunk大小
python scripts/synthesize_data_topk.py \
    --input_file example/raw_knowledge.jsonl \
    --output_dir example \
    --api_key $OPENAI_API_KEY \
    --model qwen-turbo \
    --chunk_size 1500  # 默认1000
```

---

## 常见问题

### Q1: 为什么Stage 1的docs只有1个文档？

**A**: Stage 1训练文档压缩器，学习单文档→压缩表示的映射。即使使用top-k合成，Stage 1数据也只包含1个文档，因为压缩是针对单文档的操作。

### Q2: 图片描述会增加多少tokens？

**A**: 一个中等复杂度的图片描述约200-500 tokens。对于120张图片的文档，总增加约24k-60k tokens，分散到180个chunks中，每个chunk平均增加约130-330 tokens。

### Q3: 如何处理API限流？

**A**:
- 每次LLM调用后有 `time.sleep(0.5)` 延迟
- 使用tqdm进度条监控速度
- 如需更快处理，可并行运行多个进程（分割输入文件）

### Q4: 困难负样本vs随机负样本的效果？

**A**:
- **困难负样本**: 训练质量更高，模型学会区分相似但不同的内容，但需要embedding API成本
- **随机负样本**: 成本低，适合初步训练，但区分度较低

建议：初期使用随机负样本快速迭代，后期使用困难负样本精调。

### Q5: 如何验证数据质量？

```bash
# 检查数据格式和top-k一致性
python scripts/validate_topk_data.py \
    --input_file example/instruction_data.jsonl \
    --expected_top_k 5

# 抽样查看（前5条）
head -5 example/instruction_data.jsonl | jq .

# 统计问题类型分布
jq -r '.question' example/instruction_data.jsonl | \
    grep -c "[\u4e00-\u9fa5]"  # 中文问题数量
```

### Q6: 原始图片文件和嵌入图片的区别？

| 维度 | 嵌入图片 | 原始图片文件 |
|------|----------|--------------|
| 来源 | 从PDF/DOCX提取 | 用户直接放置在raw_data/ |
| 路径 | `extracted_assets/` | `raw_data/` |
| 处理 | 合并到源文档（通过IMAGE_REF） | 独立训练样本 |
| 训练数据 | 增强文档的一部分 | 独立的QA对 |

### Q7: 如何处理超大文档？

```bash
# 增大chunk_size以减少chunk数量
python scripts/synthesize_data_topk.py \
    --chunk_size 2000  # 默认1000

# 或者在提取阶段手动分割大文档
# 例如: 将500页PDF分割为5个100页文件
```

---

## 性能指标参考

基于典型数据（10文档 + 8独立图片 + 120嵌入图片）:

| 阶段 | 耗时 | API调用 | 成本估算（DashScope） |
|------|------|---------|---------------------|
| extract_with_docling | 2-5分钟 | 0 | ¥0 |
| extract_images (128张) | 10-20分钟 | 128 (vision) | ¥3.2 (qwen-vl-max ¥0.025/图) |
| synthesize_data (180 chunks) | 15-30分钟 | 180 (text) | ¥0.18 (qwen-turbo ¥0.001/1K tokens) |
| **总计** | **30-60分钟** | **308次调用** | **约¥3.5** |

使用OpenAI API (gpt-4o-mini + gpt-4o-vision):
- 图片描述: 128张 × $0.002 = $0.26
- 文本合成: 180 chunks × $0.0015 = $0.27
- **总计**: 约$0.53 (约¥3.8)

---

## 参考资料

- [CLaRa Paper](https://arxiv.org/abs/2511.18659)
- [IBM Docling](https://github.com/DS4SD/docling)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Qwen-VL](https://help.aliyun.com/zh/model-studio/getting-started/models)

---

**最后更新**: 2025-12-04
**版本**: 1.0
