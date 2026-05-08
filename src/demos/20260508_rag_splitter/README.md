# RAG文档分割演示项目

## 项目简介

本项目是一个教学性质的RAG（检索增强生成）技术演示项目，重点展示文档分割功能。项目包含三种常用的文档分割策略实现，帮助初学者理解RAG技术中的文档预处理环节。

## 项目结构

```
20260508-demo-rag-splitter/
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖配置文件
├── demo.py               # 主演示脚本
├── data/                 # 示例文档目录
│   └── sample_document.txt
└── src/                  # 源代码目录
    ├── __init__.py
    ├── splitters/        # 分割器实现
    │   ├── __init__.py
    │   ├── character_splitter.py  # 字符分割器
    │   ├── token_splitter.py      # Token分割器
    │   └── semantic_splitter.py   # 语义分割器
    └── utils/            # 工具函数
        ├── __init__.py
        └── text_utils.py
```

## 分割策略对比

| 分割策略 | 实现方式 | 优点 | 缺点 | 适用场景 |
|---------|---------|------|------|---------|
| **字符分割** | 按固定字符数分割 | 简单、快速、无需依赖 | 可能切断语义 | 简单文本、短文档 |
| **Token分割** | 按Token数分割 | 直接考虑模型上下文限制 | 依赖Token编码库 | 与大语言模型配合 |
| **语义分割** | 基于语义相似度分割 | 保持语义完整性 | 计算成本高 | 复杂文档、需要上下文 |

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python demo.py
```

### 3. 单独运行各个分割器

```bash
# 字符分割器
python -m src.splitters.character_splitter

# Token分割器
python -m src.splitters.token_splitter

# 语义分割器
python -m src.splitters.semantic_splitter
```

## 使用示例

### 基础用法

```python
from src import CharacterSplitter, TokenSplitter, SemanticSplitter

# 读取文档
with open("data/sample_document.txt", "r", encoding="utf-8") as f:
    document = f.read()

# 字符分割器
char_splitter = CharacterSplitter(chunk_size=500, chunk_overlap=50)
char_chunks = char_splitter.split(document)

# Token分割器
token_splitter = TokenSplitter(chunk_size=512, chunk_overlap=50)
token_chunks = token_splitter.split(document)

# 语义分割器
semantic_splitter = SemanticSplitter(similarity_threshold=0.5)
semantic_chunks = semantic_splitter.split(document)
```

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| chunk_size | int | 每个chunk的最大大小（字符或Token） | 500/512 |
| chunk_overlap | int | 相邻chunk的重叠大小 | 50 |
| similarity_threshold | float | 语义相似度阈值（0-1） | 0.5 |
| model_name | str | 使用的模型名称 | "gpt-3.5-turbo" |

## 技术原理

### 字符分割器

将文档按照固定字符数进行分割，遇到换行符或空格时会尽量在边界处分割，以保持句子完整性。

### Token分割器

使用OpenAI的tiktoken库将文本转换为Token，确保每个chunk的Token数不超过模型的上下文窗口限制。

### 语义分割器

使用预训练的句子嵌入模型（如all-MiniLM-L6-v2）计算句子之间的语义相似度，在相似度较低的地方进行分割，从而保持chunk的语义连贯性。

## 选择建议

1. **快速原型开发**：使用字符分割器，简单快速
2. **生产环境RAG系统**：使用Token分割器，符合模型限制
3. **复杂文档处理**：使用语义分割器，保持语义完整

## 扩展开发

### 添加新的分割策略

1. 在 `src/splitters/` 目录下创建新的分割器文件
2. 实现分割器类，继承统一接口
3. 在 `src/splitters/__init__.py` 中导出新类
4. 在 `demo.py` 中添加演示代码

### 修改示例文档

替换 `data/sample_document.txt` 文件中的内容，测试不同文档的分割效果。

## 依赖列表

| 依赖 | 版本 | 用途 |
|------|------|------|
| python | >=3.8 | 基础环境 |
| langchain | 0.1.10 | RAG框架 |
| tiktoken | 0.6.0 | Token计算 |
| sentence-transformers | 2.7.0 | 语义相似度 |
| transformers | 4.38.2 | 预训练模型 |
| jieba | 0.42.1 | 中文分词 |

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

---

**项目名称**: RAG文档分割演示项目  
**创建时间**: 2026年5月  
**版本**: v1.0