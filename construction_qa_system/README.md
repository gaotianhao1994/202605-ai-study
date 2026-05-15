
# 建筑行业知识问答系统

基于深度学习的建筑领域智能问答系统，支持建筑材料、施工工艺、安全规范等知识的智能检索与问答。

## 项目结构

```
construction_qa_system/
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖包
├── utils/
│   └── data_preprocessor.py   # 数据预处理模块
├── models/
│   ├── vector_db.py          # 向量数据库模块
│   └── qa_engine.py          # 问答引擎模块
├── data/                     # 知识文档目录
├── docs/                     # 文档目录
└── README.md
```

## 核心模块说明

### 1. 数据预处理模块 (data_preprocessor.py)

- **TextCleaner**: 文本清洗，去除多余空格和特殊字符
- **Chunker**: 文本分块，将长文档切分为合适大小的片段
- **ConstructionKnowledgeExtractor**: 建筑领域实体提取和知识分类
- **KnowledgeBaseBuilder**: 知识库构建器，从目录批量处理文档

### 2. 向量数据库模块 (vector_db.py)

- **SimpleEmbeddingModel**: 简化的嵌入模型（生产环境可替换为Sentence-BERT等）
- **VectorDatabase**: 向量数据库，支持余弦相似度搜索

### 3. 问答引擎模块 (qa_engine.py)

- **QuestionAnalyzer**: 问题分类与关键词提取
- **AnswerGenerator**: 基于模板的答案生成
- **ConstructionQASystem**: 问答系统主类

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python main.py
```

### 示例问题

- 什么是混凝土？
- 混凝土浇筑的步骤是什么？
- 如何防止混凝土开裂？
- 施工现场有哪些安全注意事项？
- 钢筋有哪些强度等级？

## 使用示例代码

### 基础使用

```python
from models.vector_db import VectorDatabase
from models.qa_engine import ConstructionQASystem

# 1. 准备知识
knowledge = [
    {
        'id': 'doc_001',
        'content': '混凝土由水泥、砂、石子和水组成...',
        'metadata': {'source': '手册.txt'}
    }
]

# 2. 构建向量库
vector_db = VectorDatabase()
vector_db.add_items(knowledge)

# 3. 初始化问答系统
qa_system = ConstructionQASystem(vector_db)

# 4. 提问
result = qa_system.ask('什么是混凝土？')
print(result['answer'])
```

### 从文档构建知识库

```python
from utils.data_preprocessor import KnowledgeBaseBuilder

builder = KnowledgeBaseBuilder()
chunks = builder.build_from_directory('./data')

vector_db = VectorDatabase()
vector_db.add_items(chunks)
```

## 扩展建议

### 替换更好的嵌入模型

```python
# 使用Sentence-BERT
from sentence_transformers import SentenceTransformer

class SBERTEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def encode(self, text):
        return self.model.encode(text)

vector_db = VectorDatabase(embedding_model=SBERTEmbedding())
```

### 接入真实LLM生成答案

```python
# 可集成OpenAI、Claude等API
from openai import OpenAI

class LLMAnswerGenerator:
    def __init__(self):
        self.client = OpenAI()
    
    def generate(self, question, contexts):
        prompt = f"基于以下信息回答问题：\n{contexts}\n问题：{question}"
        response = self.client.chat.completions.create(...)
        return response.choices[0].message.content
```

## 技术栈

- Python 3.8+
- NumPy（基础计算）
- 可选：Sentence-Transformers、FAISS、ChromaDB

## 许可证

MIT License

