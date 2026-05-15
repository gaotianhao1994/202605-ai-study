# RAG智能问答系统 - LangChain教程

基于阿里云百炼大模型的 RAG（检索增强生成）智能问答系统教程，完整实现RAG技术栈全流程。

## 项目简介

本项目通过5个渐进式章节，从零构建一个完整的RAG智能问答系统：

| 章节 | 主题 | 核心内容 |
|------|------|---------|
| 第1章 | 文档加载与预处理 | TextLoader、DirectoryLoader、Document对象、元数据管理 |
| 第2章 | 文本分割策略 | RecursiveCharacterTextSplitter、chunk_size/overlap参数调优 |
| 第3章 | 向量嵌入与FAISS存储 | Embeddings、FAISS向量库创建/保存/加载 |
| 第4章 | 检索器与相似度搜索 | Retriever、相似度搜索 vs MMR、参数调优 |
| 第5章 | 问答链构建与优化 | LCEL RAG链、自定义Prompt、交互式问答 |

## 快速开始

### 环境要求

- Python >= 3.8
- 阿里云百炼 API Key

### 安装步骤

1. **安装依赖**

```bash
cd src/demos/20260514-rag-qa-system
pip install -r requirements.txt
```

2. **配置环境变量**

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的阿里云百炼 API 配置：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen3.5-122b-a10b
EMBEDDING_MODEL_NAME=text-embedding-v3
```

### 运行方式

#### 方式1：交互式模式

```bash
python main.py
```

#### 方式2：运行指定章节

```bash
python main.py --chapter 1  # 运行第一章
python main.py -c 3         # 简写形式
```

#### 方式3：运行所有章节

```bash
python main.py --all
```

#### 方式4：直接运行章节文件

```bash
python chapter1_document_loader.py  # 第1章
python chapter2_text_splitter.py    # 第2章
python chapter3_vector_store.py     # 第3章
python chapter4_retriever.py        # 第4章
python chapter5_qa_chain.py         # 第5章
```

## 项目结构

```
20260514-rag-qa-system/
├── README.md                        # 项目说明文档
├── tutorial.md                      # 教学说明文档
├── .env.example                     # 环境变量模板
├── requirements.txt                 # 项目依赖
├── config.py                        # 配置管理模块
├── logger.py                        # 日志配置模块
├── main.py                          # 主程序入口
├── chapter1_document_loader.py      # 第1章：文档加载与预处理
├── chapter2_text_splitter.py        # 第2章：文本分割策略
├── chapter3_vector_store.py         # 第3章：向量嵌入与FAISS存储
├── chapter4_retriever.py            # 第4章：检索器与相似度搜索
├── chapter5_qa_chain.py             # 第5章：问答链构建与优化
└── data/                            # 示例数据目录
    ├── ai_basics.txt                # AI基础知识文档
    ├── python_learning.txt          # Python学习指南
    └── langchain_intro.txt          # LangChain入门介绍
```

## 学习路径

### 第1章：文档加载与预处理（15-20分钟）

**学习目标**：理解RAG中文档加载的作用

**核心内容**：
- RAG技术概述：为什么需要检索增强？
- TextLoader：加载纯文本文件
- DirectoryLoader：批量加载目录下所有文档
- Document对象结构：page_content + metadata
- 元数据管理：为文档添加来源、标题等元信息

### 第2章：文本分割策略（15-20分钟）

**学习目标**：理解为什么需要文本分割，掌握分割参数调优

**核心内容**：
- 为什么需要分割？LLM上下文窗口限制
- RecursiveCharacterTextSplitter：递归字符分割
- chunk_size和chunk_overlap参数对比实验
- 中文文本分割的特殊考虑

### 第3章：向量嵌入与FAISS存储（20-25分钟）

**学习目标**：理解向量嵌入的原理，掌握FAISS使用

**核心内容**：
- 什么是向量嵌入？文本到高维向量的语义映射
- OpenAIEmbeddings：通过阿里云百炼API生成嵌入
- FAISS向量库：创建、添加文档、保存/加载
- 相似度度量：L2距离 vs 余弦相似度

### 第4章：检索器与相似度搜索（20-25分钟）

**学习目标**：理解检索器的作用，掌握不同检索策略

**核心内容**：
- vectorstore.as_retriever()
- 相似度搜索 vs MMR检索
- 参数调优：k、fetch_k、lambda_mult
- 检索质量评估

### 第5章：问答链构建与优化（25-30分钟）

**学习目标**：掌握RAG问答链的构建，实现端到端问答系统

**核心内容**：
- LCEL RAG链：检索器与LLM连接
- 自定义Prompt模板
- 带来源引用的问答
- 端到端RAG流程演示
- 交互式问答系统

## RAG技术流程

```
用户问题
   ↓
[检索器] ← [向量数据库] ← [文本分割] ← [文档加载]
   ↓
相关文档片段
   ↓
[LLM生成] ← 问题 + 相关文档
   ↓
回答
```

## 常见问题

### Q: 提示".env 文件不存在"

复制 `.env.example` 为 `.env` 并填写配置：
```bash
cp .env.example .env
```

### Q: 提示"ModuleNotFoundError: No module named 'langchain'"

安装项目依赖：
```bash
pip install -r requirements.txt
```

### Q: 提示"ModuleNotFoundError: No module named 'faiss'"

安装FAISS：
```bash
pip install faiss-cpu
```

### Q: API调用失败

可能的原因：
1. API Key 无效或过期
2. 网络连接问题
3. API 服务不可用

请检查 `.env` 文件中的配置是否正确。

### Q: 如何获取阿里云百炼 API Key？

1. 访问阿里云百炼控制台
2. 创建应用并获取 API Key
3. 将 API Key 填入 `.env` 文件

## 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [FAISS 文档](https://github.com/facebookresearch/faiss)
- [RAG 论文](https://arxiv.org/abs/2005.11401)

## 学习建议

1. **按顺序学习**：章节之间有递进关系，建议按顺序学习
2. **动手实践**：每个章节都有可运行的代码，务必动手运行
3. **修改参数**：尝试修改代码中的参数，观察输出变化
4. **阅读注释**：代码注释包含详细的原理解释，值得仔细阅读
5. **阅读tutorial.md**：教学说明文档包含更深入的技术原理解释

---

**项目名称**: RAG智能问答系统 - LangChain教程
**创建时间**: 2026年5月
**版本**: v1.0
