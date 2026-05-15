# RAG智能问答系统演示项目 - 设计文档

## 概述

基于LangChain实现完整的RAG（检索增强生成）智能问答系统演示项目，采用5章渐进式教程结构，覆盖RAG技术栈全流程。

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| LLM | 阿里云百炼 qwen3.5-122b-a10b | 与项目现有配置一致 |
| Embedding | text-embedding-v3 | 阿里云百炼提供的嵌入模型 |
| 向量存储 | FAISS | 本地运行、无需额外服务、适合教学 |
| 文本分割 | RecursiveCharacterTextSplitter | LangChain最常用的分割器 |
| 问答链 | RetrievalQA | LangChain标准RAG问答链 |

## 项目目录结构

```
src/demos/20260514-rag-qa-system/
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

## 章节设计

### 第1章：文档加载与预处理

核心内容：
- RAG技术概述：LLM局限性、为什么需要检索增强
- TextLoader：加载纯文本文件
- DirectoryLoader：批量加载目录下所有文档
- Document对象结构：page_content + metadata
- 元数据管理：为文档添加来源、标题等元信息

### 第2章：文本分割策略

核心内容：
- 为什么需要分割：LLM上下文窗口限制
- RecursiveCharacterTextSplitter：递归字符分割
- 关键参数：chunk_size、chunk_overlap、separators
- 分割效果对比：不同参数对检索质量的影响
- 中文文本分割的特殊考虑

### 第3章：向量嵌入与FAISS存储

核心内容：
- 向量嵌入原理：文本→高维向量的语义映射
- OpenAIEmbeddings：通过阿里云百炼API生成嵌入
- FAISS向量库：创建、添加文档、保存/加载
- 相似度度量：L2距离 vs 余弦相似度
- 向量存储持久化

### 第4章：检索器与相似度搜索

核心内容：
- vectorstore.as_retriever()
- 检索策略：相似度搜索 vs MMR
- 参数调优：k、fetch_k、lambda_mult
- similarity_search_with_score()
- 检索质量评估

### 第5章：问答链构建与优化

核心内容：
- RetrievalQA链：检索器与LLM连接
- 自定义Prompt模板
- RetrievalQAWithSourcesChain：带来源引用
- 端到端RAG流程：问题→检索→增强→生成
- 交互式问答演示

## 依赖

```
langchain>=1.2.17
langchain-openai>=0.1.0
langchain-community>=0.1.0
langchain-text-splitters>=0.1.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
tiktoken>=0.6.0
```

## 示例数据

3个内置文本文件，每个约500-1000字：
- ai_basics.txt：AI基础知识
- python_learning.txt：Python学习指南
- langchain_intro.txt：LangChain框架介绍

## 主程序入口

main.py 提供三种运行方式：
- 交互式菜单选择章节
- --chapter N 运行指定章节
- --all 运行所有章节
