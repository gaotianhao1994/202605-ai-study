
# -*- coding: utf-8 -*-
"""
Chroma + LangChain 问答系统示例
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 加载环境变量
load_dotenv()

print("=== Chroma + LangChain 问答系统示例 ===\n")

# ----------------
# 步骤1：准备一些示例文档
# ----------------
print("步骤1：准备示例文档")

# 模拟从文件加载的文档
sample_text = """
什么是人工智能？
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

什么是机器学习？
机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。

什么是深度学习？
深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的表示。

什么是神经网络？
神经网络是一种受生物大脑启发的计算模型，由相互连接的节点（神经元）组成。

什么是LangChain？
LangChain是一个用于开发由语言模型驱动的应用程序的框架。

什么是向量数据库？
向量数据库专门用于存储和检索高维向量数据，常用于AI应用中。
"""

# 创建文档对象
doc = Document(page_content=sample_text, metadata={"source": "AI基础知识"})

# ----------------
# 步骤2：切分文档
# ----------------
print("\n步骤2：切分文档")

# 使用文本分割器把长文本切分成小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,        # 每块的大小
    chunk_overlap=20,      # 块之间的重叠
    length_function=len
)

splits = text_splitter.split_documents([doc])
print(f"切分成了 {len(splits)} 个文档块")

# ----------------
# 步骤3：创建向量数据库
# ----------------
print("\n步骤3：创建向量数据库")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="qa_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_qa_db"
)

# 添加切分后的文档
vector_store.add_documents(splits)
print("文档已添加到向量数据库")

# ----------------
# 步骤4：创建问答链
# ----------------
print("\n步骤4：创建问答系统")

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 定义提示模板
prompt_template = """
你是一个AI助手，使用以下上下文来回答问题。
如果不知道答案，就说你不知道，不要编造答案。

上下文：
{context}

问题：{question}

答案：
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ----------------
# 步骤5：测试问答系统
# ----------------
print("\n步骤5：测试问答系统")

questions = [
    "什么是深度学习？",
    "LangChain是什么？",
    "机器学习和人工智能的关系是什么？"
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"问题：{question}")
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\n答案：{result['result']}")
    
    print("\n参考文档：")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n[{i}] {doc.page_content}")

print("\n" + "="*50)
print("\n=== 问答系统示例完成！===")
