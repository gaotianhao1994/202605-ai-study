
# -*- coding: utf-8 -*-
"""
Chroma向量数据库入门示例
"""

# 1. 安装必要的包（如果还没安装）
# pip install langchain langchain-chroma langchain-openai python-dotenv

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

print("=== Chroma向量数据库入门示例 ===\n")

# ----------------
# 示例1：基础用法
# ----------------
print("示例1：基础用法 - 创建集合、添加文档、查询")

# 1. 初始化Embedding模型（把文本转换成向量）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 创建Chroma向量存储
vector_store = Chroma(
    collection_name="my_first_collection",  # 集合名称
    embedding_function=embeddings,          # 向量化函数
    persist_directory="./chroma_db"        # 数据保存位置（本地文件夹）
)

# 3. 准备一些示例文档
documents = [
    Document(
        page_content="Python是一种高级编程语言，以其简洁的语法和强大的功能著称。",
        metadata={"source": "编程入门", "topic": "Python"}
    ),
    Document(
        page_content="JavaScript主要用于网页开发，可以在浏览器中运行。",
        metadata={"source": "编程入门", "topic": "JavaScript"}
    ),
    Document(
        page_content="机器学习是人工智能的一个分支，让计算机从数据中学习。",
        metadata={"source": "AI基础", "topic": "机器学习"}
    ),
    Document(
        page_content="深度学习是机器学习的一个子领域，使用神经网络。",
        metadata={"source": "AI基础", "topic": "深度学习"}
    ),
    Document(
        page_content="向量数据库用于存储和检索高维向量数据。",
        metadata={"source": "数据库", "topic": "向量数据库"}
    )
]

# 4. 添加文档到向量存储
print("\n正在添加文档...")
vector_store.add_documents(documents)
print(f"已添加 {len(documents)} 个文档")

# 5. 查询
query = "什么是深度学习？"
print(f"\n查询问题：{query}")

# 搜索最相似的2个文档
results = vector_store.similarity_search(query, k=2)

print("\n搜索结果：")
for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容：{doc.page_content}")
    print(f"元数据：{doc.metadata}")

# ----------------
# 示例2：相似度得分
# ----------------
print("\n" + "="*50)
print("示例2：带相似度得分的搜索")

results_with_score = vector_store.similarity_search_with_score(query, k=3)

print("\n搜索结果（带得分）：")
for i, (doc, score) in enumerate(results_with_score, 1):
    print(f"\n结果 {i}:")
    print(f"内容：{doc.page_content}")
    print(f"相似度得分：{score:.4f}（越小越相似）")
    print(f"元数据：{doc.metadata}")

# ----------------
# 示例3：持久化和加载
# ----------------
print("\n" + "="*50)
print("示例3：持久化和重新加载")

# 数据已经保存在 ./chroma_db 文件夹中了
# 现在我们创建一个新的Chroma实例，加载已保存的数据

print("\n正在加载已保存的向量数据库...")
new_vector_store = Chroma(
    collection_name="my_first_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 验证数据是否还在
query2 = "Python用于什么？"
results2 = new_vector_store.similarity_search(query2, k=1)

print(f"\n查询问题：{query2}")
print(f"结果：{results2[0].page_content}")

# ----------------
# 示例4：删除集合
# ----------------
print("\n" + "="*50)
print("示例4：管理集合（可选）")

# 获取所有集合
collections = new_vector_store._client.list_collections()
print(f"\n当前集合：{[c.name for c in collections]}")

# 如果你想删除集合，可以这样做：
# new_vector_store.delete_collection()

print("\n=== 示例运行完成！===")
