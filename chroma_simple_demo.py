
# -*- coding: utf-8 -*-
"""
Chroma向量数据库 - 无API Key版本
使用简单的本地向量化演示
"""

import chromadb
from chromadb.utils import embedding_functions

print("=== Chroma向量数据库 - 简化示例 ===\n")

# ----------------
# 1. 初始化Chroma客户端
# ----------------
print("1. 初始化Chroma客户端")
client = chromadb.PersistentClient(path="./simple_chroma_db")

# ----------------
# 2. 使用默认的Embedding函数（不需要API Key）
# ----------------
print("2. 准备Embedding函数")
# 使用Chroma自带的默认embedding（基于Sentence Transformers）
default_ef = embedding_functions.DefaultEmbeddingFunction()

# ----------------
# 3. 创建集合
# ----------------
print("3. 创建集合")
collection = client.get_or_create_collection(
    name="simple_demo",
    embedding_function=default_ef
)

# ----------------
# 4. 添加文档
# ----------------
print("4. 添加示例文档")

documents = [
    "Python是一种高级编程语言，以其简洁的语法和强大的功能著称。",
    "JavaScript主要用于网页开发，可以在浏览器中运行。",
    "机器学习是人工智能的一个分支，让计算机从数据中学习。",
    "深度学习是机器学习的一个子领域，使用神经网络。",
    "向量数据库用于存储和检索高维向量数据。"
]

# 添加文档
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
)

print(f"已添加 {len(documents)} 个文档")

# ----------------
# 5. 查询测试
# ----------------
print("\n5. 查询测试")

queries = [
    "什么是深度学习？",
    "Python用来做什么？",
    "向量数据库是什么？"
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"查询：{query}")
    
    results = collection.query(
        query_texts=[query],
        n_results=2  # 返回最相似的2个结果
    )
    
    print("\n最相似的文档：")
    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        print(f"\n[{i}] {doc}")
        print(f"    距离：{dist:.4f}（越小越相似）")

print("\n" + "="*50)
print("\n=== 简化示例运行完成！===")
