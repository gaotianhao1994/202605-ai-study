#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章：向量嵌入与FAISS存储

学习目标：
1. 理解什么是向量嵌入以及为什么需要它
2. 掌握OpenAIEmbeddings的使用方法
3. 掌握FAISS向量库的创建、保存和加载
4. 理解相似度度量的原理

核心问题：为什么需要向量嵌入？
- 关键词搜索只能匹配字面相同的词 -> "汽车"搜不到"轿车"
- 向量嵌入捕获语义信息 -> 语义相似的文本在向量空间中距离更近
- 这是RAG能够"理解"文档内容的关键技术
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('chapter3_vector_store')


def demo_what_is_embedding():
    """
    解释向量嵌入的原理，展示嵌入向量的维度和含义

    知识点：向量嵌入（Vector Embedding）

    是什么？
    将文本转换为高维数值向量（如1536维浮点数数组），使得语义相似的文本
    在向量空间中距离更近。

    为什么需要向量嵌入？
    - 关键词搜索只能匹配字面相同的词 -> "汽车"搜不到"轿车"
    - 向量嵌入捕获语义信息 -> "汽车"和"轿车"的向量距离很近
    - 计算机只能处理数值 -> 必须将文本转换为数值表示才能计算相似度
    - 向量空间中的距离反映语义相似度 -> 这是语义搜索的基础

    追问：为什么向量能表示语义？
    - 嵌入模型在大量文本上训练 -> 学习到词语之间的语义关系
    - 语义相似的词在训练语料中经常出现在相似的上下文中
    - 模型将这种"上下文相似性"编码为"向量距离相近"
    - 例如："猫"和"狗"经常出现在相似的上下文 -> 向量距离近
    -      "猫"和"汽车"上下文差异大 -> 向量距离远

    追问：为什么不用关键词搜索而用向量搜索？
    - 关键词搜索: 精确匹配 -> "机器学习"搜不到"ML"
    - 向量搜索: 语义匹配 -> "机器学习"和"ML"向量距离近
    - 用户提问的方式千变万化 -> 关键词搜索无法覆盖所有同义表达
    - 向量搜索天然支持同义词、近义词 -> 更符合人类的检索习惯
    """
    print("=" * 60)
    print("1. 什么是向量嵌入")
    print("=" * 60)

    print("""
向量嵌入（Vector Embedding）的核心思想：
  将文本转换为数值向量，使得语义相似的文本在向量空间中距离更近

类比理解：
  想象一个三维空间，X轴代表"动物性"，Y轴代表"大小"，Z轴代表"速度"
  - "猫" -> (0.8, 0.2, 0.5)  小型动物
  - "狗" -> (0.9, 0.4, 0.6)  中型动物
  - "汽车" -> (0.0, 0.8, 0.9) 非动物，大，快

  在这个空间中，"猫"和"狗"距离近，"猫"和"汽车"距离远
  这就是向量嵌入的基本原理，只不过实际维度远不止3维

关键词搜索 vs 向量搜索的对比：

  查询: "深度学习框架"
  关键词搜索结果:
    - 包含"深度学习框架"的文档 -> 命中
    - 包含"神经网络工具包"的文档 -> 未命中（字面不同）
    - 包含"PyTorch和TensorFlow"的文档 -> 未命中（字面不同）

  向量搜索结果:
    - 包含"深度学习框架"的文档 -> 命中（语义相同）
    - 包含"神经网络工具包"的文档 -> 命中（语义相近）
    - 包含"PyTorch和TensorFlow"的文档 -> 命中（语义相关）

  -> 向量搜索能理解"深度学习框架"和"神经网络工具包"是同一类东西
  -> 关键词搜索只能做字面匹配，无法理解语义
""")

    print("--- 嵌入向量的维度 ---")
    print("""
  嵌入向量是一个固定长度的浮点数数组，维度由模型决定：

  常见嵌入模型及其维度：
    text-embedding-ada-002 (OpenAI)   -> 1536维
    text-embedding-3-small (OpenAI)   -> 1536维
    text-embedding-3-large (OpenAI)   -> 3072维
    text-embedding-v3 (阿里云百炼)     -> 1024维或2048维

  为什么维度这么高？
    - 低维向量无法表达复杂的语义关系 -> 信息容量不足
    - 高维向量有更大的"表达空间" -> 可以编码更丰富的语义信息
    - 每一维可以理解为一种"语义特征" -> 类似于"动物性""大小"等
    - 维度越高，能区分的语义差异越细微 -> 但计算成本也越高

  向量示例（简化为5维）：
    "人工智能" -> [0.82, -0.15, 0.43, 0.91, -0.27]
    "机器学习" -> [0.79, -0.12, 0.45, 0.88, -0.23]
    "烹饪食谱" -> [-0.31, 0.67, -0.54, 0.12, 0.89]

    "人工智能"和"机器学习"的向量很接近 -> 语义相似
    "烹饪食谱"的向量与前两者差异很大 -> 语义不同
""")

    print("--- 相似度度量方法 ---")
    print("""
  如何计算两个向量的"距离"？常用方法：

  1. 余弦相似度（Cosine Similarity）-- 最常用
     计算两个向量夹角的余弦值，范围[-1, 1]
     值越接近1 -> 越相似
     值越接近-1 -> 越不相似
     值接近0 -> 无关

     为什么余弦相似度最常用？
     - 只关注方向，不关注长度 -> 不受文本长度影响
     - 归一化后计算简单高效 -> 只需点积运算
     - 对高维稀疏向量效果好 -> 适合文本嵌入

  2. 欧氏距离（Euclidean Distance / L2）
     计算两个向量在空间中的直线距离
     距离越小 -> 越相似

     为什么RAG中较少使用欧氏距离？
     - 受向量长度影响 -> 长文本的向量模更大
     - 高维空间中距离趋于均匀 -> 区分度下降（维度灾难）
     - 但FAISS默认使用L2距离，效果也不错

  3. 内积（Inner Product）
     两个向量的点积
     值越大 -> 越相似
     当向量归一化后，内积 = 余弦相似度
""")

    print("--- 向量嵌入 要点总结 ---")
    print("1. 向量嵌入将文本转换为数值向量，捕获语义信息")
    print("2. 语义相似的文本在向量空间中距离更近")
    print("3. 嵌入维度通常为1024-3072维，维度越高表达力越强")
    print("4. 余弦相似度是最常用的相似度度量方法")
    print("5. 向量搜索 vs 关键词搜索: 语义匹配 vs 字面匹配")

    logger.info("向量嵌入原理演示完成")


def demo_create_embeddings():
    """
    使用OpenAIEmbeddings创建嵌入（通过阿里云百炼API）

    知识点：OpenAIEmbeddings

    是什么？
    LangChain提供的嵌入模型接口，将文本转换为高维向量。
    通过阿里云百炼API兼容OpenAI接口调用。

    为什么使用OpenAIEmbeddings而不是自己训练嵌入模型？
    - 训练嵌入模型需要海量语料和GPU资源 -> 成本极高
    - 预训练模型已经在大量文本上学习 -> 通用语义理解能力强
    - API调用即用即付 -> 无需维护模型服务
    - 阿里云百炼兼容OpenAI接口 -> 切换成本低

    追问：为什么通过阿里云百炼API而不是直接用OpenAI？
    - 国内访问OpenAI API需要特殊网络 -> 不稳定且合规风险
    - 阿里云百炼提供兼容接口 -> 代码几乎不用改
    - 阿里云百炼的text-embedding-v3对中文效果更好 -> 针对中文优化
    - 数据不出境 -> 满足数据合规要求
    """
    print("\n" + "=" * 60)
    print("2. 使用OpenAIEmbeddings创建嵌入")
    print("=" * 60)

    from langchain_openai import OpenAIEmbeddings
    from config import get_config

    config = get_config()

    print("\n--- 2.1 初始化嵌入模型 ---")

    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    print(f"\n嵌入模型配置:")
    print(f"  模型名称: {config.get_embedding_model_name()}")
    print(f"  API地址: {config.get_api_base()}")
    print(f"  API Key: {config.get_api_key()[:10]}...")

    print("\n--- 2.2 对单个文本创建嵌入 ---")

    single_text = "人工智能是计算机科学的一个分支"
    print(f"\n输入文本: '{single_text}'")

    try:
        single_vector = embeddings.embed_query(single_text)

        print(f"\n嵌入结果:")
        print(f"  向量类型: {type(single_vector).__name__}")
        print(f"  向量维度: {len(single_vector)}")
        print(f"  前10维: {[round(v, 4) for v in single_vector[:10]]}")
        print(f"  后5维: {[round(v, 4) for v in single_vector[-5:]]}")

        vector_min = min(single_vector)
        vector_max = max(single_vector)
        vector_mean = sum(single_vector) / len(single_vector)
        print(f"  最小值: {vector_min:.4f}")
        print(f"  最大值: {vector_max:.4f}")
        print(f"  均值: {vector_mean:.6f}")

        print(f"\n  观察发现:")
        print(f"    - 向量维度为{len(single_vector)}维，由模型决定")
        print(f"    - 每一维是一个浮点数，通常在[-1, 1]范围内")
        print(f"    - 均值接近0 -> 向量大致居中分布")
        print(f"    - 单看某一维没有直观含义 -> 维度是模型学习到的抽象特征")

        logger.info(f"单文本嵌入成功, 维度: {len(single_vector)}")
    except Exception as e:
        print(f"\n[错误] 创建嵌入失败: {e}")
        print("请检查API配置是否正确")
        logger.error(f"创建嵌入失败: {e}", exc_info=True)
        return

    print("\n--- 2.3 对比不同文本的嵌入向量 ---")

    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是AI的核心技术",
        "今天天气真好，适合出去散步",
    ]

    try:
        vectors = embeddings.embed_documents(texts)

        print(f"\n多文本嵌入结果:")
        print(f"  输入文本数: {len(texts)}")
        print(f"  输出向量数: {len(vectors)}")
        print(f"  每个向量维度: {len(vectors[0])}")

        import math

        def cosine_similarity(v1, v2):
            """
            计算两个向量的余弦相似度

            公式: cos(theta) = (v1 . v2) / (|v1| * |v2|)
            其中:
              v1 . v2 = 两个向量的点积（对应元素相乘再求和）
              |v1| = 向量的模（L2范数）
              |v2| = 向量的模（L2范数）

            为什么用余弦相似度而不是欧氏距离？
            - 余弦相似度只关注方向，不受向量长度影响
            - 文本长度不同会导致向量模不同，但不影响语义
            - 余弦相似度归一化到[-1,1]，便于比较和设定阈值
            """
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        print(f"\n余弦相似度矩阵:")
        print(f"  {'':<20} {'文本1':<10} {'文本2':<10} {'文本3':<10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

        for i, text_i in enumerate(texts):
            short_name = f"文本{i+1}({text_i[:6]}...)"
            row = []
            for j in range(len(texts)):
                sim = cosine_similarity(vectors[i], vectors[j])
                row.append(f"{sim:.4f}")
            print(f"  {short_name:<20} {row[0]:<10} {row[1]:<10} {row[2]:<10}")

        print(f"\n  分析:")
        sim_01 = cosine_similarity(vectors[0], vectors[1])
        sim_02 = cosine_similarity(vectors[0], vectors[2])
        sim_12 = cosine_similarity(vectors[1], vectors[2])

        print(f"    文本1('人工智能') vs 文本2('机器学习'): {sim_01:.4f}")
        print(f"    文本1('人工智能') vs 文本3('天气散步'): {sim_02:.4f}")
        print(f"    文本2('机器学习') vs 文本3('天气散步'): {sim_12:.4f}")

        if sim_01 > sim_02 and sim_01 > sim_12:
            print(f"\n    -> '人工智能'和'机器学习'的相似度最高")
            print(f"    -> 这符合直觉: 它们是相关主题")
            print(f"    -> 向量嵌入成功捕获了语义关系!")

        logger.info(f"多文本嵌入对比完成, 相似度: AI-ML={sim_01:.4f}, AI-Weather={sim_02:.4f}")
    except Exception as e:
        print(f"\n[错误] 多文本嵌入失败: {e}")
        logger.error(f"多文本嵌入失败: {e}", exc_info=True)

    print("\n--- 2.4 embed_query vs embed_documents 的区别 ---")
    print("""
    embed_query(text):
      - 用于嵌入用户查询（单个文本）
      - 返回: 一个向量（List[float]）
      - 为什么查询用单独的方法？某些模型对查询和文档使用不同的编码策略

    embed_documents(texts):
      - 用于嵌入文档（多个文本）
      - 返回: 向量列表（List[List[float]]）
      - 批量嵌入效率更高 -> 一次API调用处理多个文本

    为什么区分query和document？
    - 某些嵌入模型（如bge系列）对查询和文档使用不同前缀
    - 查询前缀: "为这个句子生成表示以用于检索相关文章："
    - 文档前缀: 无特殊前缀
    - 目的: 让查询向量和文档向量在同一个语义空间中对齐
    - OpenAI/百炼的模型通常不需要区分，但保持接口一致性是好习惯
""")

    print("--- OpenAIEmbeddings 要点总结 ---")
    print("1. OpenAIEmbeddings通过API调用嵌入模型，无需本地部署")
    print("2. embed_query()嵌入单个查询，embed_documents()批量嵌入文档")
    print("3. 嵌入向量维度由模型决定（如1536维、1024维）")
    print("4. 余弦相似度可以量化文本间的语义相似程度")
    print("5. 语义相关的文本相似度高，无关文本相似度低")

    logger.info("OpenAIEmbeddings演示完成")


def demo_faiss_create():
    """
    从文档创建FAISS向量库

    知识点：FAISS向量库

    是什么？
    FAISS（Facebook AI Similarity Search）是Facebook开源的高效向量相似度搜索库，
    支持十亿级别的向量检索。

    为什么选择FAISS而不是其他向量数据库？
    - 轻量级: 纯库而非服务 -> 无需部署数据库服务
    - 速度快: 使用C++实现，支持GPU加速 -> 检索延迟极低
    - 易上手: pip install即可使用 -> 适合学习和原型开发
    - 功能全: 支持多种索引类型和搜索策略 -> 满足大多数场景

    追问：FAISS vs Chroma vs Pinecone vs Milvus？
    - FAISS: 本地库，最快最轻量，适合单机场景
    - Chroma: 嵌入式数据库，API友好，适合开发调试
    - Pinecone: 云服务，免运维，适合生产环境
    - Milvus: 分布式数据库，适合大规模生产环境
    - 学习阶段选FAISS -> 简单直接，无需额外服务
    - 生产阶段根据规模选择 -> 小规模Chroma，大规模Milvus/Pinecone

    追问：为什么从文档创建向量库需要"加载+分割+嵌入"三步？
    - 加载: 获取原始文本 -> 没有文本就无法处理
    - 分割: 切分为合适大小的块 -> 太长语义模糊，太短信息不足
    - 嵌入: 将文本块转换为向量 -> 计算机只能处理数值
    - 三步缺一不可 -> 每一步都直接影响最终的检索质量
    """
    print("\n" + "=" * 60)
    print("3. 从文档创建FAISS向量库")
    print("=" * 60)

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from config import get_config

    config = get_config()
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("\n--- 3.1 加载和分割文档 ---")

    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        logger.error(f"数据目录不存在: {data_dir}")
        return None

    loader = DirectoryLoader(
        path=data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    documents = loader.load()
    print(f"\n加载文档数: {len(documents)}")

    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"  - {source}: {len(doc.page_content)} 字符")

    chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=chinese_separators,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"\n分割后chunk数: {len(chunks)}")
    print(f"  平均长度: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")

    print("\n--- 3.2 创建FAISS向量库 ---")

    print(f"\n创建过程:")
    print(f"  1. 对每个chunk调用嵌入模型生成向量")
    print(f"  2. 将所有向量构建FAISS索引")
    print(f"  3. 保存chunk文本和元数据的映射关系")

    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    try:
        print(f"\n正在创建向量库（需要调用嵌入模型，请稍候）...")

        batch_size = 10
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  嵌入批次 {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} ({len(batch_texts)} 条)...")
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        import numpy as np
        embedding_array = np.array(all_embeddings, dtype=np.float32)

        db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embedding_array)),
            embedding=embeddings,
            metadatas=metadatas,
        )

        print(f"\nFAISS向量库创建成功!")

        print(f"\n向量库信息:")
        print(f"  索引类型: {type(db.index).__name__}")
        print(f"  向量数量: {db.index.ntotal}")
        print(f"  向量维度: {db.index.d}")

        print(f"\n  为什么需要分批嵌入？")
        print(f"    - 阿里云百炼API限制每次请求最多10条文本")
        print(f"    - 超过限制会返回400错误")
        print(f"    - 分批处理是调用API的最佳实践")

        print(f"\n  为什么FAISS创建这么快？")
        print(f"    - FAISS使用C++实现核心计算 -> 比纯Python快几十倍")
        print(f"    - 构建索引只是将向量组织成数据结构 -> 不需要训练")
        print(f"    - 真正耗时的是嵌入模型API调用 -> 每个chunk都需要一次调用")

        logger.info(f"FAISS向量库创建成功, 向量数: {db.index.ntotal}, 维度: {db.index.d}")
        return db
    except Exception as e:
        print(f"\n[错误] 创建FAISS向量库失败: {e}")
        print("请检查API配置和网络连接")
        logger.error(f"创建FAISS向量库失败: {e}", exc_info=True)
        return None


def demo_faiss_similarity_search(db):
    """
    使用FAISS进行相似度搜索

    知识点：FAISS相似度搜索

    是什么？
    给定一个查询文本，FAISS在向量库中找到与查询最相似的文档块。

    为什么相似度搜索是RAG的核心？
    - RAG的第一步就是"检索" -> 找到与用户问题最相关的文档
    - 相似度搜索就是"检索"的具体实现方式
    - 搜索质量直接决定RAG的回答质量 -> 垃圾进，垃圾出

    追问：为什么similarity_search_with_score返回的距离分数越小越相似？
    - FAISS默认使用L2距离（欧氏距离）-> 距离越小越近
    - L2距离 = 两个向量在空间中的直线距离
    - 距离为0 -> 两个向量完全相同
    - 距离越大 -> 两个向量差异越大
    - 注意: 余弦相似度是越大越相似，L2距离是越小越相似 -> 不要混淆

    追问：为什么需要similarity_search_with_score而不只是similarity_search？
    - similarity_search只返回文档 -> 不知道相似程度
    - similarity_search_with_score返回文档+分数 -> 可以判断检索质量
    - 分数可以用于: 过滤低质量结果、调试检索效果、设置相似度阈值
    - 实际应用中，分数低于某个阈值的检索结果应该被丢弃 -> 避免引入无关信息
    """
    print("\n" + "=" * 60)
    print("4. 使用FAISS进行相似度搜索")
    print("=" * 60)

    if db is None:
        print("[跳过] 向量库未创建，请先运行demo_faiss_create()")
        logger.warning("跳过相似度搜索演示: 向量库未创建")
        return

    print("\n--- 4.1 基本相似度搜索 ---")

    queries = [
        "什么是RAG技术？",
        "Python有哪些数据结构？",
        "LangChain的核心组件有哪些？",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")

        try:
            results = db.similarity_search_with_score(query, k=3)

            print(f"  检索结果 (top-{len(results)}):")
            for i, (doc, score) in enumerate(results, 1):
                source = os.path.basename(doc.metadata.get("source", "未知"))
                content_preview = doc.page_content[:80].replace("\n", " ")
                print(f"    {i}. [距离: {score:.4f}] 来源: {source}")
                print(f"       内容: {content_preview}...")

            logger.info(f"相似度搜索: query='{query}', top1_score={results[0][1]:.4f}")
        except Exception as e:
            print(f"  [错误] 搜索失败: {e}")
            logger.error(f"相似度搜索失败: {e}", exc_info=True)

    print("\n--- 4.2 距离分数的含义 ---")
    print("""
    FAISS的similarity_search_with_score返回的距离分数:

    - 分数越小 -> 越相似（L2距离）
    - 分数 = 0 -> 完全相同
    - 分数越大 -> 越不相关

    如何判断分数是否"好"？
    - 没有绝对的阈值 -> 取决于嵌入模型和向量维度
    - 经验参考:
      L2距离 < 0.5  -> 非常相关（几乎相同的语义）
      L2距离 0.5-1.0 -> 相关（语义相近）
      L2距离 1.0-2.0 -> 弱相关（有一定关联）
      L2距离 > 2.0  -> 可能不相关

    注意: 以上阈值仅供参考，实际效果需要根据具体场景调整
""")

    print("\n--- 4.3 按相似度阈值过滤 ---")

    query = "深度学习和神经网络有什么关系？"
    print(f"\n查询: '{query}'")

    try:
        all_results = db.similarity_search_with_score(query, k=5)

        print(f"\n所有检索结果 (k=5):")
        for i, (doc, score) in enumerate(all_results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:60].replace("\n", " ")
            relevance = "相关" if score < 1.0 else ("弱相关" if score < 2.0 else "可能不相关")
            print(f"  {i}. [距离: {score:.4f}] [{relevance}] 来源: {source}")
            print(f"     内容: {content_preview}...")

        threshold = 1.5
        filtered_results = [(doc, score) for doc, score in all_results if score < threshold]

        print(f"\n按阈值过滤 (L2距离 < {threshold}):")
        print(f"  过滤前: {len(all_results)} 条结果")
        print(f"  过滤后: {len(filtered_results)} 条结果")

        if len(filtered_results) < len(all_results):
            print(f"  被过滤的{len(all_results) - len(filtered_results)}条结果可能引入无关信息")
            print(f"  -> 过滤低质量结果可以提高RAG回答的准确性")

        logger.info(f"阈值过滤演示完成: 阈值={threshold}, 过滤前={len(all_results)}, 过滤后={len(filtered_results)}")
    except Exception as e:
        print(f"  [错误] 搜索失败: {e}")
        logger.error(f"阈值过滤搜索失败: {e}", exc_info=True)

    print("\n--- 4.4 带元数据过滤的搜索 ---")

    query = "编程语言"
    print(f"\n查询: '{query}'")

    try:
        results = db.similarity_search(query, k=10)

        print(f"\n所有检索结果中按来源分组:")
        source_groups = {}
        for doc in results:
            source = os.path.basename(doc.metadata.get("source", "未知"))
            if source not in source_groups:
                source_groups[source] = 0
            source_groups[source] += 1

        for source, count in source_groups.items():
            print(f"  {source}: {count} 条")

        print(f"\n  观察: 查询'编程语言'时，python_learning.txt的结果应该更多")
        print(f"  -> 这说明向量搜索能正确识别文档的主题相关性")
        print(f"  -> 如果需要只搜索特定文档，可以使用FAISS的元数据过滤功能")
    except Exception as e:
        print(f"  [错误] 搜索失败: {e}")
        logger.error(f"元数据过滤搜索失败: {e}", exc_info=True)

    print("\n--- 相似度搜索 要点总结 ---")
    print("1. similarity_search返回最相似的文档列表")
    print("2. similarity_search_with_score额外返回距离分数")
    print("3. FAISS默认使用L2距离，分数越小越相似")
    print("4. 可以根据距离分数过滤低质量检索结果")
    print("5. 检索质量直接决定RAG的回答质量")

    logger.info("FAISS相似度搜索演示完成")


def demo_faiss_save_load(db):
    """
    FAISS向量库的保存和加载

    知识点：FAISS持久化

    是什么？
    将FAISS向量库保存到磁盘，下次使用时直接加载，无需重新创建。

    为什么需要保存和加载？
    - 创建向量库需要调用嵌入模型API -> 每次都要花费时间和费用
    - 保存到磁盘后可以反复使用 -> 一次创建，多次使用
    - 生产环境中知识库通常很大 -> 每次重新创建不现实
    - 知识库更新频率远低于查询频率 -> 预构建索引是合理的

    追问：什么时候需要重新创建向量库？
    - 知识库文档有更新 -> 添加了新文档或修改了现有文档
    - 嵌入模型更换了 -> 不同模型的向量空间不兼容
    - 分割策略调整了 -> chunk_size等参数改变
    - 其他情况直接加载即可 -> 避免不必要的API调用

    追问：为什么FAISS的保存格式包含多个文件？
    - index.faiss: FAISS索引文件 -> 存储向量数据和索引结构
    - index.pkl: Python pickle文件 -> 存储文档文本和元数据
    - 两者必须同时存在 -> 缺少任何一个都无法正确加载
    - 分离存储的原因: FAISS索引是C++对象，元数据是Python对象 -> 格式不同
    """
    print("\n" + "=" * 60)
    print("5. FAISS向量库的保存和加载")
    print("=" * 60)

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from config import get_config

    config = get_config()

    if db is None:
        print("[跳过] 向量库未创建，请先运行demo_faiss_create()")
        logger.warning("跳过保存加载演示: 向量库未创建")
        return

    save_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    print("\n--- 5.1 保存向量库 ---")

    print(f"\n保存路径: {save_path}")

    try:
        db.save_local(save_path)
        print(f"\n保存成功!")

        print(f"\n保存的文件:")
        for filename in os.listdir(save_path):
            filepath = os.path.join(save_path, filename)
            file_size = os.path.getsize(filepath)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.2f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size} B"
            print(f"  - {filename}: {size_str}")

        print(f"\n  文件说明:")
        print(f"    index.faiss: FAISS索引文件，存储向量数据和索引结构")
        print(f"    index.pkl: Python pickle文件，存储文档文本和元数据")

        logger.info(f"FAISS向量库保存成功: {save_path}")
    except Exception as e:
        print(f"\n[错误] 保存失败: {e}")
        logger.error(f"FAISS保存失败: {e}", exc_info=True)
        return

    print("\n--- 5.2 加载向量库 ---")

    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    try:
        print(f"\n从磁盘加载向量库...")
        loaded_db = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"\n加载成功!")

        print(f"\n加载后的向量库信息:")
        print(f"  向量数量: {loaded_db.index.ntotal}")
        print(f"  向量维度: {loaded_db.index.d}")

        print(f"\n  为什么需要allow_dangerous_deserialization=True？")
        print(f"    - index.pkl使用Python的pickle格式 -> pickle可以执行任意代码")
        print(f"    - 如果pickle文件被篡改 -> 加载时可能执行恶意代码")
        print(f"    - allow_dangerous_deserialization=True表示你信任这个文件来源")
        print(f"    - 只加载自己保存的文件 -> 安全; 加载来源不明的文件 -> 危险")

        logger.info(f"FAISS向量库加载成功, 向量数: {loaded_db.index.ntotal}")
    except Exception as e:
        print(f"\n[错误] 加载失败: {e}")
        logger.error(f"FAISS加载失败: {e}", exc_info=True)
        return

    print("\n--- 5.3 验证加载后的向量库 ---")

    try:
        query = "什么是RAG？"
        print(f"\n使用加载后的向量库进行搜索: '{query}'")

        results = loaded_db.similarity_search_with_score(query, k=3)

        print(f"\n检索结果:")
        for i, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:60].replace("\n", " ")
            print(f"  {i}. [距离: {score:.4f}] 来源: {source}")
            print(f"     内容: {content_preview}...")

        print(f"\n  -> 加载后的向量库与原始向量库搜索结果一致")
        print(f"  -> 验证了保存和加载的正确性")

        logger.info(f"加载后向量库验证成功, 查询: '{query}'")
    except Exception as e:
        print(f"\n[错误] 验证搜索失败: {e}")
        logger.error(f"验证搜索失败: {e}", exc_info=True)

    print("\n--- 5.4 向量库更新 ---")

    print("""
    向量库创建后，如何更新？

    方式1: 全量重建
      - 删除旧的向量库，重新加载所有文档并创建
      - 优点: 简单可靠，保证数据一致性
      - 缺点: 耗时较长，需要重新调用嵌入API
      - 适合: 文档变化较大或需要调整分割参数时

    方式2: 增量添加
      - 使用add_documents()方法添加新文档
      - 优点: 只处理新增文档，效率高
      - 缺点: 无法删除或修改已有文档
      - 适合: 只需添加新文档，不修改旧文档时

    示例代码:
      # 增量添加新文档
      new_docs = loader.load()  # 加载新文档
      new_chunks = splitter.split_documents(new_docs)  # 分割
      db.add_documents(new_chunks)  # 添加到现有向量库
      db.save_local(save_path)  # 保存更新后的向量库

    方式3: 删除重建
      - 先删除需要更新的文档，再重新添加
      - FAISS不支持直接删除单条记录 -> 需要全量重建
      - 适合: 需要删除或修改特定文档时

    生产建议:
      - 定期全量重建（如每天凌晨）-> 保证数据一致性
      - 增量添加用于实时更新 -> 新文档立即可检索
      - 保存版本备份 -> 出问题时可以回滚
""")

    print("--- FAISS保存加载 要点总结 ---")
    print("1. save_local()保存向量库到磁盘，避免重复创建")
    print("2. load_local()从磁盘加载向量库，直接使用")
    print("3. 保存包含两个文件: index.faiss(向量) + index.pkl(元数据)")
    print("4. 加载时需要allow_dangerous_deserialization=True（信任文件来源）")
    print("5. 更新方式: 全量重建、增量添加、删除重建")

    logger.info("FAISS保存加载演示完成")


def main():
    """
    主函数：依次运行所有演示

    执行顺序的设计逻辑：
    1. 向量嵌入原理 -> 先理解"是什么"和"为什么"
    2. 创建嵌入 -> 动手实践，感受向量的维度和相似度
    3. 创建FAISS向量库 -> 将文档转化为可检索的向量存储
    4. 相似度搜索 -> 体验向量检索的效果和原理
    5. 保存和加载 -> 学会持久化向量库，避免重复创建

    为什么按这个顺序？
    - 从原理到实践 -> 先理解向量嵌入的本质，再动手操作
    - 从创建到使用 -> 先有向量库，再进行检索
    - 从临时到持久 -> 先学会创建和使用，再学会保存和加载
    - 每一步都依赖前一步的结果 -> 逻辑递进
    """
    print("=" * 60)
    print("第3章：向量嵌入与FAISS存储")
    print("=" * 60)

    logger.info("开始第3章演示")

    demo_what_is_embedding()

    demo_create_embeddings()

    db = demo_faiss_create()

    demo_faiss_similarity_search(db)

    demo_faiss_save_load(db)

    print("\n" + "=" * 60)
    print("第3章学习总结")
    print("=" * 60)
    print("""
核心要点回顾：

1. 什么是向量嵌入？
   - 将文本转换为高维数值向量，捕获语义信息
   - 语义相似的文本在向量空间中距离更近
   - 向量搜索 vs 关键词搜索: 语义匹配 vs 字面匹配
   - 这是RAG能够"理解"文档内容的关键技术

2. OpenAIEmbeddings的使用
   - 通过API调用嵌入模型，将文本转换为向量
   - embed_query()嵌入查询，embed_documents()嵌入文档
   - 余弦相似度量化文本间的语义相似程度
   - 阿里云百炼兼容OpenAI接口，对中文效果更好

3. FAISS向量库的创建
   - 加载文档 -> 分割文本 -> 嵌入向量 -> 构建索引
   - FAISS轻量快速，适合学习和单机场景
   - 向量数量和维度由文档和模型决定

4. 相似度搜索
   - similarity_search返回最相似的文档
   - similarity_search_with_score额外返回距离分数
   - L2距离越小越相似，可根据阈值过滤低质量结果
   - 检索质量直接决定RAG的回答质量

5. 保存和加载
   - save_local()保存到磁盘，避免重复创建
   - load_local()从磁盘加载，直接使用
   - 更新方式: 全量重建、增量添加、删除重建

下一章预告：RAG检索器与问答链 - 如何将检索和生成串联成完整的问答系统
""")

    logger.info("第3章演示完成")


if __name__ == '__main__':
    main()
