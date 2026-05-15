#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4章：检索器与相似度搜索

学习目标：
1. 理解检索器在RAG中的作用
2. 掌握相似度搜索和MMR检索的区别
3. 学会调优检索参数
4. 理解如何评估检索质量

核心问题：为什么需要检索器？
- 向量库存储了所有文档的向量 -> 但RAG只需要最相关的几段
- 检索器是"从海量信息中找到最相关内容"的组件
- 检索质量直接决定RAG系统的回答质量 -> 垃圾进，垃圾出
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('chapter4_retriever')


def _build_vectorstore():
    """
    构建或加载FAISS向量库（内部辅助函数）

    为什么需要这个辅助函数？
    - 第4章的每个demo都需要向量库 -> 避免在每个函数中重复构建逻辑
    - 优先加载已有索引 -> 节省API调用费用和时间
    - 只在索引不存在时才重新创建 -> 按需构建

    追问：为什么不把向量库作为全局变量？
    - 全局变量在模块加载时就创建 -> 即使不使用也会消耗资源
    - 函数内部创建更灵活 -> 可以控制创建时机和条件
    - 辅助函数可以复用 -> 多个demo共享同一套构建逻辑

    Returns:
        FAISS向量库实例，失败时返回None
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from config import get_config

    config = get_config()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    if os.path.exists(index_path):
        print(f"发现已有FAISS索引，直接加载: {index_path}")
        try:
            db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"加载成功! 向量数量: {db.index.ntotal}, 维度: {db.index.d}")
            logger.info(f"加载已有FAISS索引, 向量数: {db.index.ntotal}")
            return db
        except Exception as e:
            print(f"加载已有索引失败: {e}，将重新创建")
            logger.warning(f"加载已有索引失败: {e}，将重新创建")

    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        logger.error(f"数据目录不存在: {data_dir}")
        return None

    print("未找到已有索引，开始构建向量库...")

    loader = DirectoryLoader(
        path=data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    documents = loader.load()
    print(f"加载文档数: {len(documents)}")

    chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=chinese_separators,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"分割后chunk数: {len(chunks)}")

    try:
        print("正在创建向量库（需要调用嵌入模型，请稍候）...")

        batch_size = 10
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_emb)

        import numpy as np
        embedding_array = np.array(all_embeddings, dtype=np.float32)

        db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embedding_array)),
            embedding=embeddings,
            metadatas=metadatas,
        )

        print(f"向量库创建成功! 向量数量: {db.index.ntotal}, 维度: {db.index.d}")

        db.save_local(index_path)
        print(f"向量库已保存到: {index_path}")

        logger.info(f"向量库创建并保存成功, 向量数: {db.index.ntotal}")
        return db
    except Exception as e:
        print(f"[错误] 创建向量库失败: {e}")
        logger.error(f"创建向量库失败: {e}", exc_info=True)
        return None


def demo_basic_retriever():
    """
    基础检索器：vectorstore.as_retriever()

    知识点：检索器（Retriever）

    是什么？
    检索器是LangChain中统一的信息检索接口，将"查询问题"映射为"相关文档列表"。
    它封装了向量库的搜索逻辑，提供更高级的检索策略。

    为什么需要检索器，而不直接用向量库的similarity_search？
    - vectorstore.similarity_search()只支持简单的相似度搜索 -> 功能单一
    - retriever支持多种搜索策略（相似度、MMR等） -> 灵活选择
    - retriever是LangChain Chain的标准接口 -> 可以无缝接入问答链
    - retriever的invoke()方法返回Document列表 -> 接口统一，便于组合

    追问：as_retriever()的search_type参数有什么作用？
    - "similarity": 返回最相似的k个文档 -> 最简单直接
    - "mmr": 最大边际相关性检索 -> 兼顾相关性和多样性
    - "similarity_score_threshold": 只返回相似度高于阈值的文档 -> 质量可控
    - 不同search_type适合不同场景 -> 根据需求选择

    追问：为什么检索器返回的是Document而不是原始文本？
    - Document包含page_content和metadata -> 保留了来源信息
    - RAG生成回答时需要引用来源 -> metadata中的source字段用于溯源
    - 后续处理可能需要元数据过滤 -> metadata提供额外维度
    """
    print("=" * 60)
    print("1. 基础检索器")
    print("=" * 60)

    print("""
检索器（Retriever）的核心概念：

  向量库 vs 检索器：
    向量库（VectorStore）: 存储和搜索的底层引擎
    检索器（Retriever）:  封装搜索逻辑的高级接口

  类比理解：
    向量库 = 数据库（存储数据，支持SQL查询）
    检索器 = ORM（封装查询逻辑，提供友好的接口）

  为什么需要检索器？
    - 直接用向量库: db.similarity_search(query, k=4) -> 只能做相似度搜索
    - 用检索器: retriever = db.as_retriever(search_type="mmr") -> 支持多种策略
    - 检索器可以接入LangChain的Chain -> 构建完整的问答流程
""")

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过基础检索器演示: 向量库构建失败")
        return

    print("\n--- 1.1 similarity检索器（默认） ---")

    retriever_sim = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    print(f"\n创建检索器:")
    print(f"  search_type: similarity")
    print(f"  search_kwargs: {{'k': 3}}")
    print(f"\n  参数含义:")
    print(f"    k=3: 返回最相似的3个文档块")
    print(f"    similarity: 按相似度从高到低返回，不做任何过滤")

    query = "什么是RAG？"
    print(f"\n查询: '{query}'")

    try:
        docs = retriever_sim.invoke(query)
        print(f"\n检索结果 ({len(docs)} 条):")
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. 来源: {source}")
            print(f"     内容: {content_preview}...")

        logger.info(f"similarity检索器演示完成, 查询: '{query}', 结果数: {len(docs)}")
    except Exception as e:
        print(f"  [错误] 检索失败: {e}")
        logger.error(f"similarity检索失败: {e}", exc_info=True)

    print("\n--- 1.2 similarity_score_threshold检索器 ---")

    retriever_threshold = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )

    print(f"\n创建检索器:")
    print(f"  search_type: similarity_score_threshold")
    print(f"  search_kwargs: {{'k': 5, 'score_threshold': 0.5}}")
    print(f"\n  参数含义:")
    print(f"    k=5: 最多返回5个文档块")
    print(f"    score_threshold=0.5: 只返回相似度分数 >= 0.5 的结果")
    print(f"\n  为什么需要score_threshold？")
    print(f"    - 没有阈值时，即使完全不相关的文档也会被返回")
    print(f"    - 设定阈值可以过滤掉低质量结果 -> 避免引入噪音")
    print(f"    - 阈值需要根据实际效果调整 -> 太高会漏掉相关结果，太低会引入噪音")

    query = "深度学习框架有哪些？"
    print(f"\n查询: '{query}'")

    try:
        docs = retriever_threshold.invoke(query)
        print(f"\n检索结果 ({len(docs)} 条):")
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. 来源: {source}")
            print(f"     内容: {content_preview}...")

        if len(docs) == 0:
            print(f"\n  注意: 没有结果超过阈值 -> 可能需要降低score_threshold")
            print(f"  -> 这说明阈值设置需要根据实际数据调整")

        logger.info(f"score_threshold检索器演示完成, 查询: '{query}', 结果数: {len(docs)}")
    except Exception as e:
        print(f"  [错误] 检索失败: {e}")
        logger.error(f"score_threshold检索失败: {e}", exc_info=True)

    print("\n--- 1.3 retriever.invoke() vs db.similarity_search() ---")
    print("""
    两者的区别：

    db.similarity_search(query, k=3):
      - 直接调用向量库的搜索方法
      - 返回: List[Document]
      - 只支持相似度搜索
      - 适合: 简单的搜索场景

    retriever.invoke(query):
      - 通过检索器接口调用
      - 返回: List[Document]
      - 支持多种搜索策略（similarity, mmr, score_threshold）
      - 适合: 需要灵活检索策略或接入Chain的场景

    为什么推荐使用retriever？
    - retriever是LangChain的标准接口 -> 可以无缝替换实现
    - 可以轻松切换搜索策略 -> 只需修改search_type参数
    - 可以直接传入RetrievalQA等Chain -> 构建完整问答系统
    - 符合"面向接口编程"原则 -> 降低耦合，提高灵活性
""")

    print("--- 基础检索器 要点总结 ---")
    print("1. as_retriever()将向量库包装为检索器接口")
    print("2. search_type='similarity'按相似度返回top-k结果")
    print("3. search_type='similarity_score_threshold'按阈值过滤低质量结果")
    print("4. retriever.invoke()是LangChain标准的检索接口")
    print("5. 推荐使用retriever而非直接调用similarity_search")

    logger.info("基础检索器演示完成")


def demo_similarity_search():
    """
    相似度搜索：similarity_search_with_score()，展示距离分数

    知识点：相似度搜索与距离分数

    是什么？
    相似度搜索是向量检索的基础方法，通过计算查询向量与所有文档向量的距离，
    返回距离最近（最相似）的文档。

    为什么需要关注距离分数？
    - 只看返回结果无法判断检索质量 -> 分数量化了相似程度
    - 分数可以用于过滤低质量结果 -> 避免引入无关信息
    - 分数可以用于排序和加权 -> 更精细地控制检索结果的使用
    - 分数是调试检索效果的依据 -> 发现问题、优化参数

    追问：FAISS的L2距离和余弦相似度有什么关系？
    - L2距离: 两个向量在空间中的直线距离，越小越相似
    - 余弦相似度: 两个向量夹角的余弦值，越大越相似
    - 当向量归一化后（长度为1），L2距离和余弦相似度有确定的数学关系
    - L2^2 = 2 - 2*cos(theta) -> 距离越小，夹角越小，越相似
    - FAISS默认使用L2距离 -> 但语义上等价于余弦相似度排序

    追问：为什么不同查询的距离分数范围差异很大？
    - 查询的语义清晰度不同 -> "什么是RAG？"比"介绍一下"更具体
    - 文档库中相关文档的密度不同 -> 有些主题文档多，有些少
    - 嵌入模型的特性 -> 不同模型产生的向量分布不同
    - 因此不能用固定阈值判断"是否相关" -> 需要根据实际效果调整
    """
    print("\n" + "=" * 60)
    print("2. 相似度搜索与距离分数")
    print("=" * 60)

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过相似度搜索演示: 向量库构建失败")
        return

    print("\n--- 2.1 基本相似度搜索（带分数） ---")

    query = "什么是RAG？"
    print(f"\n查询: '{query}'")

    try:
        results = db.similarity_search_with_score(query, k=5)

        print(f"\n检索结果 (k=5):")
        for i, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. [L2距离: {score:.4f}] 来源: {source}")
            print(f"     内容: {content_preview}...")

        print(f"\n  距离分数解读:")
        print(f"    - L2距离越小 -> 越相似")
        print(f"    - 第1条结果距离最小 -> 与查询最相关")
        print(f"    - 距离递增 -> 相关性递减")

        logger.info(f"相似度搜索演示, 查询: '{query}', 结果数: {len(results)}")
    except Exception as e:
        print(f"  [错误] 搜索失败: {e}")
        logger.error(f"相似度搜索失败: {e}", exc_info=True)

    print("\n--- 2.2 不同查询的分数对比 ---")

    queries = [
        "什么是RAG？",
        "Python有哪些数据结构？",
        "LangChain的核心组件有哪些？",
        "今天天气怎么样？",
    ]

    print(f"\n对比不同查询的检索分数:")
    print(f"  {'查询':<30} {'最小距离':<12} {'最大距离':<12} {'平均距离':<12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    for q in queries:
        try:
            results = db.similarity_search_with_score(q, k=5)
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            print(f"  {q:<30} {min_score:<12.4f} {max_score:<12.4f} {avg_score:<12.4f}")
            logger.info(f"查询: '{q}', 最小距离: {min_score:.4f}, 平均距离: {avg_score:.4f}")
        except Exception as e:
            print(f"  {q:<30} [搜索失败: {e}]")
            logger.error(f"查询失败: '{q}', 错误: {e}")

    print(f"\n  分析:")
    print(f"    - 与知识库相关的查询（RAG、Python、LangChain）-> 距离分数较低（更相似）")
    print(f"    - 与知识库无关的查询（天气）-> 距离分数较高（不太相似）")
    print(f"    - 这验证了向量搜索能区分相关和无关的查询")
    print(f"    - 但即使无关查询也会返回结果 -> 需要阈值过滤")

    print("\n--- 2.3 similarity_search vs similarity_search_with_score ---")
    print("""
    两个方法的区别：

    similarity_search(query, k=3):
      - 返回: List[Document] -> 只有文档，没有分数
      - 适合: 只需要文档内容，不关心相似程度的场景
      - 缺点: 无法判断检索质量，无法过滤低质量结果

    similarity_search_with_score(query, k=3):
      - 返回: List[Tuple[Document, float]] -> 文档 + 距离分数
      - 适合: 需要评估检索质量、设置阈值的场景
      - 优点: 分数可以用于过滤、排序、调试

    为什么推荐使用similarity_search_with_score？
    - 分数是检索质量的量化指标 -> 可以判断结果是否可靠
    - 可以基于分数设置过滤阈值 -> 丢弃低质量结果
    - 调试时可以查看分数 -> 快速定位检索问题
    - 生产环境中分数是重要的监控指标 -> 发现性能退化
""")

    print("--- 相似度搜索 要点总结 ---")
    print("1. similarity_search_with_score返回文档+L2距离分数")
    print("2. L2距离越小越相似，距离为0表示完全相同")
    print("3. 相关查询的距离分数通常低于无关查询")
    print("4. 即使无关查询也会返回结果 -> 需要阈值过滤")
    print("5. 推荐使用带分数的搜索方法，便于评估和调试")

    logger.info("相似度搜索演示完成")


def demo_mmr_retriever():
    """
    MMR检索器：最大边际相关性，兼顾相关性和多样性

    知识点：最大边际相关性（Maximal Marginal Relevance, MMR）

    是什么？
    MMR是一种检索策略，在保证结果与查询相关的同时，尽量减少结果之间的冗余。
    它在"与查询的相关性"和"结果之间的多样性"之间寻找平衡。

    为什么需要MMR？
    - 普通相似度搜索可能返回高度重复的结果 -> 信息冗余
    - 例如查询"Python数据结构"，可能返回3段内容几乎相同的文档
    - MMR确保返回的结果既相关又多样 -> 覆盖更多维度的信息
    - RAG需要多样化的上下文 -> 单一视角的信息不足以生成全面的回答

    追问：MMR的数学原理是什么？
    - MMR评分 = lambda * relevance - (1-lambda) * redundancy
    - relevance: 候选文档与查询的相似度
    - redundancy: 候选文档与已选文档的最大相似度
    - lambda控制相关性和多样性的权衡:
      lambda=1.0 -> 退化为纯相似度搜索（只看相关性）
      lambda=0.0 -> 只追求多样性（忽略相关性）
      lambda=0.5 -> 均衡权衡（默认值）

    追问：为什么MMR比纯相似度搜索更适合RAG？
    - RAG的上下文窗口有限 -> 每个位置都很珍贵，不能浪费在重复信息上
    - 多样化的上下文帮助LLM生成更全面的回答 -> 覆盖多个角度
    - 减少冗余信息 -> LLM不容易被重复内容误导
    - 但MMR计算量更大 -> 需要计算文档间的相似度矩阵
    """
    print("\n" + "=" * 60)
    print("3. MMR检索器（最大边际相关性）")
    print("=" * 60)

    print("""
MMR（Maximal Marginal Relevance）核心思想：

  问题: 普通相似度搜索可能返回高度重复的结果
    查询: "Python数据结构"
    相似度搜索结果:
      1. "Python的列表是一种有序的数据结构..." (文档A)
      2. "Python列表是有序的数据结构，可以..." (文档B)  <- 和第1条几乎一样
      3. "Python的列表作为数据结构，支持..." (文档C)  <- 还是差不多

    -> 3条结果都在说同一件事 -> 信息冗余，浪费了上下文窗口

  MMR搜索结果:
      1. "Python的列表是一种有序的数据结构..." (文档A)
      2. "Python的字典是键值对的数据结构..." (文档D)  <- 不同的数据结构
      3. "Python的集合是无序不重复的数据结构..." (文档E) <- 又一种不同的

    -> 3条结果覆盖了3种数据结构 -> 信息多样，上下文利用率高

  MMR评分公式:
    score = lambda * similarity(query, doc) - (1-lambda) * max_similarity(doc, selected)
    - 第一项: 候选文档与查询的相似度 -> 越相关越好
    - 第二项: 候选文档与已选文档的最大相似度 -> 越冗余越差
    - lambda: 权衡参数，0到1之间
""")

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过MMR检索器演示: 向量库构建失败")
        return

    print("\n--- 3.1 对比相似度搜索和MMR搜索 ---")

    query = "Python有哪些数据结构？"
    print(f"\n查询: '{query}'")

    try:
        sim_results = db.similarity_search_with_score(query, k=4)
        print(f"\n相似度搜索结果 (k=4):")
        for i, (doc, score) in enumerate(sim_results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. [距离: {score:.4f}] 来源: {source}")
            print(f"     内容: {content_preview}...")

        mmr_results = db.max_marginal_relevance_search(query, k=4, fetch_k=10)
        print(f"\nMMR搜索结果 (k=4, fetch_k=10):")
        for i, doc in enumerate(mmr_results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. 来源: {source}")
            print(f"     内容: {content_preview}...")

        print(f"\n  对比分析:")
        print(f"    相似度搜索: 优先返回与查询最相似的文档 -> 可能内容重复")
        print(f"    MMR搜索: 在相似的基础上增加多样性约束 -> 内容覆盖面更广")
        print(f"    fetch_k=10: 先取10个候选，再从中选4个最多样化的")

        logger.info(f"MMR vs 相似度对比完成, 查询: '{query}'")
    except Exception as e:
        print(f"  [错误] 搜索失败: {e}")
        logger.error(f"MMR搜索失败: {e}", exc_info=True)

    print("\n--- 3.2 使用as_retriever创建MMR检索器 ---")

    retriever_mmr = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        }
    )

    print(f"\n创建MMR检索器:")
    print(f"  search_type: mmr")
    print(f"  search_kwargs:")
    print(f"    k=4: 最终返回4个文档")
    print(f"    fetch_k=10: 先从向量库取10个候选文档")
    print(f"    lambda_mult=0.5: 相关性和多样性的权衡系数")
    print(f"\n  lambda_mult参数详解:")
    print(f"    lambda_mult=1.0 -> 只看相关性，退化为纯相似度搜索")
    print(f"    lambda_mult=0.5 -> 均衡权衡相关性和多样性（默认值）")
    print(f"    lambda_mult=0.0 -> 只看多样性，忽略与查询的相关性")
    print(f"    实际使用中0.5-0.7通常是较好的选择")

    query = "LangChain的核心组件有哪些？"
    print(f"\n查询: '{query}'")

    try:
        docs = retriever_mmr.invoke(query)
        print(f"\nMMR检索结果 ({len(docs)} 条):")
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"  {i}. 来源: {source}")
            print(f"     内容: {content_preview}...")

        logger.info(f"MMR检索器演示完成, 查询: '{query}', 结果数: {len(docs)}")
    except Exception as e:
        print(f"  [错误] 检索失败: {e}")
        logger.error(f"MMR检索器失败: {e}", exc_info=True)

    print("\n--- 3.3 不同lambda_mult值的效果对比 ---")

    query = "什么是人工智能？"
    print(f"\n查询: '{query}'")

    lambda_values = [1.0, 0.7, 0.5, 0.3]

    for lam in lambda_values:
        try:
            results = db.max_marginal_relevance_search(
                query, k=4, fetch_k=10, lambda_mult=lam
            )
            print(f"\n  lambda_mult={lam}:")
            for i, doc in enumerate(results, 1):
                source = os.path.basename(doc.metadata.get("source", "未知"))
                content_preview = doc.page_content[:60].replace("\n", " ")
                print(f"    {i}. [{source}] {content_preview}...")
        except Exception as e:
            print(f"\n  lambda_mult={lam}: [搜索失败: {e}]")

    print(f"\n  分析:")
    print(f"    lambda_mult=1.0: 纯相似度 -> 结果最相关，但可能重复")
    print(f"    lambda_mult=0.7: 偏向相关性 -> 结果较相关，有一定多样性")
    print(f"    lambda_mult=0.5: 均衡 -> 相关性和多样性兼顾")
    print(f"    lambda_mult=0.3: 偏向多样性 -> 结果多样，但可能不够相关")
    print(f"    -> 实际使用中需要根据效果调整lambda_mult值")

    print("\n--- 3.4 fetch_k参数的作用 ---")
    print("""
    fetch_k参数的含义：
      - MMR先从向量库中取fetch_k个最相似的候选文档
      - 然后从这fetch_k个候选中，按MMR策略选出k个最终结果
      - fetch_k必须 >= k -> 否则无法选出足够的结果

    为什么需要fetch_k？
      - MMR需要在候选集中计算文档间的相似度 -> 候选集太小无法保证多样性
      - fetch_k越大，候选集越大，多样性选择空间越大
      - 但fetch_k越大，计算量也越大 -> 需要权衡

    fetch_k的选择建议：
      - fetch_k = 2*k 到 3*k: 适中的候选集，兼顾效果和性能
      - fetch_k = 10*k: 更大的候选集，多样性更好，但计算量更大
      - 实际效果取决于数据分布 -> 需要实验确定最佳值
""")

    print("--- MMR检索器 要点总结 ---")
    print("1. MMR在相关性和多样性之间寻找平衡，减少结果冗余")
    print("2. lambda_mult控制权衡: 1.0=纯相关性, 0.5=均衡, 0.0=纯多样性")
    print("3. fetch_k决定候选集大小，影响多样性的选择空间")
    print("4. MMR比纯相似度搜索更适合RAG -> 上下文窗口有限，不能浪费在重复信息上")
    print("5. 实际使用中lambda_mult=0.5-0.7通常是较好的起点")

    logger.info("MMR检索器演示完成")


def demo_parameter_tuning():
    """
    参数调优：对比不同k值、fetch_k值的效果

    知识点：检索参数调优

    是什么？
    检索参数（k、fetch_k、lambda_mult、score_threshold等）直接影响检索质量，
    需要根据实际数据和需求进行调整。

    为什么参数调优很重要？
    - k太小 -> 信息不足，LLM缺少关键上下文
    - k太大 -> 引入噪音，LLM被无关信息干扰，且浪费token
    - fetch_k太小 -> MMR的多样性选择空间不足
    - lambda_mult不合适 -> 相关性或多样性失衡
    - 参数没有"万能最优值" -> 必须根据实际效果调整

    追问：为什么k值的选择这么重要？
    - k决定了送给LLM的上下文量 -> 直接影响生成质量
    - k太小: 信息不足 -> LLM可能无法回答或回答不完整
    - k太大: 噪音过多 -> LLM可能被无关信息误导（"迷失在中间"问题）
    - 研究表明LLM更关注上下文的开头和结尾 -> 中间的信息容易被忽略
    - 因此k值不是越大越好 -> 需要在信息量和噪音之间权衡

    追问：为什么不存在"万能最优参数"？
    - 不同领域的文档分布不同 -> 技术文档和小说的最佳k值不同
    - 不同查询的难度不同 -> 简单问题k=2就够了，复杂问题可能需要k=6
    - 不同LLM的上下文窗口不同 -> GPT-4和Claude的窗口大小不同
    - 因此参数调优必须基于实际数据和场景 -> 没有捷径
    """
    print("\n" + "=" * 60)
    print("4. 检索参数调优")
    print("=" * 60)

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过参数调优演示: 向量库构建失败")
        return

    print("\n--- 4.1 不同k值的效果对比 ---")

    query = "什么是RAG？"
    print(f"\n查询: '{query}'")
    print(f"\n对比不同k值的检索结果:")

    k_values = [1, 2, 3, 5, 8]

    for k in k_values:
        try:
            results = db.similarity_search_with_score(query, k=k)
            scores = [score for _, score in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)

            print(f"\n  k={k}:")
            print(f"    结果数: {len(results)}")
            print(f"    最小距离: {min(scores):.4f} (最相关)")
            print(f"    最大距离: {max_score:.4f} (最不相关)")
            print(f"    平均距离: {avg_score:.4f}")

            if k <= 3:
                for i, (doc, score) in enumerate(results, 1):
                    source = os.path.basename(doc.metadata.get("source", "未知"))
                    content_preview = doc.page_content[:50].replace("\n", " ")
                    print(f"      {i}. [{score:.4f}] {source}: {content_preview}...")

            logger.info(f"k={k}参数对比, 平均距离: {avg_score:.4f}")
        except Exception as e:
            print(f"\n  k={k}: [搜索失败: {e}]")

    print(f"\n  分析:")
    print(f"    k=1: 只返回最相关的1条 -> 信息可能不够全面")
    print(f"    k=2-3: 返回2-3条最相关的 -> 通常是最实用的范围")
    print(f"    k=5: 返回5条 -> 信息更全面，但可能引入弱相关内容")
    print(f"    k=8: 返回8条 -> 覆盖面广，但噪音增加，浪费token")
    print(f"    -> k值选择需要在信息量和噪音之间权衡")

    print("\n--- 4.2 不同fetch_k值对MMR的影响 ---")

    query = "Python有哪些数据结构？"
    print(f"\n查询: '{query}'")
    print(f"\n对比不同fetch_k值的MMR检索结果 (k=3, lambda_mult=0.5):")

    fetch_k_values = [3, 6, 10, 15]

    for fk in fetch_k_values:
        try:
            results = db.max_marginal_relevance_search(
                query, k=3, fetch_k=fk, lambda_mult=0.5
            )

            sources = []
            for doc in results:
                source = os.path.basename(doc.metadata.get("source", "未知"))
                sources.append(source)

            unique_sources = len(set(sources))

            print(f"\n  fetch_k={fk}:")
            print(f"    结果来源: {sources}")
            print(f"    不同来源数: {unique_sources}/{len(sources)}")

            for i, doc in enumerate(results, 1):
                source = os.path.basename(doc.metadata.get("source", "未知"))
                content_preview = doc.page_content[:50].replace("\n", " ")
                print(f"      {i}. [{source}] {content_preview}...")

            logger.info(f"fetch_k={fk}对比, 不同来源数: {unique_sources}")
        except Exception as e:
            print(f"\n  fetch_k={fk}: [搜索失败: {e}]")

    print(f"\n  分析:")
    print(f"    fetch_k=3 (=k): 没有多样性选择空间 -> 退化为相似度搜索")
    print(f"    fetch_k=6 (2*k): 有一定选择空间 -> 多样性略有提升")
    print(f"    fetch_k=10 (3*k): 选择空间充足 -> 多样性较好")
    print(f"    fetch_k=15 (5*k): 选择空间很大 -> 但计算量也更大")
    print(f"    -> fetch_k=2*k到3*k通常是较好的起点")

    print("\n--- 4.3 不同查询的最佳k值可能不同 ---")

    queries = [
        ("什么是RAG？", 2),
        ("Python有哪些数据结构？", 4),
        ("LangChain的核心组件有哪些？", 3),
    ]

    print(f"\n不同查询的推荐k值:")
    for q, recommended_k in queries:
        try:
            results = db.similarity_search_with_score(q, k=6)
            scores = [score for _, score in results]

            print(f"\n  查询: '{q}'")
            print(f"    推荐k值: {recommended_k}")

            significant_gap = None
            for i in range(1, len(scores)):
                gap = scores[i] - scores[i - 1]
                if significant_gap is None or gap > significant_gap[1]:
                    significant_gap = (i, gap)

            if significant_gap:
                print(f"    最大距离跳跃: 第{significant_gap[0]}到第{significant_gap[0]+1}条")
                print(f"      跳跃幅度: {significant_gap[1]:.4f}")
                print(f"    -> 跳跃点之前的结果相关性较高，之后可能引入噪音")
                print(f"    -> 可以将k设为跳跃点位置附近")

            logger.info(f"查询: '{q}', 推荐k: {recommended_k}")
        except Exception as e:
            print(f"\n  查询: '{q}' [搜索失败: {e}]")

    print("\n--- 4.4 参数调优的实践建议 ---")
    print("""
    参数调优的通用策略：

    1. 从默认值开始
       k=3-4, fetch_k=10, lambda_mult=0.5
       -> 默认值在大多数场景下效果不错

    2. 观察检索结果
       - 结果是否相关？ -> 不相关则减小k或增加score_threshold
       - 结果是否重复？ -> 重复则使用MMR或减小lambda_mult
       - 结果是否全面？ -> 不全面则增大k或增大fetch_k

    3. 量化评估
       - 使用一组测试问题评估检索质量
       - 关注精确率（返回结果中有多少是相关的）
       - 关注召回率（所有相关结果中有多少被返回了）

    4. 迭代优化
       - 调整一个参数 -> 评估效果 -> 再调整下一个
       - 不要同时调整多个参数 -> 无法确定哪个变化起了作用
       - 记录每次调整的参数和效果 -> 便于回溯

    5. 考虑LLM的上下文窗口
       - k * 平均chunk长度 < LLM上下文窗口的50%
       - 留出空间给系统提示和生成的回答
       - 例如: 上下文4K, chunk平均200字 -> k最大约10
""")

    print("--- 参数调优 要点总结 ---")
    print("1. k值决定返回的文档数量，需要在信息量和噪音之间权衡")
    print("2. k=2-4通常是较好的起点，根据效果调整")
    print("3. fetch_k影响MMR的多样性选择空间，建议2*k到3*k")
    print("4. 不同查询的最佳参数可能不同 -> 需要灵活调整")
    print("5. 参数调优应基于量化评估，而非主观感受")

    logger.info("参数调优演示完成")


def demo_retrieval_quality():
    """
    检索质量评估：用多个测试问题评估检索效果

    知识点：检索质量评估

    是什么？
    用一组预定义的测试问题和期望答案，量化评估检索器的效果，
    包括精确率、召回率等指标。

    为什么需要评估检索质量？
    - 主观感受不可靠 -> 需要量化指标来客观评估
    - 参数调整后需要验证效果 -> 评估提供客观依据
    - 不同检索策略需要对比 -> 评估提供公平的比较基础
    - 生产环境需要持续监控 -> 评估可以发现性能退化

    追问：为什么检索质量评估比生成质量评估更容易？
    - 检索评估: 判断"返回的文档是否与问题相关" -> 相对客观
    - 生成评估: 判断"生成的回答是否正确" -> 需要领域知识，更主观
    - 检索评估可以自动化 -> 定义相关/不相关的标准
    - 生成评估通常需要人工判断 -> 成本高、速度慢

    追问：精确率和召回率哪个更重要？
    - 精确率: 返回的结果中有多少是相关的 -> 不浪费上下文窗口
    - 召回率: 所有相关结果中有多少被返回了 -> 不遗漏重要信息
    - RAG场景中，精确率通常更重要 -> 上下文窗口有限，噪音比遗漏更危险
    - 但如果遗漏了关键信息，LLM就无法正确回答 -> 召回率也不能太低
    - 实践中需要在两者之间找到平衡点
    """
    print("\n" + "=" * 60)
    print("5. 检索质量评估")
    print("=" * 60)

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过检索质量评估: 向量库构建失败")
        return

    print("\n--- 5.1 定义测试问题集 ---")

    test_cases = [
        {
            "query": "什么是RAG？",
            "expected_source": "ai_basics.txt",
            "expected_keywords": ["RAG", "检索", "生成"],
        },
        {
            "query": "Python有哪些数据结构？",
            "expected_source": "python_learning.txt",
            "expected_keywords": ["Python", "数据结构", "列表"],
        },
        {
            "query": "LangChain的核心组件有哪些？",
            "expected_source": "langchain_intro.txt",
            "expected_keywords": ["LangChain", "组件"],
        },
        {
            "query": "什么是机器学习？",
            "expected_source": "ai_basics.txt",
            "expected_keywords": ["机器学习", "学习"],
        },
        {
            "query": "Python的函数怎么定义？",
            "expected_source": "python_learning.txt",
            "expected_keywords": ["Python", "函数"],
        },
    ]

    print(f"\n测试问题集 ({len(test_cases)} 个):")
    for i, tc in enumerate(test_cases, 1):
        print(f"  {i}. 查询: '{tc['query']}'")
        print(f"     期望来源: {tc['expected_source']}")
        print(f"     期望关键词: {tc['expected_keywords']}")

    print("\n--- 5.2 评估相似度搜索的检索质量 ---")

    k = 3
    print(f"\n评估参数: k={k}")

    sim_results = []

    for tc in test_cases:
        try:
            results = db.similarity_search_with_score(tc["query"], k=k)

            top_sources = [
                os.path.basename(doc.metadata.get("source", "未知"))
                for doc, _ in results
            ]

            source_hit = tc["expected_source"] in top_sources

            all_content = " ".join(doc.page_content for doc, _ in results)
            keyword_hits = [
                kw for kw in tc["expected_keywords"]
                if kw in all_content
            ]
            keyword_coverage = len(keyword_hits) / len(tc["expected_keywords"])

            sim_results.append({
                "query": tc["query"],
                "source_hit": source_hit,
                "keyword_coverage": keyword_coverage,
                "top_score": results[0][1] if results else None,
            })

            print(f"\n  查询: '{tc['query']}'")
            print(f"    来源命中: {'是' if source_hit else '否'} (期望: {tc['expected_source']})")
            print(f"    关键词覆盖: {keyword_coverage:.0%} ({keyword_hits}/{tc['expected_keywords']})")
            if results:
                print(f"    最小距离: {results[0][1]:.4f}")

            logger.info(f"相似度评估: '{tc['query']}', 来源命中: {source_hit}, 关键词覆盖: {keyword_coverage:.0%}")
        except Exception as e:
            print(f"\n  查询: '{tc['query']}' [评估失败: {e}]")
            sim_results.append({
                "query": tc["query"],
                "source_hit": False,
                "keyword_coverage": 0.0,
                "top_score": None,
            })

    print("\n--- 5.3 评估MMR搜索的检索质量 ---")

    print(f"\n评估参数: k={k}, fetch_k=10, lambda_mult=0.5")

    mmr_results = []

    for tc in test_cases:
        try:
            results = db.max_marginal_relevance_search(
                tc["query"], k=k, fetch_k=10, lambda_mult=0.5
            )

            top_sources = [
                os.path.basename(doc.metadata.get("source", "未知"))
                for doc in results
            ]

            source_hit = tc["expected_source"] in top_sources

            all_content = " ".join(doc.page_content for doc in results)
            keyword_hits = [
                kw for kw in tc["expected_keywords"]
                if kw in all_content
            ]
            keyword_coverage = len(keyword_hits) / len(tc["expected_keywords"])

            mmr_results.append({
                "query": tc["query"],
                "source_hit": source_hit,
                "keyword_coverage": keyword_coverage,
            })

            print(f"\n  查询: '{tc['query']}'")
            print(f"    来源命中: {'是' if source_hit else '否'} (期望: {tc['expected_source']})")
            print(f"    关键词覆盖: {keyword_coverage:.0%} ({keyword_hits}/{tc['expected_keywords']})")

            logger.info(f"MMR评估: '{tc['query']}', 来源命中: {source_hit}, 关键词覆盖: {keyword_coverage:.0%}")
        except Exception as e:
            print(f"\n  查询: '{tc['query']}' [评估失败: {e}]")
            mmr_results.append({
                "query": tc["query"],
                "source_hit": False,
                "keyword_coverage": 0.0,
            })

    print("\n--- 5.4 两种检索策略的对比汇总 ---")

    sim_source_rate = sum(1 for r in sim_results if r["source_hit"]) / len(sim_results)
    sim_keyword_avg = sum(r["keyword_coverage"] for r in sim_results) / len(sim_results)

    mmr_source_rate = sum(1 for r in mmr_results if r["source_hit"]) / len(mmr_results)
    mmr_keyword_avg = sum(r["keyword_coverage"] for r in mmr_results) / len(mmr_results)

    print(f"\n  {'指标':<20} {'相似度搜索':<15} {'MMR搜索':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'来源命中率':<20} {sim_source_rate:<15.0%} {mmr_source_rate:<15.0%}")
    print(f"  {'关键词覆盖率':<20} {sim_keyword_avg:<15.0%} {mmr_keyword_avg:<15.0%}")

    print(f"\n  分析:")
    print(f"    - 来源命中率: 检索结果中是否包含期望来源的文档")
    print(f"    - 关键词覆盖率: 检索结果中包含多少期望关键词")
    print(f"    - 两种策略各有优势 -> 具体效果取决于查询和数据分布")
    print(f"    - 相似度搜索: 在来源命中上可能更好（优先最相似的）")
    print(f"    - MMR搜索: 在关键词覆盖上可能更好（多样性带来更广覆盖）")

    print("\n--- 5.5 检索质量评估的进阶方法 ---")
    print("""
    本节使用的是简化的评估方法，生产环境中可以使用更专业的评估框架：

    1. RAGAS框架（RAG Assessment）
       - 专门为RAG系统设计的评估框架
       - 指标包括: 上下文精确率、上下文召回率、答案相关性、忠实度
       - 可以自动化评估，无需人工标注

    2. 人工评估
       - 最可靠但成本最高
       - 适合: 建立基准、验证自动评估的准确性
       - 建议: 定期抽样人工评估，校准自动评估指标

    3. LLM-as-Judge
       - 用另一个LLM来评估检索和生成质量
       - 成本适中，速度较快
       - 缺点: 评估LLM本身可能有偏见

    4. A/B测试
       - 在生产环境中对比不同检索策略
       - 最接近真实效果的评估方法
       - 需要足够的流量和统计显著性

    评估驱动的优化循环：
      定义评估指标 -> 建立基准 -> 调整参数 -> 重新评估 -> 对比效果
      -> 持续迭代，直到达到目标
""")

    print("--- 检索质量评估 要点总结 ---")
    print("1. 检索质量评估用量化指标替代主观感受")
    print("2. 来源命中率和关键词覆盖率是简单有效的评估指标")
    print("3. 相似度搜索和MMR搜索各有优势，需要根据场景选择")
    print("4. 生产环境推荐使用RAGAS等专业评估框架")
    print("5. 评估驱动的迭代优化是提升检索质量的关键方法")

    logger.info("检索质量评估演示完成")


def main():
    """
    主函数：依次运行所有演示

    执行顺序的设计逻辑：
    1. 基础检索器 -> 理解检索器是什么，与向量库的区别
    2. 相似度搜索 -> 掌握距离分数的含义和用途
    3. MMR检索器 -> 理解多样性和相关性的权衡
    4. 参数调优 -> 学会根据实际效果调整检索参数
    5. 检索质量评估 -> 用量化指标评估和对比检索效果

    为什么按这个顺序？
    - 从基础到进阶 -> 先理解检索器的基本概念，再学习高级策略
    - 从原理到实践 -> 先理解MMR的原理，再进行参数调优
    - 从调优到评估 -> 先学会调优，再学会评估调优效果
    - 每一步都为下一步打基础 -> 逻辑递进
    """
    print("=" * 60)
    print("第4章：检索器与相似度搜索")
    print("=" * 60)

    logger.info("开始第4章演示")

    demo_basic_retriever()

    demo_similarity_search()

    demo_mmr_retriever()

    demo_parameter_tuning()

    demo_retrieval_quality()

    print("\n" + "=" * 60)
    print("第4章学习总结")
    print("=" * 60)
    print("""
核心要点回顾：

1. 什么是检索器？
   - 检索器是LangChain中统一的信息检索接口
   - 封装了向量库的搜索逻辑，支持多种检索策略
   - as_retriever()将向量库包装为检索器接口
   - retriever.invoke()是LangChain标准的检索调用方式

2. 相似度搜索
   - similarity_search_with_score返回文档+L2距离分数
   - L2距离越小越相似，可用于过滤低质量结果
   - 距离分数是评估检索质量的重要量化指标
   - 相关查询的距离通常低于无关查询

3. MMR检索器
   - MMR在相关性和多样性之间寻找平衡
   - lambda_mult控制权衡: 1.0=纯相关性, 0.5=均衡
   - fetch_k决定候选集大小，影响多样性选择空间
   - MMR比纯相似度搜索更适合RAG -> 减少上下文冗余

4. 参数调优
   - k值决定返回文档数量，k=2-4通常是较好起点
   - fetch_k建议设为2*k到3*k
   - 不同查询的最佳参数可能不同
   - 参数调优应基于量化评估

5. 检索质量评估
   - 用量化指标替代主观感受
   - 来源命中率和关键词覆盖率是简单有效的指标
   - 生产环境推荐使用RAGAS等专业评估框架
   - 评估驱动的迭代优化是关键方法

下一章预告：RAG问答链 - 如何将检索和生成串联成完整的问答系统
""")

    logger.info("第4章演示完成")


if __name__ == '__main__':
    main()
