#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1章：文档加载与预处理

学习目标：
1. 理解RAG技术为什么需要文档加载
2. 掌握LangChain的Document Loader使用方法
3. 理解Document对象的结构和元数据管理

核心问题：为什么RAG需要文档加载？
- LLM的知识来自训练数据，有截止日期 -> 无法获取最新信息
- LLM不知道你的私有数据 -> 企业文档、个人笔记等
- RAG通过加载外部文档 -> 为LLM提供实时、专属的知识来源
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('chapter1_document_loader')


def demo_rag_overview():
    """
    RAG技术概述：解释为什么需要RAG

    知识点：RAG（检索增强生成）

    是什么？
    RAG是一种将信息检索与文本生成相结合的技术，让LLM基于外部知识库生成回答。

    为什么？
    LLM存在三个根本性限制：
    1. 知识截止日期 -> 训练数据有时效性，无法回答最新问题
    2. 缺乏私有知识 -> 不知道企业内部文档、个人数据
    3. 幻觉问题 -> 可能生成看似合理但实际错误的内容

    追问：为什么不直接微调模型来注入新知识？
    - 微调成本极高 -> 需要大量GPU资源和训练时间
    - 微调后知识仍然固定 -> 每次更新都需要重新训练
    - RAG只需更新文档 -> 知识更新成本几乎为零
    - RAG的回答有据可查 -> 可以追溯到源文档，微调无法做到
    """
    print("=" * 60)
    print("1. RAG技术概述")
    print("=" * 60)

    print("""
RAG（Retrieval-Augmented Generation）检索增强生成

核心思想：先检索，再生成

传统LLM的工作方式：
  用户提问 -> LLM直接回答（基于训练数据）

RAG的工作方式：
  用户提问 -> 检索相关文档 -> 将文档作为上下文 -> LLM基于上下文回答

为什么RAG比直接让LLM回答更好？

  问题1：知识时效性
    LLM的训练数据有截止日期（如2024年1月）
    -> 无法回答"2024年6月发生了什么"
    -> RAG通过加载最新文档解决

  问题2：私有数据
    LLM不知道你公司的内部文档
    -> 无法回答"我们公司的报销流程是什么"
    -> RAG通过加载企业知识库解决

  问题3：幻觉问题
    LLM可能编造看似合理的错误答案
    -> 用户无法判断回答是否可靠
    -> RAG提供源文档引用，回答有据可查

追问：为什么不直接把所有文档塞给LLM？
  - LLM有上下文窗口限制（如8K、16K tokens）
  - 塞入过多无关文档会稀释关键信息
  - RAG只检索最相关的文档片段 -> 高效且精准
""")

    print("RAG的完整工作流程：")
    steps = [
        ("文档加载", "将各种格式的文件加载到系统中"),
        ("文本分割", "将长文档切分为适当大小的文本块"),
        ("向量化", "使用嵌入模型将文本块转换为向量"),
        ("存储", "将向量存入向量数据库"),
        ("检索", "根据用户问题检索最相关的文本块"),
        ("生成", "将检索结果作为上下文，让LLM生成回答"),
    ]
    for i, (step, desc) in enumerate(steps, 1):
        print(f"  步骤{i}: {step} - {desc}")

    print(f"\n本章聚焦于步骤1：文档加载 - 这是RAG的数据入口")
    print("没有文档加载，后续的分割、向量化、检索都无从谈起")

    logger.info("RAG技术概述演示完成")


def demo_text_loader():
    """
    使用TextLoader加载单个文本文件

    知识点：TextLoader

    是什么？
    LangChain提供的文本文件加载器，用于读取.txt等纯文本文件。

    为什么需要专门的Loader而不是直接open()读取？
    - Loader将文本封装为Document对象 -> 统一的数据结构
    - Document对象携带元数据 -> 记录文件来源、编码等信息
    - Loader处理编码问题 -> 自动检测和转换文件编码
    - 与LangChain生态无缝集成 -> 后续分割、向量化直接使用

    追问：为什么统一数据结构这么重要？
    - RAG需要处理多种格式（TXT、PDF、Markdown、网页等）
    - 如果每种格式返回不同的数据结构 -> 后续代码需要大量条件判断
    - 统一为Document对象 -> 后续处理逻辑与文件格式解耦
    """
    print("\n" + "=" * 60)
    print("2. 使用TextLoader加载单个文本文件")
    print("=" * 60)

    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "ai_basics.txt")

    print(f"\n加载文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        print("请确保 data/ai_basics.txt 文件存在")
        logger.error(f"文件不存在: {file_path}")
        return

    try:
        loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()

        print(f"\n加载成功!")
        print(f"  文档数量: {len(documents)}")
        print(f"  文档类型: {type(documents[0]).__name__}")

        doc = documents[0]

        content_preview = doc.page_content[:200]
        print(f"\n  内容预览（前200字符）:")
        print(f"  {content_preview}...")

        print(f"\n  元数据: {doc.metadata}")

        print(f"\n  内容总长度: {len(doc.page_content)} 字符")

        logger.info(f"TextLoader加载成功: {file_path}, 内容长度: {len(doc.page_content)}")

    except Exception as e:
        print(f"[错误] 加载文件失败: {e}")
        logger.error(f"TextLoader加载失败: {e}", exc_info=True)

    print("\n--- TextLoader 要点总结 ---")
    print("1. TextLoader是最简单的Loader，适合纯文本文件")
    print("2. 建议显式指定encoding='utf-8'，避免编码问题")
    print("3. load()方法返回Document对象列表（即使只有一个文件）")
    print("4. Document对象包含page_content（文本内容）和metadata（元数据）")


def demo_directory_loader():
    """
    使用DirectoryLoader批量加载目录

    知识点：DirectoryLoader

    是什么？
    LangChain提供的目录加载器，可以批量加载一个目录下的所有文件。

    为什么需要DirectoryLoader而不是循环调用TextLoader？
    - 一次调用加载整个目录 -> 代码更简洁
    - 自动过滤非目标文件 -> 通过glob参数控制文件类型
    - 支持静默错误处理 -> show_progress和silent_errors参数
    - 实际场景中知识库通常有大量文档 -> 批量加载是刚需

    追问：为什么glob参数默认是"**/*"？
    - 默认加载所有文件 -> 最通用的行为
    - 实际使用时通常指定"*.txt"或"*.pdf" -> 只加载目标格式
    - 避免加载无关文件（如系统文件、临时文件）-> 提高加载效率
    """
    print("\n" + "=" * 60)
    print("3. 使用DirectoryLoader批量加载目录")
    print("=" * 60)

    from langchain_community.document_loaders import TextLoader, DirectoryLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print(f"\n加载目录: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"[错误] 目录不存在: {data_dir}")
        print("请确保 data/ 目录存在")
        logger.error(f"目录不存在: {data_dir}")
        return

    try:
        loader = DirectoryLoader(
            path=data_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            silent_errors=True,
        )

        print(f"\nDirectoryLoader 配置:")
        print(f"  路径: {data_dir}")
        print(f"  文件匹配模式: *.txt")
        print(f"  加载器类型: TextLoader")
        print(f"  显示进度: True")
        print(f"  静默错误: True")

        documents = loader.load()

        print(f"\n加载成功!")
        print(f"  总文档数量: {len(documents)}")

        print(f"\n各文档概览:")
        print(f"  {'序号':<4} {'来源文件':<25} {'内容长度':<12} {'前50字符'}")
        print(f"  {'-'*4} {'-'*25} {'-'*12} {'-'*30}")

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知")
            source_name = os.path.basename(source)
            content_len = len(doc.page_content)
            preview = doc.page_content[:50].replace("\n", " ")
            print(f"  {i:<4} {source_name:<25} {content_len:<12} {preview}...")

        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"\n  总字符数: {total_chars}")

        logger.info(f"DirectoryLoader加载成功: {len(documents)}个文档, 总计{total_chars}字符")

    except Exception as e:
        print(f"[错误] 加载目录失败: {e}")
        logger.error(f"DirectoryLoader加载失败: {e}", exc_info=True)

    print("\n--- DirectoryLoader 要点总结 ---")
    print("1. DirectoryLoader批量加载目录下的文件，适合知识库场景")
    print("2. glob参数控制文件类型，如'*.txt'只加载文本文件")
    print("3. loader_cls指定使用的Loader类型，默认是TextLoader")
    print("4. loader_kwargs传递给内部Loader的额外参数（如编码）")
    print("5. silent_errors=True跳过无法加载的文件，避免整个流程中断")


def demo_document_structure():
    """
    展示Document对象的结构

    知识点：Document对象

    是什么？
    LangChain中所有文档加载器的统一输出格式，包含page_content和metadata两个字段。

    为什么用Document对象而不是纯字符串？
    - 纯字符串丢失来源信息 -> 无法追溯答案出自哪个文件
    - metadata记录来源、页码、作者等 -> 支持溯源和过滤
    - 统一接口 -> 不同格式的文档（PDF、网页、数据库）都转为Document
    - 与LangChain生态兼容 -> 分割器、向量库、检索器都接受Document

    追问：为什么metadata这么重要？
    - 检索时可以按来源过滤 -> 只在特定文档中搜索
    - 回答时可以引用来源 -> "根据《公司手册》第3章..."
    - 调试时可以追踪问题 -> 知道哪个文档导致了错误回答
    - metadata是RAG可解释性的基础 -> 没有它就无法验证回答的可靠性
    """
    print("\n" + "=" * 60)
    print("4. Document对象的结构")
    print("=" * 60)

    from langchain_core.documents import Document

    print("\n--- 4.1 从文件加载的Document ---")
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "python_learning.txt")

    if os.path.exists(file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        doc = docs[0]

        print(f"\nDocument 对象结构:")
        print(f"  类型: {type(doc).__name__}")
        print(f"  模块: {type(doc).__module__}")

        print(f"\n  page_content (文本内容):")
        print(f"    类型: {type(doc.page_content).__name__}")
        print(f"    长度: {len(doc.page_content)} 字符")
        print(f"    前100字符: {doc.page_content[:100]}...")

        print(f"\n  metadata (元数据):")
        print(f"    类型: {type(doc.metadata).__name__}")
        print(f"    内容: {doc.metadata}")

        print(f"\n  metadata中的字段说明:")
        print(f"    source: 文件路径 - 记录文档的来源位置")
        print(f"    （不同Loader会添加不同的metadata字段）")
    else:
        print(f"  [跳过] 文件不存在: {file_path}")

    print("\n--- 4.2 手动创建Document对象 ---")

    custom_doc = Document(
        page_content="这是一个手动创建的文档，用于演示Document对象的结构。",
        metadata={
            "source": "手动创建",
            "author": "教程作者",
            "category": "演示",
            "importance": "高",
            "created_at": "2026-05-14",
        }
    )

    print(f"\n手动创建的Document:")
    print(f"  page_content: {custom_doc.page_content}")
    print(f"  metadata: {custom_doc.metadata}")

    print("\n--- 4.3 Document对象的常用操作 ---")

    print(f"\n  获取内容长度: len(doc.page_content) = {len(custom_doc.page_content)}")
    print(f"  获取特定元数据: doc.metadata['author'] = {custom_doc.metadata['author']}")
    print(f"  检查元数据是否存在: 'category' in doc.metadata = {'category' in custom_doc.metadata}")
    print(f"  获取不存在的元数据（带默认值）: doc.metadata.get('page', 'N/A') = {custom_doc.metadata.get('page', 'N/A')}")

    print("\n--- Document 要点总结 ---")
    print("1. Document是LangChain中文档的统一数据结构")
    print("2. page_content: 字符串类型，存储文档的文本内容")
    print("3. metadata: 字典类型，存储文档的元信息（来源、作者等）")
    print("4. 可以手动创建Document对象，也可以通过Loader自动创建")
    print("5. metadata是RAG可解释性的关键 - 让回答可溯源")

    logger.info("Document结构演示完成")


def demo_metadata_management():
    """
    演示如何添加和管理元数据

    知识点：元数据管理

    是什么？
    在Document对象上添加、修改、删除metadata字段，为后续检索提供过滤条件。

    为什么需要主动管理metadata？
    - Loader自动添加的metadata通常只有source -> 信息不够丰富
    - 实际应用中需要更多维度 -> 分类、标签、时间、权限等
    - 检索时可以按metadata过滤 -> 提高检索精度
    - 不同来源的文档需要统一metadata格式 -> 便于管理

    追问：为什么检索时按metadata过滤能提高精度？
    - 不加过滤：在所有文档中搜索 -> 可能返回结果时，需要按条件过滤 -> metadata是过滤的依据
    - 好的metadata设计 = 好的检索质量 -> 直接影响RAG的效果

    追问：metadata和page_content有什么本质区别？
    - page_content是"非结构化"的 -> 需要向量化后才能检索
    - metadata是"结构化"的 -> 可以精确匹配、范围查询
    - 检索时可以同时利用两者 -> 向量相似度 + 元数据过滤
    - 例如：先按"部门=财务部"过滤，再在结果中做语义搜索
    """
    print("\n" + "=" * 60)
    print("5. 元数据管理")
    print("=" * 60)

    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("\n--- 5.1 Loader自动生成的metadata ---")

    file_path = os.path.join(data_dir, "ai_basics.txt")
    if os.path.exists(file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        doc = docs[0]

        print(f"\nTextLoader自动生成的metadata:")
        for key, value in doc.metadata.items():
            print(f"  {key}: {value}")

        print(f"\n可以看到，TextLoader只自动添加了'source'字段")
        print(f"这在实际应用中通常是不够的")
    else:
        print(f"  [跳过] 文件不存在: {file_path}")

    print("\n--- 5.2 手动添加metadata ---")

    file_path = os.path.join(data_dir, "python_learning.txt")
    if os.path.exists(file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        doc = docs[0]

        doc.metadata["category"] = "编程语言"
        doc.metadata["topic"] = "Python"
        doc.metadata["difficulty"] = "入门"
        doc.metadata["language"] = "中文"
        doc.metadata["tags"] = ["Python", "编程", "AI"]

        print(f"\n添加metadata后的Document:")
        print(f"  page_content长度: {len(doc.page_content)} 字符")
        print(f"  metadata:")
        for key, value in doc.metadata.items():
            print(f"    {key}: {value}")
    else:
        print(f"  [跳过] 文件不存在: {file_path}")

    print("\n--- 5.3 批量添加metadata ---")

    file_category_map = {
        "ai_basics.txt": {"category": "人工智能", "topic": "AI基础", "difficulty": "入门"},
        "python_learning.txt": {"category": "编程语言", "topic": "Python", "difficulty": "入门"},
        "langchain_intro.txt": {"category": "AI框架", "topic": "LangChain", "difficulty": "中级"},
    }

    from langchain_community.document_loaders import DirectoryLoader

    if os.path.exists(data_dir):
        loader = DirectoryLoader(
            path=data_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        documents = loader.load()

        print(f"\n为{len(documents)}个文档批量添加metadata:")

        for doc in documents:
            source = doc.metadata.get("source", "")
            filename = os.path.basename(source)

            if filename in file_category_map:
                for key, value in file_category_map[filename].items():
                    doc.metadata[key] = value

                print(f"\n  文件: {filename}")
                print(f"  metadata:")
                for key, value in doc.metadata.items():
                    print(f"    {key}: {value}")

        print("\n--- 5.4 利用metadata进行过滤 ---")

        print(f"\n示例：筛选'入门'难度的文档")
        beginner_docs = [doc for doc in documents if doc.metadata.get("difficulty") == "入门"]
        print(f"  匹配数量: {len(beginner_docs)}")
        for doc in beginner_docs:
            filename = os.path.basename(doc.metadata.get("source", ""))
            topic = doc.metadata.get("topic", "未知")
            print(f"  - {filename} (主题: {topic})")

        print(f"\n示例：筛选'AI'相关分类的文档")
        ai_docs = [doc for doc in documents
                   if "AI" in doc.metadata.get("category", "")
                   or "AI" in doc.metadata.get("topic", "")]
        print(f"  匹配数量: {len(ai_docs)}")
        for doc in ai_docs:
            filename = os.path.basename(doc.metadata.get("source", ""))
            category = doc.metadata.get("category", "未知")
            print(f"  - {filename} (分类: {category})")
    else:
        print(f"  [跳过] 目录不存在: {data_dir}")

    print("\n--- 5.5 metadata设计的最佳实践 ---")
    print("""
  1. 保持一致性: 同类文档使用相同的metadata字段名和值
     -> 方便统一过滤，避免"部门"vs"department"的混乱

  2. 使用枚举值: difficulty只用"入门/中级/高级"，不用自由文本
     -> 精确匹配，避免"初级"vs"入门"的遗漏

  3. 记录时间信息: 添加created_at、updated_at字段
     -> 支持按时间范围检索，优先返回最新文档

  4. 添加来源标识: source_type区分"内部文档/外部资料/用户生成"
     -> 控制知识库的可信度层级

  5. 避免过大: metadata不要存储大段文本，只存关键属性
     -> metadata用于精确过滤，不是存储内容的场所
""")

    logger.info("元数据管理演示完成")


def main():
    """
    主函数：依次运行所有演示

    执行顺序的设计逻辑：
    1. RAG概述 -> 先理解"为什么"，再学"怎么做"
    2. TextLoader -> 最简单的加载方式，入门首选
    3. DirectoryLoader -> 实际场景中的批量加载
    4. Document结构 -> 理解加载后的数据结构
    5. 元数据管理 -> 深入理解metadata的重要性

    为什么按这个顺序？
    - 从宏观到微观 -> 先建立全局认知，再深入细节
    - 从简单到复杂 -> TextLoader比DirectoryLoader更简单
    - 从使用到理解 -> 先会用，再理解内部结构
    """
    print("=" * 60)
    print("第1章：文档加载与预处理")
    print("=" * 60)

    logger.info("开始第1章演示")

    demos = [
        ("RAG技术概述", demo_rag_overview),
        ("TextLoader - 加载单个文件", demo_text_loader),
        ("DirectoryLoader - 批量加载目录", demo_directory_loader),
        ("Document对象结构", demo_document_structure),
        ("元数据管理", demo_metadata_management),
    ]

    for name, demo_func in demos:
        try:
            logger.info(f"开始演示: {name}")
            demo_func()
        except Exception as e:
            print(f"\n[错误] 演示'{name}'执行失败: {e}")
            logger.error(f"演示'{name}'执行失败: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("第1章学习总结")
    print("=" * 60)
    print("""
核心要点回顾：

1. RAG为什么需要文档加载？
   - LLM知识有时效性和范围限制
   - 文档加载是RAG获取外部知识的第一步
   - 没有文档加载，后续所有步骤都无法进行

2. TextLoader vs DirectoryLoader
   - TextLoader: 加载单个文件，简单直接
   - DirectoryLoader: 批量加载目录，适合知识库场景
   - 实际项目中通常使用DirectoryLoader

3. Document对象 = page_content + metadata
   - page_content: 文本内容（非结构化）
   - metadata: 元信息（结构化）
   - 两者配合实现高效、可解释的检索

4. metadata管理的重要性
   - 支持精确过滤，提高检索效率
   - 支持来源追溯，提高回答可信度
   - 好的metadata设计直接影响RAG效果

下一章预告：文本分割 - 如何将长文档切分为适合检索的文本块
""")

    logger.info("第1章演示完成")


if __name__ == '__main__':
    main()
