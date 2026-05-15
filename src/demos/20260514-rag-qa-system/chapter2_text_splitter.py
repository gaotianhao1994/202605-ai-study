#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章：文本分割策略

学习目标：
1. 理解为什么需要文本分割
2. 掌握RecursiveCharacterTextSplitter的使用
3. 理解chunk_size和chunk_overlap参数的影响
4. 了解中文文本分割的特殊考虑

核心问题：为什么需要文本分割？
- LLM有上下文窗口限制 -> 无法一次性处理整篇长文档
- 向量嵌入有最佳长度 -> 太长则语义模糊，太短则信息不足
- 检索需要精确匹配 -> 小块文本比整篇文档更容易匹配到相关内容
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('chapter2_text_splitter')


def demo_why_split():
    """
    解释为什么需要文本分割，展示不分割的问题

    知识点：文本分割的必要性

    是什么？
    将长文档切分为多个较小的文本块（chunk），每个块独立进行向量化存储。

    为什么需要文本分割？
    - LLM有上下文窗口限制 -> 无法一次性处理整篇长文档
    - 向量嵌入有最佳长度 -> 太长的文本生成的向量会"稀释"关键语义
    - 检索需要精确匹配 -> 小块文本比整篇文档更容易匹配到相关内容
    - 存储效率 -> 小块文本可以只检索需要的部分，避免加载整个文档

    追问：为什么不把整篇文档作为一个整体向量化？
    - 向量是对文本的"压缩表示" -> 文本越长，压缩丢失的信息越多
    - 假设一篇5000字的文章涵盖5个主题 -> 向量只能捕捉"平均语义"
    - 用户问其中一个主题 -> 整篇文档的向量与问题的相似度可能不高
    - 切成5个1000字的块 -> 每个块的向量更聚焦，检索更精准

    追问：为什么不把文本切成单个字或词？
    - 单个字/词没有完整语义 -> "学"字无法表达"机器学习"的含义
    - 向量化需要足够的上下文 -> 太短的文本生成的向量缺乏语义信息
    - 检索结果无法提供足够的上下文给LLM -> LLM无法基于零散的词生成回答
    - 所以需要找到"不大不小"的合适粒度 -> 这就是chunk_size的意义
    """
    print("=" * 60)
    print("1. 为什么需要文本分割")
    print("=" * 60)

    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "ai_basics.txt")

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        logger.error(f"文件不存在: {file_path}")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    full_text = docs[0].page_content

    print(f"\n--- 不分割的问题演示 ---")
    print(f"\n原始文档信息:")
    print(f"  文件: ai_basics.txt")
    print(f"  总长度: {len(full_text)} 字符")

    paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
    print(f"  段落数: {len(paragraphs)}")

    topics = []
    for p in paragraphs:
        if p.strip().startswith(("一", "二", "三", "四", "五")):
            topic_name = p.strip().split("\n")[0]
            topics.append(topic_name)
    print(f"  涵盖主题: {topics}")

    print(f"\n不分割带来的问题:")
    print(f"  问题1: 语义稀释")
    print(f"    - 整篇文档涵盖{len(topics)}个不同主题")
    print(f"    - 向量化后，向量反映的是'平均语义'，无法精确代表任何一个主题")
    print(f"    - 用户问'什么是RAG'时，整篇文档的向量与问题的相似度不够高")

    print(f"\n  问题2: 检索不精确")
    print(f"    - 即使检索到了整篇文档，用户只需要其中一小部分")
    print(f"    - 把整篇文档传给LLM，浪费了上下文窗口")

    print(f"\n  问题3: 超出限制")
    print(f"    - 如果文档更长（如一本书），可能超出嵌入模型的输入限制")
    print(f"    - 嵌入模型通常有最大输入长度（如512或8192 tokens）")

    print(f"\n--- 分割后的优势 ---")
    print(f"  优势1: 语义聚焦")
    print(f"    - 每个chunk只包含一个主题或段落")
    print(f"    - 向量更精确地表达该chunk的语义")
    print(f"    - 检索'什么是RAG'时，RAG相关的chunk相似度更高")

    print(f"  优势2: 检索精准")
    print(f"    - 只返回与问题相关的chunk，不返回无关内容")
    print(f"    - 节省上下文窗口，让LLM聚焦于相关信息")

    print(f"  优势3: 灵活组合")
    print(f"    - 可以从不同文档中检索相关chunk")
    print(f"    - 拼接多个chunk作为上下文，提供更全面的回答")

    print("\n--- 文本分割 要点总结 ---")
    print("1. 文本分割是RAG流程的关键步骤，连接文档加载和向量化")
    print("2. 不分割: 语义稀释、检索不精确、可能超出限制")
    print("3. 分割后: 语义聚焦、检索精准、灵活组合")
    print("4. 关键是找到合适的粒度: 不太大（语义模糊），不太小（信息不足）")

    logger.info("文本分割必要性演示完成")


def demo_recursive_splitter():
    """
    使用RecursiveCharacterTextSplitter，详细解释参数

    知识点：RecursiveCharacterTextSplitter

    是什么？
    LangChain中最常用的文本分割器，递归地尝试不同的分隔符来分割文本，
    尽量保持语义完整性。

    为什么RecursiveCharacterTextSplitter比CharacterTextSplitter更好？
    - CharacterTextSplitter只用一个分隔符 -> 容易在不合适的位置切断
    - RecursiveCharacterTextSplitter依次尝试多个分隔符 -> 优先在段落边界分割
    - 递归策略: 先尝试"\n\n"(段落) -> 再尝试"\n"(行) -> 再尝试" "(词) -> 最后按字符
    - 结果: 尽量在自然的语义边界处分割，保持每个chunk的语义完整性

    追问：为什么递归尝试分隔符比单次分割更好？
    - 单次分割: 只用一个分隔符，可能产生过长或过短的chunk
    - 递归分割: 先用大粒度分隔符，如果chunk仍然太大，再用小粒度分隔符
    - 类比: 切蛋糕 -> 先切大块，大块太大的再切小块 -> 每块大小更均匀
    - 这就是"Recursive"的含义: 递归地对过大的chunk继续分割
    """
    print("\n" + "=" * 60)
    print("2. RecursiveCharacterTextSplitter 详解")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "ai_basics.txt")

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        logger.error(f"文件不存在: {file_path}")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    print("\n--- 2.1 RecursiveCharacterTextSplitter 参数详解 ---")
    print("""
    RecursiveCharacterTextSplitter 的核心参数:

    chunk_size (int):
      每个文本块的最大字符数。
      为什么有最大限制？因为需要控制向量化的输入长度。
      设多大合适？通常500-1000字符，取决于嵌入模型和文档类型。

    chunk_overlap (int):
      相邻文本块之间重叠的字符数。
      为什么需要重叠？避免关键信息恰好在分割点被切断。
      设多大合适？通常为chunk_size的10%-20%。

    separators (List[str]):
      分隔符列表，按优先级从高到低排列。
      为什么需要多个分隔符？不同文本的结构不同，需要灵活适配。
      默认值: ["\\n\\n", "\\n", " ", ""]
      含义: 优先按段落分割 -> 按行分割 -> 按空格分割 -> 按字符分割

    length_function (callable):
      计算文本长度的函数，默认为len。
      为什么可自定义？某些语言（如中文）可能需要用token数而非字符数。

    keep_separator (bool):
      是否在分割结果中保留分隔符，默认为True。
      为什么保留？分隔符（如句号）本身携带语义信息。
""")

    print("--- 2.2 基本使用示例 ---")

    chunk_size = 500
    chunk_overlap = 50
    separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    print(f"\n分割器配置:")
    print(f"  chunk_size: {chunk_size}")
    print(f"  chunk_overlap: {chunk_overlap}")
    print(f"  separators: {separators}")

    chunks = splitter.split_documents(docs)

    print(f"\n分割结果:")
    print(f"  原始文档数: {len(docs)}")
    print(f"  分割后chunk数: {len(chunks)}")

    print(f"\n各chunk详情:")
    print(f"  {'序号':<4} {'长度':<8} {'前60字符'}")
    print(f"  {'-'*4} {'-'*8} {'-'*40}")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.page_content[:60].replace("\n", "\\n")
        print(f"  {i:<4} {len(chunk.page_content):<8} {preview}...")

    print(f"\n  平均chunk长度: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")
    print(f"  最短chunk: {min(len(c.page_content) for c in chunks)} 字符")
    print(f"  最长chunk: {max(len(c.page_content) for c in chunks)} 字符")

    print("\n--- 2.3 分割器的工作原理演示 ---")

    sample_text = """一、什么是人工智能

人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统。

二、机器学习

机器学习是人工智能的核心子领域，它让计算机能够从数据中自动学习规律。

三、深度学习

深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次化表示。"""

    print(f"\n原始文本 ({len(sample_text)} 字符):")
    print(sample_text)

    print(f"\n递归分割过程:")
    print(f"  第1步: 尝试用 '\\n\\n'(段落) 分割")
    paragraphs = sample_text.split("\n\n")
    for i, p in enumerate(paragraphs):
        print(f"    段落{i+1}: {len(p)} 字符 - {p[:30]}...")

    print(f"\n  第2步: 检查每个段落是否超过chunk_size")
    print(f"    如果段落 <= chunk_size -> 保留为一个chunk")
    print(f"    如果段落 > chunk_size -> 用下一个分隔符'\\n'继续分割")
    print(f"    如果仍然太大 -> 用空格分割 -> 最后按字符分割")
    print(f"    这就是'递归'的含义: 逐级降级分隔符，直到chunk大小合适")

    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    )
    small_chunks = small_splitter.split_text(sample_text)

    print(f"\n用小chunk_size(100)分割的结果:")
    for i, chunk in enumerate(small_chunks, 1):
        print(f"  chunk {i} ({len(chunk)} 字符): {chunk[:50]}...")

    print("\n--- RecursiveCharacterTextSplitter 要点总结 ---")
    print("1. 递归尝试多个分隔符，优先在语义边界分割")
    print("2. chunk_size控制每个块的最大大小")
    print("3. chunk_overlap避免关键信息在分割点丢失")
    print("4. separators决定分割的优先级顺序")
    print("5. 是LangChain中最推荐的文本分割器")

    logger.info(f"RecursiveCharacterTextSplitter演示完成, 分割出{len(chunks)}个chunk")


def demo_chunk_size_comparison():
    """
    对比不同chunk_size的效果（200 vs 500 vs 1000）

    知识点：chunk_size参数的影响

    是什么？
    chunk_size决定了每个文本块的最大字符数，直接影响分割粒度。

    为什么chunk_size的选择很重要？
    - 太小(如50): 每个chunk信息不足，向量缺乏语义，检索结果碎片化
    - 太大(如5000): 每个chunk包含过多信息，向量语义模糊，检索不精确
    - 合适(如500-1000): 每个chunk包含足够且聚焦的信息，检索效果最佳

    追问：为什么chunk_size没有"万能最优值"？
    - 不同文档类型适合不同粒度 -> 技术文档段落短，小说段落长
    - 不同嵌入模型有不同的最佳输入长度 -> 有些模型512 tokens最优
    - 不同应用场景需求不同 -> 精确问答需要小chunk，概述需要大chunk
    - 所以chunk_size需要根据具体场景实验调优 -> 不存在一刀切的最优值
    """
    print("\n" + "=" * 60)
    print("3. chunk_size 对比实验")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "langchain_intro.txt")

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        logger.error(f"文件不存在: {file_path}")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    total_chars = len(docs[0].page_content)

    print(f"\n原始文档: langchain_intro.txt ({total_chars} 字符)")
    print(f"\n实验设计: 使用相同的chunk_overlap(比例)，不同的chunk_size")

    chunk_sizes = [200, 500, 1000]

    print(f"\n{'chunk_size':<12} {'chunk数':<10} {'平均长度':<10} {'最短':<8} {'最长':<8} {'overlap':<8}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    all_results = {}

    for size in chunk_sizes:
        overlap = int(size * 0.1)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        lengths = [len(c.page_content) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0

        all_results[size] = {
            "chunks": chunks,
            "overlap": overlap,
            "avg_len": avg_len,
            "min_len": min_len,
            "max_len": max_len,
        }

        print(f"{size:<12} {len(chunks):<10} {avg_len:<10.0f} {min_len:<8} {max_len:<8} {overlap:<8}")

    print("\n--- 详细对比分析 ---")

    for size in chunk_sizes:
        result = all_results[size]
        chunks = result["chunks"]
        overlap = result["overlap"]

        print(f"\nchunk_size = {size}, chunk_overlap = {overlap}")
        print(f"  分割出 {len(chunks)} 个chunk")

        if len(chunks) <= 6:
            for i, chunk in enumerate(chunks, 1):
                preview = chunk.page_content[:80].replace("\n", "\\n")
                print(f"    chunk {i} ({len(chunk.page_content)} 字符): {preview}...")
        else:
            for i in [0, 1]:
                preview = chunks[i].page_content[:80].replace("\n", "\\n")
                print(f"    chunk {i+1} ({len(chunks[i].page_content)} 字符): {preview}...")
            print(f"    ... (省略 {len(chunks) - 4} 个chunk) ...")
            for i in [-2, -1]:
                preview = chunks[i].page_content[:80].replace("\n", "\\n")
                print(f"    chunk {len(chunks)+i+1} ({len(chunks[i].page_content)} 字符): {preview}...")

    print("\n--- chunk_size 选择建议 ---")
    print("""
    chunk_size = 200 (小粒度):
      优点: 检索非常精确，每个chunk语义聚焦
      缺点: 上下文可能不足，LLM缺乏足够信息生成完整回答
      适合: 精确问答、事实查找（如"RAG的完整工作流程是什么"）

    chunk_size = 500 (中等粒度):
      优点: 平衡了精确性和上下文完整性
      缺点: 某些复杂主题可能被截断
      适合: 通用场景，大多数RAG应用的默认选择

    chunk_size = 1000 (大粒度):
      优点: 上下文丰富，LLM有足够信息生成完整回答
      缺点: 语义可能不够聚焦，检索可能返回部分无关内容
      适合: 概述性问答、需要丰富上下文的场景

    选择原则:
      1. 从500开始，根据效果调整
      2. 问答场景偏小(300-500)，概述场景偏大(800-1500)
      3. 配合chunk_overlap使用，通常为chunk_size的10%-20%
""")

    logger.info("chunk_size对比实验完成")


def demo_overlap_effect():
    """
    演示chunk_overlap的作用，有overlap vs 无overlap

    知识点：chunk_overlap参数

    是什么？
    chunk_overlap是相邻文本块之间重叠的字符数，确保分割点附近的信息不会丢失。

    为什么需要overlap？
    - 分割可能在句子中间切断 -> 关键信息被拆成两半
    - 没有overlap: 前一个chunk的末尾和后一个chunk的开头各丢失一半信息
    - 有overlap: 两个chunk都包含分割点附近的信息 -> 信息不会断裂

    追问：为什么overlap通常设为chunk_size的10%-20%？
    - 太小(如0): 无法有效避免信息断裂，分割点附近的信息可能丢失
    - 太大(如50%): 大量重复内容，浪费存储和上下文窗口
    - 10%-20%: 既能保证信息连续性，又不会产生过多冗余
    - 这是一个经验值，实际效果需要根据文档类型调整

    追问：overlap和chunk_size的关系是什么？
    - overlap是"保险" -> 防止信息在分割点丢失
    - chunk_size是"容量" -> 决定每个chunk能承载多少信息
    - overlap不能大于chunk_size -> 否则没有意义（一个chunk都放不下）
    - overlap越大，相邻chunk的重复越多 -> 检索时可能返回重复内容
    """
    print("\n" + "=" * 60)
    print("4. chunk_overlap 效果对比")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sample_text = (
        "RAG检索增强生成技术通过将信息检索与文本生成相结合，"
        "有效解决了大语言模型的知识时效性和幻觉问题。"
        "在RAG系统中，文本分割是最关键的预处理步骤之一，"
        "它直接影响后续向量化和检索的质量。"
        "RecursiveCharacterTextSplitter是LangChain中最常用的分割器，"
        "它通过递归尝试不同分隔符来保持语义完整性。"
        "chunk_size参数控制每个文本块的最大字符数，"
        "chunk_overlap参数控制相邻文本块之间的重叠字符数。"
        "合理的参数配置需要根据文档类型和应用场景进行实验调优。"
    )

    print(f"\n示例文本 ({len(sample_text)} 字符):")
    print(f"  {sample_text[:80]}...")

    chunk_size = 80

    print(f"\n--- 对比实验: chunk_size={chunk_size} ---")
    print(f"  使用默认分隔符（不含中文标点），迫使分割器在句子内部切割")

    no_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )

    with_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    )

    no_overlap_chunks = no_overlap_splitter.split_text(sample_text)
    with_overlap_chunks = with_overlap_splitter.split_text(sample_text)

    print(f"\n--- 无overlap (chunk_overlap=0) ---")
    print(f"  分割出 {len(no_overlap_chunks)} 个chunk")
    for i, chunk in enumerate(no_overlap_chunks, 1):
        print(f"\n  chunk {i} ({len(chunk)} 字符):")
        print(f"    {chunk}")

    print(f"\n--- 有overlap (chunk_overlap=20) ---")
    print(f"  分割出 {len(with_overlap_chunks)} 个chunk")
    for i, chunk in enumerate(with_overlap_chunks, 1):
        print(f"\n  chunk {i} ({len(chunk)} 字符):")
        print(f"    {chunk}")

    print(f"\n--- overlap效果分析 ---")
    print(f"  无overlap: {len(no_overlap_chunks)} 个chunk")
    print(f"  有overlap: {len(with_overlap_chunks)} 个chunk")

    print(f"\n  关键区别:")
    if len(no_overlap_chunks) >= 2 and len(with_overlap_chunks) >= 2:
        print(f"    观察相邻chunk的衔接处:")
        no_tail = no_overlap_chunks[0][-20:] if len(no_overlap_chunks[0]) >= 20 else no_overlap_chunks[0]
        with_head = with_overlap_chunks[1][:30] if len(with_overlap_chunks[1]) >= 30 else with_overlap_chunks[1]
        print(f"    无overlap chunk1的结尾: ...{no_tail}")
        print(f"    有overlap chunk2的开头: {with_head}...")
        print(f"    有overlap时，chunk2的开头包含chunk1结尾的部分内容")
        print(f"    -> overlap确保跨chunk的信息不会断裂")
    else:
        print(f"    无overlap时，分割点附近的信息可能断裂")
        print(f"    有overlap时，相邻chunk共享部分内容，信息连续性更好")

    print(f"\n  实际影响:")
    print(f"    假设用户问'RAG的chunk_overlap参数有什么作用'")
    print(f"    无overlap: 相关信息可能被拆在两个chunk中，检索可能只命中一个")
    print(f"    有overlap: 某个chunk完整包含相关信息，检索命中率更高")
    print(f"    -> 有overlap时，检索到包含完整信息的chunk概率更高")

    print("\n--- overlap 要点总结 ---")
    print("1. overlap确保分割点附近的信息不会断裂")
    print("2. 推荐值为chunk_size的10%-20%")
    print("3. overlap太小: 信息可能断裂; overlap太大: 冗余过多")
    print("4. 实际效果需要根据文档类型和检索需求调整")

    logger.info("overlap效果对比演示完成")


def demo_chinese_splitting():
    """
    中文文本分割的特殊考虑（中文分隔符设置）

    知识点：中文文本分割

    是什么？
    针对中文文本特点，调整分隔符列表以更好地保持中文语义完整性。

    为什么中文分割需要特殊处理？
    - 默认分隔符["\\n\\n", "\\n", " ", ""]是为英文设计的
    - 中文不使用空格分词 -> " "作为分隔符对中文几乎无效
    - 中文有自己的句子结束标志: "。", "！", "？"，"；"
    - 中文段落之间可能只用换行分隔，不用空行 -> "\\n\\n"可能不够

    追问：为什么中文不用空格分词会影响分割效果？
    - 英文: "Machine learning is AI." -> 空格可以分割为词
    - 中文: "机器学习是人工智能。" -> 没有空格，无法按词分割
    - 默认分隔符跳过" "后直接按字符分割 -> 可能在字中间切断
    - 加入中文标点作为分隔符 -> 优先在句子边界分割，保持语义完整

    追问：为什么中文分隔符要设置多个级别？
    - "。"是句号 -> 最自然的句子边界，优先级最高
    - "！"和"？"是感叹号和问号 -> 也是句子边界
    - "；"是分号 -> 分句边界，粒度比句号小
    - " "是空格 -> 中文中偶尔出现（如中英混排），优先级低
    - ""是空字符串 -> 最后的兜底，按字符分割
    - 多级别确保: 优先在自然边界分割，实在不行才按字符分割
    """
    print("\n" + "=" * 60)
    print("5. 中文文本分割的特殊考虑")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("\n--- 5.1 默认分隔符 vs 中文分隔符 ---")

    print(f"\n默认分隔符 (为英文设计):")
    print(f'  ["\\n\\n", "\\n", " ", ""]')
    print(f"  问题: 中文没有空格分词，' '分隔符对中文几乎无效")
    print(f"  结果: 跳过空格后直接按字符分割，可能在句子中间切断")

    chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    print(f"\n中文分隔符 (为中文优化):")
    print(f'  {chinese_separators}')
    print(f"  改进: 加入中文标点（。！？；）作为分隔符")
    print(f"  效果: 优先在中文句子边界分割，保持语义完整")

    print("\n--- 5.2 连续中文文本对比实验 ---")
    print("\n  当文本没有明显的段落分隔（只有句号等标点）时，差异最明显。")

    continuous_chinese = (
        "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统。"
        "机器学习是人工智能的核心子领域，它让计算机能够从数据中自动学习规律。"
        "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次化表示。"
        "自然语言处理是AI中研究计算机与人类语言交互的领域。"
        "大语言模型是通过在海量文本数据上训练的深度学习模型，能够理解和生成人类语言。"
        "检索增强生成是一种将信息检索与文本生成相结合的技术，解决了大语言模型的知识时效性和幻觉问题。"
    )

    print(f"\n  示例文本 ({len(continuous_chinese)} 字符，无段落分隔):")
    print(f"  {continuous_chinese[:60]}...")

    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
    )

    chinese_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=chinese_separators,
    )

    default_chunks = default_splitter.split_text(continuous_chinese)
    chinese_chunks = chinese_splitter.split_text(continuous_chinese)

    print(f"\n  默认分隔符 (chunk_size=100):")
    for i, chunk in enumerate(default_chunks, 1):
        ends_ok = chunk.rstrip().endswith(("。", "！", "？", "；"))
        status = "完整" if ends_ok else "截断"
        print(f"    chunk {i} ({len(chunk)} 字符) [{status}]: {chunk}")

    print(f"\n  中文分隔符 (chunk_size=100):")
    for i, chunk in enumerate(chinese_chunks, 1):
        ends_ok = chunk.rstrip().endswith(("。", "！", "？", "；"))
        status = "完整" if ends_ok else "截断"
        print(f"    chunk {i} ({len(chunk)} 字符) [{status}]: {chunk}")

    default_truncated = sum(
        1 for c in default_chunks
        if not c.rstrip().endswith(("。", "！", "？", "；"))
    )
    chinese_truncated = sum(
        1 for c in chinese_chunks
        if not c.rstrip().endswith(("。", "！", "？", "；"))
    )

    print(f"\n  句子完整性对比:")
    print(f"    默认分隔符: {len(default_chunks)} 个chunk中有 {default_truncated} 个句子被截断 "
          f"({default_truncated/len(default_chunks)*100:.1f}%)")
    print(f"    中文分隔符: {len(chinese_chunks)} 个chunk中有 {chinese_truncated} 个句子被截断 "
          f"({chinese_truncated/len(chinese_chunks)*100:.1f}%)")

    print("\n--- 5.3 文件级分割对比 ---")

    file_path = os.path.join(data_dir, "python_learning.txt")
    if not os.path.exists(file_path):
        print(f"  [跳过] 文件不存在: {file_path}")
        logger.error(f"文件不存在: {file_path}")
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        file_default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", " ", ""],
        )

        file_chinese_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=chinese_separators,
        )

        file_default_chunks = file_default_splitter.split_documents(docs)
        file_chinese_chunks = file_chinese_splitter.split_documents(docs)

        print(f"\n  文件: python_learning.txt, chunk_size=300, chunk_overlap=30")
        print(f"\n  {'分割器':<16} {'chunk数':<10} {'平均长度':<10} {'最短':<8} {'最长':<8}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

        for name, chunks in [("默认分隔符", file_default_chunks), ("中文分隔符", file_chinese_chunks)]:
            lengths = [len(c.page_content) for c in chunks]
            avg = sum(lengths) / len(lengths) if lengths else 0
            mn = min(lengths) if lengths else 0
            mx = max(lengths) if lengths else 0
            print(f"  {name:<16} {len(chunks):<10} {avg:<10.0f} {mn:<8} {mx:<8}")

        print(f"\n  注意: 结构良好的文档（有明确段落分隔）两种分割器差异较小")
        print(f"  但中文分隔符在句子边界处仍然更精确，特别是当chunk_size较小时")

    print("\n--- 5.4 中文分割器创建的推荐方式 ---")
    print("""
    创建中文文本分割器的推荐配置:

    chinese_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\\n\\n", "\\n", "。", "！", "？", "；", " ", ""],
    )

    参数说明:
      chunk_size=500: 中文500字符约等于1-2个段落，语义较完整
      chunk_overlap=50: 重叠10%，防止句子在分割点断裂
      separators: 按优先级排列
        "\\n\\n" -> 段落边界（最优先）
        "\\n"   -> 行边界
        "。"    -> 句号（中文最常用的句子结束标志）
        "！"    -> 感叹号
        "？"    -> 问号
        "；"    -> 分号（分句边界）
        " "     -> 空格（中英混排时有用）
        ""      -> 兜底，按字符分割（最后手段）
""")

    print("--- 中文分割 要点总结 ---")
    print("1. 默认分隔符为英文设计，对中文效果不佳")
    print("2. 中文分隔符应加入'。！？；'等中文标点")
    print("3. 中文分隔符按优先级排列: 段落 > 行 > 句号 > 分号 > 空格 > 字符")
    print("4. 使用中文分隔符后，句子完整性显著提高")
    print("5. 实际项目中，建议始终使用中文分隔符处理中文文档")

    logger.info("中文文本分割演示完成")


def main():
    """
    主函数：依次运行所有演示

    执行顺序的设计逻辑：
    1. 为什么需要分割 -> 先理解动机，再学方法
    2. RecursiveCharacterTextSplitter -> 掌握核心工具的使用
    3. chunk_size对比 -> 理解最关键参数的影响
    4. overlap效果 -> 理解第二个关键参数的作用
    5. 中文分割 -> 针对中文场景的特殊优化

    为什么按这个顺序？
    - 从动机到方法 -> 先知道"为什么"，再学"怎么做"
    - 从核心到细节 -> 先掌握基本用法，再深入参数调优
    - 从通用到特殊 -> 先学通用方法，再针对中文优化
    """
    print("=" * 60)
    print("第2章：文本分割策略")
    print("=" * 60)

    logger.info("开始第2章演示")

    demos = [
        ("为什么需要文本分割", demo_why_split),
        ("RecursiveCharacterTextSplitter详解", demo_recursive_splitter),
        ("chunk_size对比实验", demo_chunk_size_comparison),
        ("chunk_overlap效果对比", demo_overlap_effect),
        ("中文文本分割", demo_chinese_splitting),
    ]

    for name, demo_func in demos:
        try:
            logger.info(f"开始演示: {name}")
            demo_func()
        except Exception as e:
            print(f"\n[错误] 演示'{name}'执行失败: {e}")
            logger.error(f"演示'{name}'执行失败: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("第2章学习总结")
    print("=" * 60)
    print("""
核心要点回顾：

1. 为什么需要文本分割？
   - LLM有上下文窗口限制 -> 无法处理整篇长文档
   - 向量嵌入有最佳长度 -> 太长语义模糊，太短信息不足
   - 检索需要精确匹配 -> 小块文本比整篇文档更容易匹配

2. RecursiveCharacterTextSplitter
   - 递归尝试多个分隔符，优先在语义边界分割
   - 比CharacterTextSplitter更好 -> 不会在任意位置切断
   - 是LangChain中最推荐的文本分割器

3. chunk_size的选择
   - 太小(200): 检索精确但上下文不足
   - 中等(500): 平衡精确性和完整性，通用场景推荐
   - 太大(1000): 上下文丰富但语义可能模糊
   - 需要根据具体场景实验调优

4. chunk_overlap的作用
   - 防止关键信息在分割点丢失
   - 推荐值为chunk_size的10%-20%
   - 太小无效，太大冗余

5. 中文文本分割
   - 默认分隔符为英文设计，对中文效果不佳
   - 应加入中文标点(。！？；)作为分隔符
   - 使用中文分隔符后句子完整性显著提高

下一章预告：向量化与嵌入 - 如何将文本块转换为向量表示
""")

    logger.info("第2章演示完成")


if __name__ == '__main__':
    main()
