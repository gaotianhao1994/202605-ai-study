#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG文档分割演示脚本

本脚本演示四种文档分割策略的使用方法，并对比它们的效果。

运行方式：
    python demo.py

功能：
    1. 字符分割器演示
    2. Token分割器演示
    3. 语义分割器演示
    4. 句子分割器演示
    5. 四种分割器对比
"""

import os
import sys

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    CharacterSplitter,
    TokenSplitter,
    SemanticSplitter,
    SentenceSplitter,
    read_file,
    print_chunk_info,
    display_chunks,
    compare_splitters
)


def main():
    """
    主演示函数
    """
    print("="*80)
    print("RAG文档分割演示项目")
    print("="*80)
    print("\n本项目演示四种文档分割策略：")
    print("  1. 字符分割器 - 按固定字符数分割")
    print("  2. Token分割器 - 按Token数分割")
    print("  3. 语义分割器 - 基于语义相似度分割")
    print("  4. 句子分割器 - 基于句子边界分割")
    print("="*80)

    # 读取示例文档
    document_path = os.path.join(os.path.dirname(__file__), "data", "sample_document.txt")
    
    try:
        document = read_file(document_path)
        print(f"\n已加载示例文档: {document_path}")
        print(f"文档长度: {len(document)} 字符")
    except FileNotFoundError:
        print(f"错误：未找到示例文档 {document_path}")
        sys.exit(1)

    # 存储所有分割结果用于对比
    results = {}

    # 1. 字符分割器演示
    print("\n" + "="*80)
    print("【演示1】字符分割器 (CharacterSplitter)")
    print("="*80)
    print("\n特点：")
    print("  - 按固定字符数分割")
    print("  - 简单直观，计算速度快")
    print("  - 可能切断句子或段落")
    
    char_splitter = CharacterSplitter(chunk_size=300, chunk_overlap=50)
    char_chunks = char_splitter.split(document)
    
    print(f"\n配置参数:")
    print(f"  - chunk_size: {char_splitter.chunk_size}")
    print(f"  - chunk_overlap: {char_splitter.chunk_overlap}")
    
    print_chunk_info(char_chunks, "字符分割器")
    display_chunks(char_chunks, max_chunks=3)
    results["字符分割器"] = (char_chunks, char_splitter.get_info())

    # 2. Token分割器演示
    print("\n" + "="*80)
    print("【演示2】Token分割器 (TokenSplitter)")
    print("="*80)
    print("\n特点：")
    print("  - 按Token数分割")
    print("  - 直接考虑模型上下文窗口限制")
    print("  - 需要依赖Token编码库")
    
    try:
        token_splitter = TokenSplitter(chunk_size=200, chunk_overlap=30, model_name="gpt-3.5-turbo")
        token_chunks = token_splitter.split(document)
        
        print(f"\n配置参数:")
        print(f"  - chunk_size: {token_splitter.chunk_size}")
        print(f"  - chunk_overlap: {token_splitter.chunk_overlap}")
        print(f"  - model_name: {token_splitter.model_name}")
        print(f"  - 原始文档Token数: {token_splitter.count_tokens(document)}")
        
        print_chunk_info(token_chunks, "Token分割器")
        display_chunks(token_chunks, max_chunks=3)
        results["Token分割器"] = (token_chunks, token_splitter.get_info())
    except ImportError as e:
        print(f"\n警告：{e}")
        print("跳过Token分割器演示，请安装依赖后重试")

    # 3. 语义分割器演示
    print("\n" + "="*80)
    print("【演示3】语义分割器 (SemanticSplitter)")
    print("="*80)
    print("\n特点：")
    print("  - 基于语义相似度分割")
    print("  - 保持语义完整性")
    print("  - 计算成本较高，需要加载预训练模型")
    
    try:
        semantic_splitter = SemanticSplitter(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.5,
            max_chunk_size=500
        )
        semantic_chunks = semantic_splitter.split(document)
        
        print(f"\n配置参数:")
        print(f"  - model_name: {semantic_splitter.model_name}")
        print(f"  - similarity_threshold: {semantic_splitter.similarity_threshold}")
        print(f"  - max_chunk_size: {semantic_splitter.max_chunk_size}")
        
        print_chunk_info(semantic_chunks, "语义分割器")
        display_chunks(semantic_chunks, max_chunks=3)
        results["语义分割器"] = (semantic_chunks, semantic_splitter.get_info())
    except ImportError as e:
        print(f"\n警告：{e}")
        print("跳过敏义分割器演示，请安装依赖后重试")

    # 4. 句子分割器演示
    print("\n" + "="*80)
    print("【演示4】句子分割器 (SentenceSplitter)")
    print("="*80)
    print("\n特点：")
    print("  - 基于句子边界分割")
    print("  - 保持句子完整性")
    print("  - 支持中英文混合文本")
    print("  - 处理缩写词和数字")
    
    sentence_splitter = SentenceSplitter(preserve_whitespace=False, min_sentence_length=2)
    sentences = sentence_splitter.split(document)
    
    print(f"\n配置参数:")
    print(f"  - preserve_whitespace: {sentence_splitter.preserve_whitespace}")
    print(f"  - min_sentence_length: {sentence_splitter.min_sentence_length}")
    
    print(f"\n分割结果统计:")
    print(f"  - 句子数量: {len(sentences)}")
    print("-" * 50)
    
    # 显示前5个句子
    for i, sentence in enumerate(sentences[:5]):
        print(f"\n句子 {i+1}: {sentence}")
    
    if len(sentences) > 5:
        print(f"\n... 还有 {len(sentences) - 5} 个句子未显示")
    
    # 演示句子组合功能
    print("\n" + "-" * 50)
    print("句子组合示例（每3句一组，重叠1句）:")
    sentence_chunks = sentence_splitter.split_with_overlap(document, chunk_size=3, overlap_size=1)
    print(f"组合后chunk数量: {len(sentence_chunks)}")
    display_chunks(sentence_chunks, max_chunks=3)
    results["句子分割器"] = (sentence_chunks, sentence_splitter.get_info())

    # 5. 对比结果
    if results:
        compare_splitters(results)
        
        print("\n" + "="*80)
        print("选择建议")
        print("="*80)
        print("""
选择合适的分割策略取决于您的使用场景：

1. 字符分割器：
   - 适用：简单文本、短文档、快速原型
   - 优点：简单、快速、无需额外依赖
   - 缺点：可能切断语义

2. Token分割器：
   - 适用：需要与大语言模型配合使用的场景
   - 优点：直接考虑模型上下文限制
   - 缺点：依赖Token编码库

3. 语义分割器：
   - 适用：需要保持语义完整性的复杂文档
   - 优点：保持语义连贯
   - 缺点：计算成本高、需要加载模型

4. 句子分割器：
   - 适用：需要保持句子完整性的场景、中英文混合文档
   - 优点：保持句子边界、支持中英文、处理缩写词和数字
   - 缺点：chunk大小不均匀

最佳实践建议：
- 对于大多数RAG应用，建议从Token分割器开始
- 如果文档包含长段落或复杂结构，考虑使用语义分割器
- 如果需要保持句子完整性，使用句子分割器
- 调整chunk_size和chunk_overlap参数以获得最佳效果
""")

    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("\n尝试修改demo.py中的参数来观察不同分割效果：")
    print("  - chunk_size: 控制每个chunk的大小")
    print("  - chunk_overlap: 控制相邻chunk的重叠程度")
    print("  - similarity_threshold: 语义分割器的相似度阈值")
    print("  - min_sentence_length: 句子分割器的最小句子长度")


if __name__ == "__main__":
    main()