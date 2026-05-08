"""
RAG文档分割演示项目

本项目提供多种文档分割策略的实现，用于RAG（检索增强生成）系统中的文档预处理环节。

模块结构：
- src.splitters: 文档分割器实现
- src.utils: 工具函数

主要功能：
- 字符分割：按固定字符数分割文档
- Token分割：按Token数分割文档
- 语义分割：基于语义相似度分割文档
- 句子分割：基于句子边界分割文档
"""

from .splitters import (
    CharacterSplitter,
    TokenSplitter,
    SemanticSplitter,
    SentenceSplitter
)

from .utils import (
    read_file,
    write_file,
    count_characters,
    count_words,
    print_chunk_info,
    display_chunks,
    compare_splitters
)

__all__ = [
    "CharacterSplitter",
    "TokenSplitter",
    "SemanticSplitter",
    "SentenceSplitter",
    "read_file",
    "write_file",
    "count_characters",
    "count_words",
    "print_chunk_info",
    "display_chunks",
    "compare_splitters"
]