"""
工具函数模块

提供文档读写、统计分析等辅助功能。
"""

from .text_utils import (
    read_file,
    write_file,
    count_characters,
    count_words,
    print_chunk_info,
    display_chunks,
    compare_splitters
)

__all__ = [
    "read_file",
    "write_file",
    "count_characters",
    "count_words",
    "print_chunk_info",
    "display_chunks",
    "compare_splitters"
]