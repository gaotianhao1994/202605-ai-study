"""
文档分割器模块

本模块提供多种文档分割策略，用于RAG系统中的文档预处理环节。

包含的分割器：
- CharacterSplitter: 按字符数分割
- TokenSplitter: 按Token数分割
- SemanticSplitter: 基于语义相似度分割
- SentenceSplitter: 基于句子的分割

使用示例：
    from src.splitters import CharacterSplitter, TokenSplitter, SemanticSplitter, SentenceSplitter
    
    # 创建字符分割器
    char_splitter = CharacterSplitter(chunk_size=500, chunk_overlap=50)
    
    # 创建Token分割器
    token_splitter = TokenSplitter(chunk_size=512, chunk_overlap=50)
    
    # 创建语义分割器
    semantic_splitter = SemanticSplitter(similarity_threshold=0.5)
    
    # 创建句子分割器
    sentence_splitter = SentenceSplitter()
    
    # 分割文档
    chunks = char_splitter.split(document_text)
"""

from .character_splitter import CharacterSplitter
from .token_splitter import TokenSplitter
from .semantic_splitter import SemanticSplitter
from .sentence_splitter import SentenceSplitter

__all__ = ["CharacterSplitter", "TokenSplitter", "SemanticSplitter", "SentenceSplitter"]