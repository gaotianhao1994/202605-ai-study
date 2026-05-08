import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticSplitter:
    """
    语义分割器：基于语义相似度分割文档
    
    此分割器使用预训练的语义模型来计算句子之间的相似度，
    在语义相似度较低的地方进行分割，从而保持chunk的语义完整性。
    
    优点：保持语义完整性，适合需要理解上下文的场景
    缺点：计算成本较高，需要加载预训练模型
    
    参数说明：
        model_name: 使用的语义模型名称
        similarity_threshold: 相似度阈值，低于此值则分割
        max_chunk_size: 每个chunk的最大Token数（作为安全限制）
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 similarity_threshold: float = 0.5, 
                 max_chunk_size: int = 512):
        """
        初始化语义分割器
        
        Args:
            model_name: 语义模型名称，默认使用all-MiniLM-L6-v2（轻量级但效果好）
            similarity_threshold: 相似度阈值，范围0-1，低于此值则分割
            max_chunk_size: 每个chunk的最大Token数
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.model = None
        
        # 参数验证
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold必须在0到1之间")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size必须大于0")
        
        # 加载语义模型
        if not self._load_model():
            raise ImportError("需要安装sentence-transformers库：pip install sentence-transformers")
    
    def _load_model(self):
        """
        加载预训练的语义模型
        
        首次运行时会自动下载模型，可能需要一些时间。
        
        Returns:
            bool: 是否成功加载模型
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
            
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"警告：加载模型失败，将使用简单分词作为备选方案")
            self.model = None
        
        return self.model is not None
    
    def _split_into_sentences(self, text: str) -> list:
        """
        将文本分割成句子（简单的中文句子分割）
        
        Args:
            text: 需要分割的文本
            
        Returns:
            list: 句子列表
        """
        import re
        
        # 使用正则表达式分割句子
        # 匹配中文句号、感叹号、问号，以及英文句号
        sentence_pattern = r'(?<=[。！？\?])\s*'
        sentences = re.split(sentence_pattern, text)
        
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _compute_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        计算两个句子之间的语义相似度
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            
        Returns:
            float: 相似度分数（0-1）
        """
        if self.model is None:
            # 如果模型加载失败，使用简单的词重叠相似度
            words1 = set(sentence1.split())
            words2 = set(sentence2.split())
            if len(words1) + len(words2) == 0:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
        
        # 使用预训练模型计算相似度
        embeddings = self.model.encode([sentence1, sentence2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    
    def split(self, text: str) -> list:
        """
        将文本分割成多个语义连贯的chunk
        
        Args:
            text: 需要分割的文本
            
        Returns:
            list: 分割后的chunk列表
        """
        # 检查输入
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")
        if len(text) == 0:
            return []
        
        # 将文本分割成句子
        sentences = self._split_into_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        # 遍历句子，根据语义相似度进行分割
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            
            # 计算当前句子与前一个句子的相似度
            similarity = self._compute_similarity(
                sentences[i-1], 
                current_sentence
            )
            
            # 检查是否需要分割
            # 条件1：相似度低于阈值
            # 条件2：当前chunk长度超过最大限制
            needs_split = (similarity < self.similarity_threshold) or \
                         (current_length + len(current_sentence) > self.max_chunk_size)
            
            if needs_split and len(current_chunk) > 0:
                # 将当前chunk加入结果列表
                chunks.append('\n'.join(current_chunk))
                current_chunk = [current_sentence]
                current_length = len(current_sentence)
            else:
                # 将当前句子加入当前chunk
                current_chunk.append(current_sentence)
                current_length += len(current_sentence)
        
        # 添加最后一个chunk
        if len(current_chunk) > 0:
            chunks.append('\n'.join(current_chunk))
        
        # 清理空chunk
        chunks = [c.strip() for c in chunks if c.strip()]
        
        return chunks
    
    def get_info(self) -> dict:
        """
        获取分割器的配置信息
        
        Returns:
            dict: 包含配置信息的字典
        """
        return {
            "splitter_type": "SemanticSplitter",
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "max_chunk_size": self.max_chunk_size
        }


# 示例用法
if __name__ == "__main__":
    # 读取示例文档
    with open("../data/sample_document.txt", "r", encoding="utf-8") as f:
        document = f.read()
    
    # 创建分割器实例
    # 使用轻量级模型，适合演示
    splitter = SemanticSplitter(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.5,
        max_chunk_size=500
    )
    
    # 分割文档
    chunks = splitter.split(document)
    
    # 输出分割结果
    print(f"=== 语义分割器示例 ===")
    print(f"配置: {splitter.get_info()}")
    print(f"分割后chunk数量: {len(chunks)}")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"长度: {len(chunk)} 字符")
        print(f"内容:\n{chunk}")
        print("-" * 50)