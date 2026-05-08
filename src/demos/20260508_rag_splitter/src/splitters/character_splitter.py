class CharacterSplitter:
    """
    字符分割器：按固定字符数分割文档
    
    这是最简单的文档分割方法，将文档按照指定的字符数进行分割。
    优点：简单直观，计算速度快
    缺点：可能切断句子或段落，破坏语义完整性
    
    参数说明：
        chunk_size: 每个chunk的最大字符数
        chunk_overlap: 相邻chunk之间的重叠字符数，用于保持上下文连续性
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化字符分割器
        
        Args:
            chunk_size: 每个chunk的最大字符数，默认500
            chunk_overlap: 相邻chunk之间的重叠字符数，默认50
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 参数验证
        if self.chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap不能为负数")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
    
    def split(self, text: str) -> list:
        """
        将文本分割成多个chunk
        
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
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # 循环分割文本
        while start < text_length:
            # 计算当前chunk的结束位置
            end = start + self.chunk_size
            
            # 如果还没到文本末尾，且不是最后一个chunk
            if end < text_length:
                # 找到合适的分割点，优先在换行符处分割
                # 从end位置向前查找换行符
                newline_pos = text.rfind('\n', start, end)
                
                if newline_pos != -1 and newline_pos > start + self.chunk_overlap:
                    # 如果找到了换行符且位置合适，就在换行符处分割
                    end = newline_pos
                else:
                    # 否则找到最后一个空格
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + self.chunk_overlap:
                        end = space_pos
            
            # 提取当前chunk
            chunk = text[start:end].strip()
            
            # 如果chunk不为空，添加到结果列表
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个chunk的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            
            # 防止无限循环（当chunk_overlap为0且文本末尾时）
            if start >= text_length:
                break
            if start == end:
                start += 1
        
        return chunks
    
    def get_info(self) -> dict:
        """
        获取分割器的配置信息
        
        Returns:
            dict: 包含chunk_size和chunk_overlap的字典
        """
        return {
            "splitter_type": "CharacterSplitter",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# 示例用法
if __name__ == "__main__":
    # 读取示例文档
    with open("../data/sample_document.txt", "r", encoding="utf-8") as f:
        document = f.read()
    
    # 创建分割器实例
    splitter = CharacterSplitter(chunk_size=300, chunk_overlap=50)
    
    # 分割文档
    chunks = splitter.split(document)
    
    # 输出分割结果
    print(f"=== 字符分割器示例 ===")
    print(f"配置: {splitter.get_info()}")
    print(f"原始文档长度: {len(document)} 字符")
    print(f"分割后chunk数量: {len(chunks)}")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"长度: {len(chunk)} 字符")
        print(f"内容:\n{chunk}")
        print("-" * 50)