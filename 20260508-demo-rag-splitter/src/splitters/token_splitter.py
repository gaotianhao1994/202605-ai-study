try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenSplitter:
    """
    令牌分割器：按Token数分割文档
    
    在大语言模型中，输入文本需要转换为Token（子词单元）。
    不同模型有不同的Token限制（如GPT-3.5限制4096 Token）。
    此分割器确保每个chunk的Token数不超过模型限制。
    
    优点：直接考虑模型的上下文窗口限制
    缺点：需要依赖Token编码库，计算相对复杂
    
    参数说明：
        chunk_size: 每个chunk的最大Token数
        chunk_overlap: 相邻chunk之间的重叠Token数
        model_name: 使用的模型名称，用于选择正确的Token编码器
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        """
        初始化令牌分割器
        
        Args:
            chunk_size: 每个chunk的最大Token数，默认512
            chunk_overlap: 相邻chunk之间的重叠Token数，默认50
            model_name: 模型名称，用于获取对应的Token编码器
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.encoder = None
        
        # 参数验证
        if self.chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap不能为负数")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        
        # 获取Token编码器
        if not self._get_encoder():
            raise ImportError("需要安装tiktoken库：pip install tiktoken")
    
    def _get_encoder(self):
        """
        获取指定模型的Token编码器
        
        不同模型使用不同的Token编码方式，需要选择正确的编码器。
        
        Returns:
            bool: 是否成功加载编码器
        """
        if not TIKTOKEN_AVAILABLE:
            return False
            
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # 如果模型名称不在预定义列表中，使用默认编码器
            print(f"警告：未找到模型 {self.model_name} 的编码器，使用默认编码器")
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        return self.encoder is not None
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的Token数量
        
        Args:
            text: 需要计算的文本
            
        Returns:
            int: Token数量
        """
        return len(self.encoder.encode(text))
    
    def encode(self, text: str) -> list:
        """
        将文本编码为Token列表
        
        Args:
            text: 需要编码的文本
            
        Returns:
            list: Token ID列表
        """
        return self.encoder.encode(text)
    
    def decode(self, tokens: list) -> str:
        """
        将Token列表解码为文本
        
        Args:
            tokens: Token ID列表
            
        Returns:
            str: 解码后的文本
        """
        return self.encoder.decode(tokens)
    
    def split(self, text: str) -> list:
        """
        将文本分割成多个chunk，每个chunk的Token数不超过chunk_size
        
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
        
        # 将文本编码为Token
        tokens = self.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return []
        
        chunks = []
        start = 0
        
        # 循环分割Token
        while start < total_tokens:
            # 计算当前chunk的结束位置
            end = min(start + self.chunk_size, total_tokens)
            
            # 如果不是最后一个chunk，尝试在合理位置分割
            if end < total_tokens:
                # 从end位置向前查找句子边界（优先找换行符）
                # 先将Token解码为文本片段来查找边界
                temp_text = self.decode(tokens[start:end])
                
                # 查找最后一个换行符
                newline_pos = temp_text.rfind('\n')
                if newline_pos != -1 and newline_pos > self.chunk_overlap:
                    # 计算对应的Token位置
                    newline_tokens = self.encode(temp_text[:newline_pos])
                    end = start + len(newline_tokens)
                else:
                    # 查找最后一个空格
                    space_pos = temp_text.rfind(' ')
                    if space_pos != -1 and space_pos > self.chunk_overlap:
                        space_tokens = self.encode(temp_text[:space_pos])
                        end = start + len(space_tokens)
            
            # 提取当前chunk的Token并解码为文本
            chunk_tokens = tokens[start:end]
            chunk = self.decode(chunk_tokens).strip()
            
            # 如果chunk不为空，添加到结果列表
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个chunk的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            
            # 防止无限循环
            if start >= total_tokens:
                break
            if start == end:
                start += 1
        
        return chunks
    
    def get_info(self) -> dict:
        """
        获取分割器的配置信息
        
        Returns:
            dict: 包含配置信息的字典
        """
        return {
            "splitter_type": "TokenSplitter",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model_name": self.model_name
        }


# 示例用法
if __name__ == "__main__":
    # 读取示例文档
    with open("../data/sample_document.txt", "r", encoding="utf-8") as f:
        document = f.read()
    
    # 创建分割器实例
    splitter = TokenSplitter(chunk_size=200, chunk_overlap=30, model_name="gpt-3.5-turbo")
    
    # 分割文档
    chunks = splitter.split(document)
    
    # 输出分割结果
    print(f"=== Token分割器示例 ===")
    print(f"配置: {splitter.get_info()}")
    print(f"原始文档Token数: {splitter.count_tokens(document)}")
    print(f"分割后chunk数量: {len(chunks)}")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        token_count = splitter.count_tokens(chunk)
        print(f"\nChunk {i+1}:")
        print(f"Token数: {token_count}")
        print(f"内容:\n{chunk}")
        print("-" * 50)