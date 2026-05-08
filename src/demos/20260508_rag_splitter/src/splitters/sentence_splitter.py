import re


class SentenceSplitter:
    """
    句子分割器：基于规则的文本句子分割
    
    该分割器能够将文本精准拆分为独立句子，支持中英文混合文本，
    合理处理标点符号与句子边界。
    
    特点：
    - 支持中英文标点（。！？.?!）
    - 处理缩写词（如 Mr., Mrs., No.）
    - 处理数字（如 3.14）
    - 处理括号、引号内的内容
    - 支持换行符作为分隔符
    
    参数说明：
        preserve_whitespace: 是否保留句子间的空白字符
        min_sentence_length: 最小句子长度（字符数）
    """
    
    def __init__(self, preserve_whitespace: bool = False, min_sentence_length: int = 2):
        """
        初始化句子分割器
        
        Args:
            preserve_whitespace: 是否保留句子间的空白字符，默认False
            min_sentence_length: 最小句子长度，默认2个字符
        """
        self.preserve_whitespace = preserve_whitespace
        self.min_sentence_length = min_sentence_length
        
        # 常见英文缩写词列表
        self.abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'rev', 'sr', 'jr',
            'no', 'nos', 'vol', 'pp', 'ed', 'eds', 'trans',
            'vs', 'etc', 'i.e', 'e.g', 'cf', 'viz', 'esp',
            'p', 'pp', 'ch', 'fig', 'figs', 'sec', 'secs',
            'para', 'paras', 'chap', 'chaps', 'app', 'apps',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec',
            'st', 'ave', 'blvd', 'rd', 'ln', 'ct'
        }
        
        # 初始化正则表达式模式
        self._compile_patterns()
    
    def _compile_patterns(self):
        """
        编译正则表达式模式
        """
        # 匹配句子结束标记
        # 匹配：中文句号、感叹号、问号，英文句号、感叹号、问号
        # 后面可能跟引号、括号、换行或空格
        self.sentence_ending_pattern = re.compile(
            r'([。！？.!?]+)(["\'”’））\]]*)(\s+|$|\n)'
        )
        
        # 匹配缩写词后跟句号的模式
        # 如 Mr., Mrs., No.
        self.abbreviation_pattern = re.compile(
            r'\b(' + '|'.join(self.abbreviations) + r')\.(?=\s+[A-Z])',
            re.IGNORECASE
        )
        
        # 匹配数字后跟句号的模式（如 3.14）
        self.number_pattern = re.compile(
            r'(\d+)\.(\d+)'
        )
        
        # 匹配括号内的内容
        self.parentheses_pattern = re.compile(
            r'([（\(][^）\)]*[）\)])'
        )
    
    def _is_abbreviation(self, text: str, position: int) -> bool:
        """
        判断指定位置是否为缩写词
        
        Args:
            text: 文本内容
            position: 句号的位置
            
        Returns:
            bool: 是否为缩写词
        """
        # 向前查找单词
        start = position - 1
        while start >= 0 and text[start].isalpha():
            start -= 1
        
        word = text[start+1:position].lower()
        return word in self.abbreviations
    
    def _is_number(self, text: str, position: int) -> bool:
        """
        判断指定位置的句号是否为数字的一部分（如 3.14）
        
        Args:
            text: 文本内容
            position: 句号的位置
            
        Returns:
            bool: 是否为数字的一部分
        """
        # 检查前面是否为数字
        if position > 0 and text[position-1].isdigit():
            # 检查后面是否为数字
            if position < len(text) - 1 and text[position+1].isdigit():
                return True
        return False
    
    def split(self, text: str) -> list:
        """
        将文本分割成句子列表
        
        Args:
            text: 需要分割的文本
            
        Returns:
            list: 句子列表
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")
        if len(text) == 0:
            return []
        
        sentences = []
        current_sentence = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # 检查是否为句子结束标记
            if char in '。！？.!?':
                # 检查是否为缩写词或数字
                if self._is_abbreviation(text, i) or self._is_number(text, i):
                    # 不是句子结束，继续
                    current_sentence.append(char)
                    i += 1
                    continue
                
                # 是句子结束标记
                current_sentence.append(char)
                
                # 检查后面是否有引号或括号
                j = i + 1
                while j < len(text) and text[j] in '\"\'”’））\\]':
                    current_sentence.append(text[j])
                    j += 1
                i = j
                
                # 收集句子
                sentence = ''.join(current_sentence).strip()
                if len(sentence) >= self.min_sentence_length:
                    sentences.append(sentence)
                current_sentence = []
                
                # 跳过句子间的空白
                while i < len(text) and text[i] in ' \t\n\r':
                    if self.preserve_whitespace:
                        current_sentence.append(text[i])
                    i += 1
                continue
            
            current_sentence.append(char)
            i += 1
        
        # 添加最后一个句子
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if len(sentence) >= self.min_sentence_length:
                sentences.append(sentence)
        
        return sentences
    
    def split_with_overlap(self, text: str, chunk_size: int = 3, overlap_size: int = 1) -> list:
        """
        将句子组合成有重叠的chunk
        
        Args:
            text: 需要分割的文本
            chunk_size: 每个chunk包含的句子数，默认3
            overlap_size: 相邻chunk的重叠句子数，默认1
            
        Returns:
            list: chunk列表，每个chunk是多个句子的组合
        """
        sentences = self.split(text)
        
        if len(sentences) == 0:
            return []
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            end = min(i + chunk_size, len(sentences))
            chunk = ' '.join(sentences[i:end])
            chunks.append(chunk)
            
            # 计算下一个chunk的起始位置
            if i + chunk_size >= len(sentences):
                break
            i += chunk_size - overlap_size
        
        return chunks
    
    def get_info(self) -> dict:
        """
        获取分割器的配置信息
        
        Returns:
            dict: 包含配置信息的字典
        """
        return {
            "splitter_type": "SentenceSplitter",
            "preserve_whitespace": self.preserve_whitespace,
            "min_sentence_length": self.min_sentence_length
        }


# 示例用法
if __name__ == "__main__":
    # 读取示例文档
    with open("../data/sample_document.txt", "r", encoding="utf-8") as f:
        document = f.read()
    
    # 创建分割器实例
    splitter = SentenceSplitter(preserve_whitespace=False, min_sentence_length=2)
    
    # 分割文档
    sentences = splitter.split(document)
    
    # 输出分割结果
    print(f"=== 句子分割器示例 ===")
    print(f"配置: {splitter.get_info()}")
    print(f"分割后句子数量: {len(sentences)}")
    print("-" * 50)
    
    for i, sentence in enumerate(sentences[:10]):
        print(f"\n句子 {i+1}: {sentence}")
    
    if len(sentences) > 10:
        print(f"\n... 还有 {len(sentences) - 10} 个句子未显示")
    
    # 演示句子组合功能
    print("\n" + "-" * 50)
    print("句子组合示例（每3句一组，重叠1句）:")
    chunks = splitter.split_with_overlap(document, chunk_size=3, overlap_size=1)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:\n{chunk}")