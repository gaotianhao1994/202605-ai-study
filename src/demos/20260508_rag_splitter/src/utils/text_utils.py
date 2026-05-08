def read_file(file_path: str) -> str:
    """
    读取文本文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件失败: {str(e)}")


def write_file(file_path: str, content: str) -> None:
    """
    将内容写入文本文件
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise Exception(f"写入文件失败: {str(e)}")


def count_characters(text: str) -> int:
    """
    计算文本的字符数
    
    Args:
        text: 文本内容
        
    Returns:
        int: 字符数量
    """
    return len(text)


def count_words(text: str) -> int:
    """
    计算文本的词数（中文按字符，英文按空格分割）
    
    Args:
        text: 文本内容
        
    Returns:
        int: 词数
    """
    # 统计中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    # 统计英文单词
    import re
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    
    return chinese_chars + english_words


def print_chunk_info(chunks: list, splitter_name: str = "分割器") -> None:
    """
    打印分割结果的统计信息
    
    Args:
        chunks: 分割后的chunk列表
        splitter_name: 分割器名称（用于输出标识）
    """
    if not chunks:
        print(f"{splitter_name} - 没有生成任何chunk")
        return
    
    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    max_length = max(len(chunk) for chunk in chunks)
    min_length = min(len(chunk) for chunk in chunks)
    
    print(f"\n{'-'*60}")
    print(f"{splitter_name} 分割结果统计:")
    print(f"  - Chunk数量: {len(chunks)}")
    print(f"  - 总字符数: {total_length}")
    print(f"  - 平均长度: {avg_length:.1f} 字符")
    print(f"  - 最大长度: {max_length} 字符")
    print(f"  - 最小长度: {min_length} 字符")
    print(f"{'-'*60}")


def display_chunks(chunks: list, max_chunks: int = 5) -> None:
    """
    显示分割后的chunk内容
    
    Args:
        chunks: 分割后的chunk列表
        max_chunks: 最多显示的chunk数量，默认5
    """
    num_to_display = min(len(chunks), max_chunks)
    
    for i in range(num_to_display):
        print(f"\n=== Chunk {i+1}/{len(chunks)} ===")
        print(f"长度: {len(chunks[i])} 字符")
        print("-" * 40)
        print(chunks[i])
        print("-" * 40)
    
    if len(chunks) > max_chunks:
        print(f"\n... 还有 {len(chunks) - max_chunks} 个chunk未显示")


def compare_splitters(splitter_results: dict) -> None:
    """
    比较不同分割器的分割结果
    
    Args:
        splitter_results: 字典，键为分割器名称，值为(chunks, info)元组
    """
    print("\n" + "="*80)
    print("分割器对比结果")
    print("="*80)
    
    # 打印表头
    print(f"{'分割器':<20} {'Chunk数':<10} {'平均长度':<10} {'最大长度':<10} {'最小长度':<10}")
    print("-"*80)
    
    for name, (chunks, info) in splitter_results.items():
        if chunks:
            avg_len = sum(len(c) for c in chunks) / len(chunks)
            max_len = max(len(c) for c in chunks)
            min_len = min(len(c) for c in chunks)
            print(f"{name:<20} {len(chunks):<10} {avg_len:<10.1f} {max_len:<10} {min_len:<10}")
        else:
            print(f"{name:<20} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
    
    print("="*80)