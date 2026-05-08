#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆/知识系统模块
核心功能：存储、检索和管理知识与经验

这个模块模拟了人类的记忆系统，包括：
1. 短期记忆：临时存储当前会话信息
2. 长期记忆：永久存储知识和经验
3. 知识检索：根据关键词查找相关信息
4. 记忆巩固：将短期记忆转化为长期记忆
"""

from typing import List, Dict, Any, Optional
import time
import json


class MemoryItem:
    """
    单个记忆项
    代表一条存储的知识或经验
    
    记忆类型：
    - fact: 事实知识（如"水的沸点是100°C"）
    - experience: 经验（如"上次用这个方法成功了"）
    - reflection: 反思结果（如"上次的决策不够好"）
    - skill: 技能（如"如何使用Python"）
    """
    
    def __init__(self, content: str, memory_type: str = "fact", confidence: float = 0.8):
        """
        初始化记忆项
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            confidence: 置信度（0.0-1.0）
        """
        self.content = content
        self.memory_type = memory_type
        self.confidence = confidence
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆项转换为字典格式以便存储"""
        return {
            'content': self.content,
            'memory_type': self.memory_type,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'accessed_at': self.accessed_at,
            'access_count': self.access_count
        }
    
    def __repr__(self):
        return f"MemoryItem(type={self.memory_type}, confidence={self.confidence:.2f}, content={self.content[:40]}...)"


class MemorySystem:
    """
    记忆系统
    管理短期记忆和长期记忆
    
    核心功能：
    1. store(): 存储记忆
    2. retrieve(): 根据关键词检索记忆
    3. consolidate(): 将短期记忆巩固到长期记忆
    4. forget(): 删除记忆
    5. save/load(): 持久化存储
    """
    
    def __init__(self, max_short_term=50, max_long_term=1000):
        """
        初始化记忆系统
        
        Args:
            max_short_term: 短期记忆最大容量
            max_long_term: 长期记忆最大容量
        """
        self.short_term_memory: List[MemoryItem] = []  # 短期记忆（临时）
        self.long_term_memory: List[MemoryItem] = []   # 长期记忆（永久）
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
    
    def store(self, memory_data: Dict[str, Any], short_term: bool = False):
        """
        存储记忆
        
        Args:
            memory_data: 记忆数据字典，包含 content, memory_type, confidence
            short_term: 是否存储到短期记忆
        """
        # 创建记忆项
        memory = MemoryItem(
            content=memory_data.get('content', ''),
            memory_type=memory_data.get('memory_type', 'fact'),
            confidence=memory_data.get('confidence', 0.8)
        )
        
        if short_term:
            # 存储到短期记忆
            self.short_term_memory.append(memory)
            # 如果超过容量，移除最旧的
            if len(self.short_term_memory) > self.max_short_term:
                self.short_term_memory.pop(0)
        else:
            # 存储到长期记忆
            self.long_term_memory.append(memory)
            # 如果超过容量，移除访问次数最少的
            if len(self.long_term_memory) > self.max_long_term:
                self._evict_least_used()
    
    def retrieve(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        根据关键词检索相关记忆
        
        Args:
            query: 查询关键词
            top_n: 返回结果数量
        
        Returns:
            匹配的记忆列表，按相关性排序
        """
        all_memories = self.short_term_memory + self.long_term_memory
        results = []
        
        for memory in all_memories:
            # 计算匹配度
            score = self._calculate_match_score(query, memory)
            if score > 0:
                results.append({
                    'content': memory.content,
                    'memory_type': memory.memory_type,
                    'confidence': memory.confidence,
                    'match_score': score,
                    'access_count': memory.access_count
                })
                # 更新访问时间和次数
                memory.accessed_at = time.time()
                memory.access_count += 1
        
        # 按匹配度排序
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results[:top_n]
    
    def _calculate_match_score(self, query: str, memory: MemoryItem) -> float:
        """
        计算查询与记忆的匹配度
        
        Args:
            query: 查询字符串
            memory: 记忆项
        
        Returns:
            匹配分数（0.0-1.0）
        """
        score = 0.0
        
        # 将查询和记忆内容转换为小写进行比较
        query_lower = query.lower()
        content_lower = memory.content.lower()
        
        # 检查完全匹配的词
        query_words = query_lower.split()
        content_words = content_lower.split()
        
        # 计算词匹配率
        matched_words = [word for word in query_words if word in content_words]
        if query_words:
            score += len(matched_words) / len(query_words) * 0.6
        
        # 检查子字符串匹配
        if query_lower in content_lower:
            score += 0.4
        
        # 根据记忆类型给予不同权重
        type_weights = {
            'experience': 1.2,
            'reflection': 1.1,
            'skill': 1.1,
            'fact': 1.0
        }
        score *= type_weights.get(memory.memory_type, 1.0)
        
        # 根据置信度调整
        score *= memory.confidence
        
        # 根据访问次数调整（访问次数越多可能越重要）
        score *= min(1.0 + memory.access_count * 0.01, 1.5)
        
        return min(1.0, score)
    
    def consolidate(self):
        """
        将短期记忆巩固到长期记忆
        
        模拟人类的记忆巩固过程：
        - 短期记忆中的内容经过重复访问后会被转移到长期记忆
        - 低置信度的记忆可能被遗忘
        """
        memories_to_consolidate = []
        memories_to_forget = []
        
        for memory in self.short_term_memory:
            # 如果访问次数足够多，巩固到长期记忆
            if memory.access_count >= 3 or (time.time() - memory.created_at) > 300:
                memories_to_consolidate.append(memory)
            # 如果置信度太低，遗忘
            elif memory.confidence < 0.3:
                memories_to_forget.append(memory)
        
        # 巩固到长期记忆
        for memory in memories_to_consolidate:
            self.store(memory.to_dict(), short_term=False)
            self.short_term_memory.remove(memory)
        
        # 遗忘低置信度记忆
        for memory in memories_to_forget:
            self.short_term_memory.remove(memory)
        
        return {
            'consolidated': len(memories_to_consolidate),
            'forgotten': len(memories_to_forget),
            'remaining_short': len(self.short_term_memory)
        }
    
    def forget(self, index: int, long_term: bool = True) -> bool:
        """
        删除指定记忆
        
        Args:
            index: 记忆在列表中的索引
            long_term: 是否删除长期记忆（False则删除短期记忆）
        
        Returns:
            是否删除成功
        """
        memory_list = self.long_term_memory if long_term else self.short_term_memory
        
        if 0 <= index < len(memory_list):
            del memory_list[index]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'total_count': len(self.short_term_memory) + len(self.long_term_memory),
            'short_term_types': self._get_type_distribution(self.short_term_memory),
            'long_term_types': self._get_type_distribution(self.long_term_memory)
        }
    
    def _get_type_distribution(self, memory_list: List[MemoryItem]) -> Dict[str, int]:
        """获取记忆类型分布统计"""
        distribution = {}
        for memory in memory_list:
            distribution[memory.memory_type] = distribution.get(memory.memory_type, 0) + 1
        return distribution
    
    def _evict_least_used(self):
        """
        淘汰访问次数最少的记忆
        
        当长期记忆超过容量时，移除访问次数最少的记忆
        """
        if not self.long_term_memory:
            return
        
        # 找到访问次数最少的记忆
        least_used = min(self.long_term_memory, key=lambda m: m.access_count)
        self.long_term_memory.remove(least_used)
    
    def save(self, filename: str = "memory.json"):
        """
        将记忆系统保存到文件
        
        Args:
            filename: 保存的文件名
        """
        data = {
            'short_term': [m.to_dict() for m in self.short_term_memory],
            'long_term': [m.to_dict() for m in self.long_term_memory],
            'metadata': {
                'saved_at': time.time(),
                'short_term_count': len(self.short_term_memory),
                'long_term_count': len(self.long_term_memory)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filename: str = "memory.json") -> bool:
        """
        从文件加载记忆系统
        
        Args:
            filename: 要加载的文件名
        
        Returns:
            是否加载成功
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载短期记忆
            self.short_term_memory = []
            for item in data.get('short_term', []):
                memory = MemoryItem(
                    content=item['content'],
                    memory_type=item['memory_type'],
                    confidence=item['confidence']
                )
                memory.created_at = item['created_at']
                memory.accessed_at = item['accessed_at']
                memory.access_count = item['access_count']
                self.short_term_memory.append(memory)
            
            # 加载长期记忆
            self.long_term_memory = []
            for item in data.get('long_term', []):
                memory = MemoryItem(
                    content=item['content'],
                    memory_type=item['memory_type'],
                    confidence=item['confidence']
                )
                memory.created_at = item['created_at']
                memory.accessed_at = item['accessed_at']
                memory.access_count = item['access_count']
                self.long_term_memory.append(memory)
            
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"加载记忆失败: {e}")
            return False


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("记忆系统模块测试")
    print("=" * 60)
    
    # 创建记忆系统
    memory = MemorySystem()
    
    # 存储一些初始知识
    initial_knowledge = [
        {'content': '学习编程需要练习和耐心', 'memory_type': 'experience', 'confidence': 0.9},
        {'content': 'Python是一种流行的编程语言', 'memory_type': 'fact', 'confidence': 1.0},
        {'content': '坚持每天写代码可以提高编程能力', 'memory_type': 'skill', 'confidence': 0.85},
        {'content': '良好的代码注释有助于理解和维护', 'memory_type': 'fact', 'confidence': 0.95},
    ]
    
    for item in initial_knowledge:
        memory.store(item)
    
    print("\n初始存储的知识：")
    stats = memory.get_statistics()
    print(f"长期记忆: {stats['long_term_count']} 条")
    print(f"短期记忆: {stats['short_term_count']} 条")
    
    # 测试检索功能
    print("\n检索 '编程' 相关知识：")
    results = memory.retrieve("编程")
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. [{result['memory_type']}] {result['content']} (匹配度: {result['match_score']:.2f})")
    
    # 测试短期记忆
    print("\n存储到短期记忆：")
    memory.store({'content': '当前任务：学习Python函数', 'memory_type': 'fact', 'confidence': 0.9}, short_term=True)
    memory.store({'content': '下一步：练习编写循环', 'memory_type': 'fact', 'confidence': 0.85}, short_term=True)
    
    stats = memory.get_statistics()
    print(f"短期记忆: {stats['short_term_count']} 条")
    
    # 测试记忆巩固
    print("\n执行记忆巩固：")
    result = memory.consolidate()
    print(f"巩固了 {result['consolidated']} 条，遗忘了 {result['forgotten']} 条")
    
    # 测试保存和加载
    print("\n保存记忆系统...")
    memory.save("test_memory.json")
    print("已保存到 test_memory.json")
    
    # 创建新的记忆系统并加载
    new_memory = MemorySystem()
    if new_memory.load("test_memory.json"):
        print("\n加载记忆成功！")
        stats = new_memory.get_statistics()
        print(f"长期记忆: {stats['long_term_count']} 条")
        print(f"短期记忆: {stats['short_term_count']} 条")
    else:
        print("加载失败")