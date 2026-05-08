#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
思考与自我反思框架 - 主演示程序

本演示展示了一个完整的"思考-反思-学习"循环：
1. 使用思考引擎处理问题
2. 使用反思引擎评估思考过程
3. 将经验存储到记忆系统
4. 持续改进决策质量

框架架构：
┌─────────────────────────────────────────────────────────────┐
│                    思考与自我反思框架                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────────┐     ┌──────────────────┐               │
│    │  输入问题     │────▶│   思考引擎        │               │
│    └──────────────┘     │ (ThinkingEngine) │               │
│                         └────────┬─────────┘               │
│                                  │                         │
│                                  ▼                         │
│                         ┌──────────────────┐               │
│                         │   反思引擎        │               │
│                         │(SelfReflection) │               │
│                         └────────┬─────────┘               │
│                                  │                         │
│                     ┌────────────┼────────────┐             │
│                     ▼            ▼            ▼             │
│              ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│              │ 评估结果 │  │ 偏差识别 │  │改进建议 │         │
│              └─────────┘  └─────────┘  └─────────┘         │
│                     │            │            │             │
│                     └────────────┼────────────┘             │
│                                  ▼                         │
│                         ┌──────────────────┐               │
│                         │   记忆系统        │               │
│                         │ (MemorySystem)  │               │
│                         │  (存储经验)       │               │
│                         └──────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

from thinking_engine import ThinkingEngine
from self_reflection import SelfReflectionEngine
from memory_system import MemorySystem
import time


def print_separator(title: str, char: str = "="):
    """打印分隔线"""
    print(f"\n{char * 70}")
    print(f"{title:^70}")
    print(f"{char * 70}")


def print_thoughts(thoughts):
    """打印思考过程"""
    print("\n【思考过程】")
    for i, thought in enumerate(thoughts, 1):
        print(f"{i:2d}. [{thought.thought_type:12s}] {thought.content}")


def print_reflection(reflection_result):
    """打印反思结果"""
    print("\n【反思评估结果】")
    for i, assessment in enumerate(reflection_result['assessments'], 1):
        print(f"{i:2d}. [{assessment['aspect']}] 评分: {assessment['rating']:.2f}")
        print(f"    └─ 评价: {assessment['comment']}")
        if assessment.get('suggestion'):
            print(f"    └─ 建议: {assessment['suggestion']}")
    
    if reflection_result['biases']:
        print("\n【识别到的认知偏差】")
        for i, bias in enumerate(reflection_result['biases'], 1):
            print(f"{i:2d}. [{bias['severity']}] {bias['name']}: {bias['description']}")
    
    if reflection_result['suggestions']:
        print("\n【改进建议】")
        for i, suggestion in enumerate(reflection_result['suggestions'], 1):
            print(f"{i:2d}. {suggestion}")
    
    print(f"\n【综合质量评分】: {reflection_result['overall_quality']:.2f}")


def print_memory_stats(memory):
    """打印记忆系统统计"""
    stats = memory.get_statistics()
    print(f"\n【记忆系统状态】")
    print(f"  长期记忆: {stats['long_term_count']} 条")
    print(f"  短期记忆: {stats['short_term_count']} 条")
    print(f"  总计: {stats['total_count']} 条")


def main():
    """主演示程序"""
    print_separator("思考与自我反思框架演示", "=")
    print("""
本演示展示了一个AI代理的完整思考与自我反思流程：

1. 思考阶段：接收问题，分析问题，生成解决方案，做出决策
2. 反思阶段：评估思考过程的质量，识别认知偏差，提出改进建议
3. 学习阶段：将经验存储到记忆系统，用于未来决策

让我们开始探索这个框架的能力！
    """)
    
    # 初始化各个模块
    print_separator("初始化系统模块", "-")
    
    # 创建记忆系统
    memory = MemorySystem()
    print("✓ 创建记忆系统")
    
    # 存储一些初始知识
    initial_knowledge = [
        {'content': '学习编程需要练习和耐心', 'memory_type': 'experience', 'confidence': 0.9},
        {'content': 'Python是一种流行的编程语言', 'memory_type': 'fact', 'confidence': 1.0},
        {'content': '坚持每天写代码可以提高编程能力', 'memory_type': 'skill', 'confidence': 0.85},
        {'content': '良好的代码注释有助于理解和维护', 'memory_type': 'fact', 'confidence': 0.95},
        {'content': '分步骤解决问题更容易成功', 'memory_type': 'experience', 'confidence': 0.9},
    ]
    
    for item in initial_knowledge:
        memory.store(item)
    print("✓ 加载初始知识库")
    
    # 创建思考引擎（连接记忆系统）
    thinker = ThinkingEngine(memory_system=memory)
    print("✓ 创建思考引擎（已连接记忆系统）")
    
    # 创建反思引擎（连接记忆系统）
    reflector = SelfReflectionEngine(memory_system=memory)
    print("✓ 创建反思引擎（已连接记忆系统）")
    
    print_memory_stats(memory)
    
    # 测试问题列表
    test_problems = [
        "如何学习编程？",
        "什么是人工智能？",
        "为什么需要自我反思？"
    ]
    
    # 处理每个问题
    for problem in test_problems:
        print_separator(f"处理问题: {problem}", "-")
        
        # 阶段1：思考
        print("\n【阶段1：思考】")
        print("开始分析问题...")
        thinking_result = thinker.think(problem)
        print_thoughts(thinking_result['thoughts'])
        
        print(f"\n【决策结论】")
        print(f"  结论: {thinking_result['conclusion']}")
        print(f"  置信度: {thinking_result['confidence']:.2f}")
        print(f"  推理: {thinking_result['reasoning']}")
        
        # 阶段2：反思
        print("\n【阶段2：自我反思】")
        print("开始评估思考过程...")
        reflection_result = reflector.reflect(thinking_result['thoughts'], outcome='成功')
        print_reflection(reflection_result)
        
        # 阶段3：学习（存储经验）
        print("\n【阶段3：学习与记忆】")
        experience = {
            'content': f"问题: {problem} -> 解决方案: {thinking_result['conclusion']} -> 质量评分: {reflection_result['overall_quality']:.2f}",
            'memory_type': 'experience',
            'confidence': reflection_result['overall_quality']
        }
        memory.store(experience)
        print(f"✓ 将本次经验存储到长期记忆")
        
        print_memory_stats(memory)
        
        # 模拟记忆巩固
        print("\n【执行记忆巩固】")
        consolidate_result = memory.consolidate()
        print(f"  巩固到长期记忆: {consolidate_result['consolidated']} 条")
        print(f"  遗忘低置信度记忆: {consolidate_result['forgotten']} 条")
        
        time.sleep(1)  # 稍作停顿，便于阅读
    
    # 测试记忆检索
    print_separator("测试记忆检索功能", "-")
    query = "编程"
    print(f"检索与 '{query}' 相关的知识：")
    results = memory.retrieve(query)
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. [{result['memory_type']}] {result['content']}")
        print(f"    匹配度: {result['match_score']:.2f} | 置信度: {result['confidence']:.2f}")
    
    # 保存记忆
    print_separator("保存记忆系统", "-")
    memory.save("thinking_memory.json")
    print("✓ 记忆系统已保存到 thinking_memory.json")
    
    print_separator("演示完成", "=")
    print("""
框架工作流程总结：

1. 思考引擎接收问题，进行分析、生成方案、评估并决策
2. 反思引擎对思考过程进行评估，识别偏差，提出改进建议
3. 记忆系统存储经验，用于未来决策
4. 记忆巩固机制将短期记忆转化为长期记忆

这个框架展示了AI代理如何通过"思考-反思-学习"循环不断提升决策质量。
    """)


if __name__ == "__main__":
    main()