"""
第四章：链式调用（Chains）

学习目标：
- 理解链的概念
- 学习 LCEL 语法
- 构建简单和复杂的处理链

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter4_chains')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info(f"模型实例创建成功")
    return llm


def demo_simple_chain():
    """
    演示简单链
    
    使用 LCEL 语法（| 操作符）连接组件
    """
    print("\n" + "=" * 60)
    print("演示 1: 简单链 - LCEL 语法")
    print("=" * 60)
    
    logger.info("开始演示简单链")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
        
        logger.debug("创建链: prompt | llm")
        chain = prompt | llm
        
        logger.info("调用链，topic='程序员'")
        print("\n链的组成: prompt | llm")
        print("\n输入: {'topic': '程序员'}")
        
        start_time = time.time()
        response = chain.invoke({"topic": "程序员"})
        elapsed_time = time.time() - start_time
        
        logger.info(f"链调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 简单链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 简单链演示失败: {e}", exc_info=True)
        print(f"\n❌ 简单链演示失败: {e}")


def demo_chain_with_parser():
    """
    演示带输出解析器的链
    
    添加 StrOutputParser 将输出转换为字符串
    """
    print("\n" + "=" * 60)
    print("演示 2: 带输出解析器的链")
    print("=" * 60)
    
    logger.info("开始演示带输出解析器的链")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
        output_parser = StrOutputParser()
        
        logger.debug("创建链: prompt | llm | output_parser")
        chain = prompt | llm | output_parser
        
        logger.info("调用链，topic='人工智能'")
        print("\n链的组成: prompt | llm | output_parser")
        print("\n输入: {'topic': '人工智能'}")
        
        start_time = time.time()
        result = chain.invoke({"topic": "人工智能"})
        elapsed_time = time.time() - start_time
        
        logger.info(f"链调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答（字符串格式）:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        print(f"\n✅ 带输出解析器的链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 带输出解析器的链演示失败: {e}", exc_info=True)
        print(f"\n❌ 带输出解析器的链演示失败: {e}")


def demo_summary_chain():
    """
    演示文章摘要生成链
    
    创建一个可以生成文章摘要的链
    """
    print("\n" + "=" * 60)
    print("演示 3: 文章摘要生成链")
    print("=" * 60)
    
    logger.info("开始演示文章摘要生成链")
    
    try:
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请为以下文章生成一个简洁的摘要（不超过50字）：\n\n{article}"
        )
        
        logger.debug("创建摘要生成链")
        chain = prompt | llm | StrOutputParser()
        
        article = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
这些任务包括学习、推理、问题解决、感知和语言理解。AI 技术已经广泛应用于各个领域，
包括医疗诊断、金融分析、自动驾驶汽车和智能家居设备。
        """
        
        logger.info("调用摘要生成链")
        print("\n原文:")
        print("-" * 60)
        print(article)
        print("-" * 60)
        
        start_time = time.time()
        summary = chain.invoke({"article": article})
        elapsed_time = time.time() - start_time
        
        logger.info(f"摘要生成成功，耗时 {elapsed_time:.2f}s")
        
        print("\n生成的摘要:")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        print(f"\n✅ 文章摘要生成链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 文章摘要生成链演示失败: {e}", exc_info=True)
        print(f"\n❌ 文章摘要生成链演示失败: {e}")


def demo_translation_chain():
    """
    演示多语言翻译链
    
    创建一个可以翻译多种语言的链
    """
    print("\n" + "=" * 60)
    print("演示 4: 多语言翻译链")
    print("=" * 60)
    
    logger.info("开始演示多语言翻译链")
    
    try:
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请将以下文本翻译成{target_language}：\n\n{text}"
        )
        
        logger.debug("创建翻译链")
        chain = prompt | llm | StrOutputParser()
        
        test_cases = [
            {"target_language": "英文", "text": "你好，世界！"},
            {"target_language": "日文", "text": "你好，世界！"}
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"翻译案例 {i}")
            print("=" * 60)
            
            logger.info(f"翻译到 {case['target_language']}")
            print(f"\n原文: {case['text']}")
            print(f"目标语言: {case['target_language']}")
            
            start_time = time.time()
            result = chain.invoke(case)
            elapsed_time = time.time() - start_time
            
            logger.info(f"翻译成功，耗时 {elapsed_time:.2f}s")
            
            print(f"\n译文:")
            print("-" * 60)
            print(result)
            print("-" * 60)
        
        print("\n✅ 多语言翻译链演示成功！")
        
    except Exception as e:
        logger.error(f"❌ 多语言翻译链演示失败: {e}", exc_info=True)
        print(f"\n❌ 多语言翻译链演示失败: {e}")


def main():
    """
    主函数：运行所有链式调用演示
    
    执行顺序：
    1. 简单链演示
    2. 带输出解析器的链演示
    3. 文章摘要生成链演示
    4. 多语言翻译链演示
    """
    print("=" * 60)
    print("🔗 第四章：链式调用（Chains）")
    print("=" * 60)
    
    logger.info("开始第四章演示...")
    
    demos = [
        ("简单链", demo_simple_chain),
        ("带输出解析器的链", demo_chain_with_parser),
        ("文章摘要生成链", demo_summary_chain),
        ("多语言翻译链", demo_translation_chain),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第四章演示完成！")
    print("=" * 60)
    
    logger.info("第四章演示完成")


if __name__ == '__main__':
    main()
