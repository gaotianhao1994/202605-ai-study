"""
第三章：提示词模板（Prompts）

学习目标：
- 理解模板的作用
- 学习模板的基本使用
- 掌握模板组合技巧

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
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

logger = setup_logger('chapter3_prompts')


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


def demo_prompt_template():
    """
    演示 PromptTemplate
    
    简单字符串模板的使用
    """
    print("\n" + "=" * 60)
    print("演示 1: PromptTemplate - 简单字符串模板")
    print("=" * 60)
    
    logger.info("开始演示 PromptTemplate")
    
    try:
        template = "请用一句话介绍{subject}"
        
        logger.debug(f"创建模板: {template}")
        prompt = PromptTemplate.from_template(template)
        
        final_prompt = prompt.format(subject="Python")
        
        logger.info(f"生成的提示词: {final_prompt}")
        print(f"\n模板: {template}")
        print(f"变量: subject='Python'")
        print(f"\n生成的提示词:")
        print("-" * 60)
        print(final_prompt)
        print("-" * 60)
        
        llm = create_model(temperature=0.7)
        
        logger.info("调用模型...")
        start_time = time.time()
        response = llm.invoke(final_prompt)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ PromptTemplate 演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ PromptTemplate 演示失败: {e}", exc_info=True)
        print(f"\n❌ PromptTemplate 演示失败: {e}")


def demo_chat_prompt_template():
    """
    演示 ChatPromptTemplate
    
    聊天消息模板的使用，包含系统消息和用户消息
    """
    print("\n" + "=" * 60)
    print("演示 2: ChatPromptTemplate - 聊天消息模板")
    print("=" * 60)
    
    logger.info("开始演示 ChatPromptTemplate")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}，请用专业但易懂的语言回答问题。"),
            ("user", "{question}")
        ])
        
        logger.debug(f"创建聊天模板，包含系统消息和用户消息")
        
        messages = prompt.format_messages(
            role="Python 专家",
            question="什么是装饰器？"
        )
        
        logger.info("生成的消息:")
        for msg in messages:
            logger.info(f"  {msg.type}: {msg.content}")
        
        print("\n模板消息:")
        print("-" * 60)
        for msg in messages:
            print(f"{msg.type}: {msg.content}")
        print("-" * 60)
        
        llm = create_model(temperature=0.7)
        
        logger.info("调用模型...")
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ ChatPromptTemplate 演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ ChatPromptTemplate 演示失败: {e}", exc_info=True)
        print(f"\n❌ ChatPromptTemplate 演示失败: {e}")


def demo_translation_template():
    """
    演示翻译助手模板
    
    创建一个可以翻译多种语言的模板
    """
    print("\n" + "=" * 60)
    print("演示 3: 翻译助手模板")
    print("=" * 60)
    
    logger.info("开始演示翻译助手模板")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的翻译助手。"),
            ("user", "请将以下{source_lang}翻译成{target_lang}：{text}")
        ])
        
        logger.debug("创建翻译模板")
        
        test_cases = [
            {
                "source_lang": "英文",
                "target_lang": "中文",
                "text": "Hello, World!"
            },
            {
                "source_lang": "中文",
                "target_lang": "英文",
                "text": "你好，世界！"
            }
        ]
        
        llm = create_model(temperature=0.3)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"翻译案例 {i}")
            print("=" * 60)
            
            messages = prompt.format_messages(**case)
            
            logger.info(f"翻译: {case['source_lang']} -> {case['target_lang']}")
            print(f"\n原文 ({case['source_lang']}): {case['text']}")
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            logger.info(f"翻译成功，耗时 {elapsed_time:.2f}s")
            
            print(f"\n译文 ({case['target_lang']}):")
            print("-" * 60)
            print(response.content)
            print("-" * 60)
        
        print("\n✅ 翻译助手模板演示成功！")
        
    except Exception as e:
        logger.error(f"❌ 翻译助手模板演示失败: {e}", exc_info=True)
        print(f"\n❌ 翻译助手模板演示失败: {e}")


def demo_code_explainer():
    """
    演示代码解释器模板
    
    创建一个可以解释代码的模板
    """
    print("\n" + "=" * 60)
    print("演示 4: 代码解释器模板")
    print("=" * 60)
    
    logger.info("开始演示代码解释器模板")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{language}专家，请用简单易懂的语言解释代码。"),
            ("user", "请解释以下代码：\n{code}")
        ])
        
        logger.debug("创建代码解释模板")
        
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        messages = prompt.format_messages(
            language="Python",
            code=code
        )
        
        print(f"\n代码:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        
        llm = create_model(temperature=0.3)
        
        logger.info("调用模型解释代码...")
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        logger.info(f"代码解释成功，耗时 {elapsed_time:.2f}s")
        
        print("\n代码解释:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 代码解释器模板演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 代码解释器模板演示失败: {e}", exc_info=True)
        print(f"\n❌ 代码解释器模板演示失败: {e}")


def main():
    """
    主函数：运行所有提示词模板演示
    
    执行顺序：
    1. PromptTemplate 演示
    2. ChatPromptTemplate 演示
    3. 翻译助手模板演示
    4. 代码解释器模板演示
    """
    print("=" * 60)
    print("📝 第三章：提示词模板（Prompts）")
    print("=" * 60)
    
    logger.info("开始第三章演示...")
    
    demos = [
        ("PromptTemplate", demo_prompt_template),
        ("ChatPromptTemplate", demo_chat_prompt_template),
        ("翻译助手模板", demo_translation_template),
        ("代码解释器模板", demo_code_explainer),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第三章演示完成！")
    print("=" * 60)
    
    logger.info("第三章演示完成")


if __name__ == '__main__':
    main()
