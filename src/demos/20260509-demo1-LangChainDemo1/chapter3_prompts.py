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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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


def demo_message_types():
    """
    演示消息类型：SystemMessage, HumanMessage, AIMessage
    
    手动创建消息对象，提供更精细的控制
    """
    print("\n" + "=" * 60)
    print("演示 5: 消息类型 - SystemMessage, HumanMessage, AIMessage")
    print("=" * 60)
    
    logger.info("开始演示消息类型")
    
    try:
        messages = [
            SystemMessage(content="你是一个专业的Python编程导师，擅长用简单易懂的方式解释概念。"),
            HumanMessage(content="什么是列表推导式？请举例说明。")
        ]
        
        logger.info("创建消息对象:")
        for msg in messages:
            logger.info(f"  {type(msg).__name__}: {msg.content[:50]}...")
        
        print("\n消息列表:")
        print("-" * 60)
        for msg in messages:
            print(f"{type(msg).__name__}: {msg.content}")
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
        
        print("\n💡 知识点:")
        print("  - SystemMessage: 设置AI的角色和行为")
        print("  - HumanMessage: 用户的输入")
        print("  - AIMessage: AI的回复（用于多轮对话历史）")
        
        print(f"\n✅ 消息类型演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 消息类型演示失败: {e}", exc_info=True)
        print(f"\n❌ 消息类型演示失败: {e}")


def demo_multi_turn_conversation():
    """
    演示多轮对话场景
    
    使用 AIMessage 构建对话历史
    """
    print("\n" + "=" * 60)
    print("演示 6: 多轮对话 - 使用 AIMessage 构建对话历史")
    print("=" * 60)
    
    logger.info("开始演示多轮对话")
    
    try:
        conversation = [
            SystemMessage(content="你是一个友好的旅行顾问，专注于推荐亚洲旅游目的地。"),
            HumanMessage(content="我想去日本旅游，有什么推荐的地方吗？"),
            AIMessage(content="日本有很多值得一游的地方！我推荐东京、京都和大阪。东京有现代都市风光，京都有传统寺庙，大阪有美食文化。"),
            HumanMessage(content="那韩国呢？")
        ]
        
        logger.info("构建多轮对话历史:")
        for i, msg in enumerate(conversation, 1):
            logger.info(f"  {i}. {type(msg).__name__}: {msg.content[:50]}...")
        
        print("\n对话历史:")
        print("-" * 60)
        for i, msg in enumerate(conversation, 1):
            role = type(msg).__name__.replace("Message", "")
            print(f"{i}. [{role}] {msg.content}")
        print("-" * 60)
        
        llm = create_model(temperature=0.7)
        
        logger.info("调用模型...")
        start_time = time.time()
        response = llm.invoke(conversation)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        
        print("\n💡 知识点:")
        print("  - AIMessage 用于保存之前的对话历史")
        print("  - 模型可以基于完整上下文生成连贯的回复")
        print("  - 多轮对话需要维护消息列表")
        
        print(f"\n✅ 多轮对话演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 多轮对话演示失败: {e}", exc_info=True)
        print(f"\n❌ 多轮对话演示失败: {e}")


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
        ("消息类型", demo_message_types),
        ("多轮对话", demo_multi_turn_conversation),
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
