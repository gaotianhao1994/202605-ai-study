"""
第五章：记忆组件（Memory）

学习目标：
- 理解为什么需要记忆
- 学习不同的记忆类型
- 实现多轮对话

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
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain

logger = setup_logger('chapter5_memory')


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


def demo_buffer_memory():
    """
    演示 ConversationBufferMemory
    
    保存完整的对话历史
    """
    print("\n" + "=" * 60)
    print("演示 1: ConversationBufferMemory - 完整记忆")
    print("=" * 60)
    
    logger.info("开始演示 ConversationBufferMemory")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferMemory()
        
        logger.debug("创建对话链，使用 ConversationBufferMemory")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        conversations = [
            ("你好，我叫小明", "第一轮对话"),
            ("你还记得我的名字吗？", "第二轮对话")
        ]
        
        for user_input, round_name in conversations:
            print(f"\n{'=' * 60}")
            print(f"{round_name}")
            print("=" * 60)
            
            logger.info(f"用户输入: {user_input}")
            print(f"用户: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
        
        print(f"\n✅ ConversationBufferMemory 演示成功！")
        print("\n💡 提示: AI 记住了第一轮对话中的名字")
        
    except Exception as e:
        logger.error(f"❌ ConversationBufferMemory 演示失败: {e}", exc_info=True)
        print(f"\n❌ ConversationBufferMemory 演示失败: {e}")


def demo_window_memory():
    """
    演示 ConversationBufferWindowMemory
    
    只保存最近 N 轮对话
    """
    print("\n" + "=" * 60)
    print("演示 2: ConversationBufferWindowMemory - 窗口记忆")
    print("=" * 60)
    
    logger.info("开始演示 ConversationBufferWindowMemory")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferWindowMemory(k=2)
        
        logger.debug("创建对话链，使用 ConversationBufferWindowMemory(k=2)")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        conversations = [
            ("我喜欢吃苹果", "第一轮对话"),
            ("我喜欢吃香蕉", "第二轮对话"),
            ("我喜欢吃什么水果？", "第三轮对话")
        ]
        
        for user_input, round_name in conversations:
            print(f"\n{'=' * 60}")
            print(f"{round_name}")
            print("=" * 60)
            
            logger.info(f"用户输入: {user_input}")
            print(f"用户: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
        
        print(f"\n✅ ConversationBufferWindowMemory 演示成功！")
        print("\n💡 提示: AI 只记得最近 2 轮对话")
        
    except Exception as e:
        logger.error(f"❌ ConversationBufferWindowMemory 演示失败: {e}", exc_info=True)
        print(f"\n❌ ConversationBufferWindowMemory 演示失败: {e}")


def demo_chatbot():
    """
    演示有记忆的聊天机器人
    
    创建一个能记住对话历史的聊天机器人
    """
    print("\n" + "=" * 60)
    print("演示 3: 有记忆的聊天机器人")
    print("=" * 60)
    
    logger.info("开始演示有记忆的聊天机器人")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferMemory()
        
        logger.debug("创建聊天机器人")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        print("\n聊天机器人已启动！")
        print("=" * 60)
        
        test_inputs = [
            "你好，我想学习 Python",
            "你能推荐一些学习资源吗？",
            "谢谢！我刚才说我想学什么？"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n【第 {i} 轮对话】")
            print(f"用户: {user_input}")
            
            logger.info(f"用户输入: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
            print("-" * 60)
        
        print("\n✅ 聊天机器人演示成功！")
        print("\n💡 提示: AI 记住了整个对话历史")
        
    except Exception as e:
        logger.error(f"❌ 聊天机器人演示失败: {e}", exc_info=True)
        print(f"\n❌ 聊天机器人演示失败: {e}")


def main():
    """
    主函数：运行所有记忆组件演示
    
    执行顺序：
    1. ConversationBufferMemory 演示
    2. ConversationBufferWindowMemory 演示
    3. 有记忆的聊天机器人演示
    """
    print("=" * 60)
    print("💾 第五章：记忆组件（Memory）")
    print("=" * 60)
    
    logger.info("开始第五章演示...")
    
    demos = [
        ("ConversationBufferMemory", demo_buffer_memory),
        ("ConversationBufferWindowMemory", demo_window_memory),
        ("有记忆的聊天机器人", demo_chatbot),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第五章演示完成！")
    print("=" * 60)
    
    logger.info("第五章演示完成")


if __name__ == '__main__':
    main()
