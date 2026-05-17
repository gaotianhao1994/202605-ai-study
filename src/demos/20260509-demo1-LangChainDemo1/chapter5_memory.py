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
from langchain_core.messages import AIMessage, HumanMessage

# ============================================================================
# 关于 Memory API 的说明
# ============================================================================
#
# ⚠️ 注意：langchain.memory 模块（如 ConversationBufferMemory）是旧版 API
#    虽然仍可用，但 LangChain 1.x 推荐使用更简洁的方式管理对话历史
#
# ✅ 推荐方式：使用 langchain_core.messages 手动管理消息列表
#    - 更透明：你完全控制消息的添加和删除
#    - 更灵活：可以自定义记忆策略（窗口、摘要等）
#    - 更简洁：不需要额外的 Memory 类
#
# 示例：
#   messages = []
#   messages.append(HumanMessage(content="你好"))
#   response = llm.invoke(messages)
#   messages.append(AIMessage(content=response.content))
#
# 对比旧版 API：
#   from langchain.memory import ConversationBufferMemory  # 不推荐
#   memory = ConversationBufferMemory(return_messages=True)
#   memory.save_context({"input": "你好"}, {"output": "你好！"})
#
# ============================================================================

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
    演示完整记忆 - 保存所有对话历史
    """
    print("\n" + "=" * 60)
    print("演示 1: 完整记忆 - 保存所有对话")
    print("=" * 60)
    
    logger.info("开始演示完整记忆")
    
    try:
        llm = create_model(temperature=0.7)
        
        messages = []
        
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
            
            messages.append(HumanMessage(content=user_input))
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            messages.append(AIMessage(content=response.content))
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response.content}")
        
        print(f"\n✅ 完整记忆演示成功！")
        print("\n💡 提示: AI 记住了第一轮对话中的名字")
        
    except Exception as e:
        logger.error(f"❌ 完整记忆演示失败: {e}", exc_info=True)
        print(f"\n❌ 完整记忆演示失败: {e}")


def demo_window_memory():
    """
    演示窗口记忆 - 只保存最近 N 轮对话
    """
    print("\n" + "=" * 60)
    print("演示 2: 窗口记忆 - 限制历史消息数量")
    print("=" * 60)
    
    logger.info("开始演示窗口记忆")
    
    try:
        llm = create_model(temperature=0.7)
        
        messages = []
        max_history = 2
        
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
            
            messages.append(HumanMessage(content=user_input))
            
            if len(messages) > max_history * 2:
                messages = messages[-max_history * 2:]
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            messages.append(AIMessage(content=response.content))
            
            if len(messages) > max_history * 2:
                messages = messages[-max_history * 2:]
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response.content}")
        
        print(f"\n✅ 窗口记忆演示成功！")
        print(f"\n💡 提示: AI 只记得最近 {max_history} 轮对话")
        
    except Exception as e:
        logger.error(f"❌ 窗口记忆演示失败: {e}", exc_info=True)
        print(f"\n❌ 窗口记忆演示失败: {e}")


def demo_chatbot():
    """
    演示有记忆的聊天机器人
    """
    print("\n" + "=" * 60)
    print("演示 3: 有记忆的聊天机器人")
    print("=" * 60)
    
    logger.info("开始演示有记忆的聊天机器人")
    
    try:
        llm = create_model(temperature=0.7)
        
        messages = []
        
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
            
            messages.append(HumanMessage(content=user_input))
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            messages.append(AIMessage(content=response.content))
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response.content}")
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
    1. 完整记忆演示
    2. 窗口记忆演示
    3. 有记忆的聊天机器人演示
    """
    print("=" * 60)
    print("💾 第五章：记忆组件（Memory）- 新版 API")
    print("=" * 60)
    
    logger.info("开始第五章演示...")
    
    demos = [
        ("完整记忆", demo_buffer_memory),
        ("窗口记忆", demo_window_memory),
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