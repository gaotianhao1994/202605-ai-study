"""
第六章：综合实战

学习目标：
- 综合运用所有知识
- 构建一个智能学习助手
- 学习扩展和优化方法

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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

logger = setup_logger('chapter6_integration')


class StudyAssistant:
    """
    智能学习助手类
    
    集成了模型调用、提示词模板、链式调用和记忆组件
    """
    
    def __init__(self, temperature: float = 0.7):
        """
        初始化学习助手
        
        Args:
            temperature: 创造性程度，范围 0-1
        """
        logger.info("初始化智能学习助手...")
        
        config = get_config()
        model_config = config.get_model_config()
        
        logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
        
        self.llm = ChatOpenAI(
            **model_config,
            temperature=temperature
        )
        
        self.memory = ConversationBufferMemory()
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        logger.info("智能学习助手初始化完成")
    
    def chat(self, user_input: str) -> str:
        """
        进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI 的回复
        """
        logger.info(f"用户输入: {user_input}")
        
        start_time = time.time()
        response = self.conversation.predict(input=user_input)
        elapsed_time = time.time() - start_time
        
        logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
        
        return response
    
    def clear_memory(self):
        """清空对话记忆"""
        logger.info("清空对话记忆")
        self.memory.clear()


def create_study_assistant(temperature: float = 0.7) -> StudyAssistant:
    """
    创建智能学习助手
    
    Args:
        temperature: 创造性程度
        
    Returns:
        StudyAssistant 实例
    """
    return StudyAssistant(temperature=temperature)


def test_study_assistant():
    """
    测试智能学习助手
    
    模拟多轮对话，测试助手的记忆和回答能力
    """
    print("\n" + "=" * 60)
    print("测试智能学习助手")
    print("=" * 60)
    
    logger.info("开始测试智能学习助手")
    
    try:
        assistant = create_study_assistant(temperature=0.7)
        
        print("\n✅ 智能学习助手已创建！")
        print("=" * 60)
        
        test_conversations = [
            "你好！我想学习 Python，应该从哪里开始？",
            "变量是什么？能举个例子吗？",
            "函数怎么定义？",
            "你刚才说的变量和函数有什么关系？"
        ]
        
        for i, user_input in enumerate(test_conversations, 1):
            print(f"\n【第 {i} 轮对话】")
            print(f"学生: {user_input}")
            
            response = assistant.chat(user_input)
            
            print(f"\n学习助手: {response}")
            print("-" * 60)
        
        print("\n✅ 智能学习助手测试完成！")
        print("\n💡 提示: 助手记住了整个对话历史，能够连贯地回答问题")
        
    except Exception as e:
        logger.error(f"❌ 智能学习助手测试失败: {e}", exc_info=True)
        print(f"\n❌ 智能学习助手测试失败: {e}")


def demo_full_integration():
    """
    完整集成演示
    
    展示如何将所有组件组合成完整应用
    """
    print("\n" + "=" * 60)
    print("完整集成演示")
    print("=" * 60)
    
    logger.info("开始完整集成演示")
    
    print("\n智能学习助手架构:")
    print("-" * 60)
    print("1. 配置管理 (config.py)")
    print("   - 加载环境变量")
    print("   - 提供 API 配置")
    print()
    print("2. 模型调用 (ChatOpenAI)")
    print("   - 连接阿里云百炼 API")
    print("   - 调用大语言模型")
    print()
    print("3. 记忆组件 (ConversationBufferMemory)")
    print("   - 保存对话历史")
    print("   - 实现多轮对话")
    print()
    print("4. 对话链 (ConversationChain)")
    print("   - 组合模型和记忆")
    print("   - 提供统一的对话接口")
    print("-" * 60)
    
    test_study_assistant()
    
    print("\n" + "=" * 60)
    print("扩展思路")
    print("=" * 60)
    
    print("\n功能扩展:")
    print("1. 添加知识库：使用 LangChain 的检索功能，让 AI 能查询特定文档")
    print("2. 多轮对话优化：使用更好的记忆策略，如摘要记忆")
    print("3. 个性化学习路径：根据学生的学习进度推荐内容")
    
    print("\n技术扩展:")
    print("1. 使用不同的 LLM：如 GPT-4、Claude 等")
    print("2. 添加工具调用：让 AI 能执行代码、搜索网络等")
    print("3. 部署为 Web 应用：使用 Streamlit、Gradio 等框架")
    
    logger.info("完整集成演示完成")


def main():
    """
    主函数：运行综合实战
    
    执行步骤：
    1. 创建智能学习助手
    2. 测试学习助手
    3. 展示扩展思路
    """
    print("=" * 60)
    print("🎯 第六章：综合实战")
    print("=" * 60)
    
    logger.info("开始第六章演示...")
    
    demo_full_integration()
    
    print("\n" + "=" * 60)
    print("🎉 第六章演示完成！")
    print("=" * 60)
    
    logger.info("第六章演示完成")


if __name__ == '__main__':
    main()
