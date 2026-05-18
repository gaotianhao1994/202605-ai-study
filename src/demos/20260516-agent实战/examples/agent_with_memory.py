import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core_agent import SimpleAgent
from tools import calculator_tool

def example_memory():
    print("=" * 50)
    print("示例：Agent 的记忆功能")
    print("=" * 50)
    print("这个示例展示 Agent 如何记住对话历史")
    print("=" * 50)
    
    agent = SimpleAgent(tools=[calculator_tool])
    
    print("第一轮对话：")
    result1 = agent.run("我有 100 元钱")
    print("Agent:", result1)
    print()
    
    print("第二轮对话（引用上一轮的信息）：")
    result2 = agent.run("如果我花掉 35 元，还剩多少钱？")
    print("Agent:", result2)
    print()
    
    print("第三轮对话（继续引用）：")
    result3 = agent.run("如果剩下的钱我想分成 5 份，每份多少钱？")
    print("Agent:", result3)
    print()
    
    print("查看对话历史：")
    history = agent.get_chat_history()
    print(history)

if __name__ == "__main__":
    example_memory()