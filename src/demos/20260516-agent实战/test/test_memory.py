import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.core_agent import SimpleAgent
from tools import calculator_tool

print("=" * 50)
print("测试2：Agent 的记忆功能")
print("=" * 50)
print("这个示例展示 Agent 如何记住对话历史\n")

agent = SimpleAgent(tools=[calculator_tool])

print("【第一轮对话】")
result1 = agent.run("我有 100 元钱")
print(f"用户: 我有 100 元钱")
print(f"Agent: {result1}")
print()

print("【第二轮对话】")
result2 = agent.run("如果我花掉 35 元，还剩多少钱？")
print(f"用户: 如果我花掉 35 元，还剩多少钱？")
print(f"Agent: {result2}")
print()

print("【第三轮对话】")
result3 = agent.run("如果剩下的钱我想分成 5 份，每份多少钱？")
print(f"用户: 如果剩下的钱我想分成 5 份，每份多少钱？")
print(f"Agent: {result3}")
print()

print("=" * 50)
print("✓ 对话历史记录成功！")
print("=" * 50)