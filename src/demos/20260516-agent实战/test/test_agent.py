import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.core_agent import SimpleAgent
from tools import calculator_tool

print("=" * 50)
print("测试1：使用计算器工具")
print("=" * 50)

agent = SimpleAgent(tools=[calculator_tool])
print(f"✓ Agent 初始化成功")
print(f"✓ 使用的模型: {os.getenv('OPENAI_MODEL_NAME', 'qwen3.5-122b-a10b')}")
print(f"✓ 可用工具: {[tool.name for tool in agent.tools]}")
print()

result = agent.run("计算 25 * 4 + 18 / 3")
print(f"✓ 计算结果: {result}")
print()