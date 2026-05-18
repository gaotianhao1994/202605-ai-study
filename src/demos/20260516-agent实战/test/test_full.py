import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.core_agent import SimpleAgent
from tools import calculator_tool, web_search_tool, read_file_tool

print("=" * 50)
print("测试3：完整功能测试")
print("=" * 50)

tools = [calculator_tool, web_search_tool, read_file_tool]
agent = SimpleAgent(tools=tools)

print(f"✓ Agent 初始化成功")
print(f"✓ 使用的模型: {os.getenv('OPENAI_MODEL_NAME', 'qwen3.5-122b-a10b')}")
print(f"✓ 可用工具: {[tool.name for tool in agent.tools]}")
print()

test_questions = [
    "你好，请介绍一下你自己",
    "用计算器帮我算一下 156 * 23 + 87",
    "你是用什么模型运行的？"
]

for i, question in enumerate(test_questions, 1):
    print(f"【测试问题 {i}】")
    print(f"用户: {question}")
    print()
    
    result = agent.run(question)
    print(f"Agent: {result}")
    print()
    print("-" * 50)

print()
print("=" * 50)
print("✓ 所有功能测试完成！")
print("=" * 50)