import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core_agent import SimpleAgent
from tools import calculator_tool, web_search_tool, read_file_tool

def example1_calculator():
    print("=" * 50)
    print("示例1：使用计算器工具")
    print("=" * 50)
    
    agent = SimpleAgent(tools=[calculator_tool])
    result = agent.run("计算 25 * 4 + 18 / 3")
    print("结果:", result)
    print()

def example2_web_search():
    print("=" * 50)
    print("示例2：使用网页搜索工具")
    print("=" * 50)
    
    agent = SimpleAgent(tools=[web_search_tool])
    result = agent.run("搜索一下 LangChain 的最新版本")
    print("搜索结果:", result)
    print()

def example3_read_file():
    print("=" * 50)
    print("示例3：使用文件读取工具")
    print("=" * 50)
    
    agent = SimpleAgent(tools=[read_file_tool])
    result = agent.run(f"读取文件 {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/requirements.txt")
    print("文件内容:", result)
    print()

def example4_combined_tools():
    print("=" * 50)
    print("示例4：组合使用多个工具")
    print("=" * 50)
    
    tools = [calculator_tool, web_search_tool, read_file_tool]
    agent = SimpleAgent(tools=tools)
    
    print("Agent 已配置以下工具:", [tool.name for tool in agent.tools])
    
    while True:
        user_input = input("请输入问题（输入 'exit' 退出）: ")
        if user_input.lower() == 'exit':
            break
        
        result = agent.run(user_input)
        print("Agent 回答:", result)
        print()

if __name__ == "__main__":
    example1_calculator()
    example2_web_search()
    example3_read_file()
    example4_combined_tools()