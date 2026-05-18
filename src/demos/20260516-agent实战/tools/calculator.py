from langchain.tools import Tool
import math
import re

def calculate(expression):
    try:
        expression = expression.strip()
        pattern = r'^[0-9+\-*/().\s]+$'
        if not re.match(pattern, expression):
            return "错误：表达式只能包含数字、运算符(+、-、*、/)和括号"
        
        result = eval(expression)
        return f"计算结果: {result}"
    except ZeroDivisionError:
        return "错误：不能除以零"
    except SyntaxError:
        return "错误：表达式语法不正确"
    except Exception as e:
        return f"计算错误: {str(e)}"

calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description="用于执行数学计算。输入应该是一个数学表达式，例如 '2 + 3 * 4' 或 '(5 + 8) / 2'。当你需要计算数值时使用此工具。"
)

if __name__ == "__main__":
    print(calculator_tool.run("2 + 3 * 4"))
    print(calculator_tool.run("(10 - 3) * 2"))