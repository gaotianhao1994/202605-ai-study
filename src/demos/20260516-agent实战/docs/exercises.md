# 练习题目

## 一、基础练习

### 练习1：运行现有示例

**目标**：熟悉 Agent 的基本用法

**步骤**：
1. 激活虚拟环境
2. 安装依赖
3. 运行 `python examples/basic_usage.py`
4. 观察输出，理解 Agent 的思考过程

**问题**：
- Agent 在调用工具前会输出什么？
- `verbose=True` 的作用是什么？

---

### 练习2：测试计算器工具

**目标**：理解工具调用机制

**步骤**：
1. 直接运行 `python tools/calculator.py`
2. 修改 `basic_usage.py`，测试不同的数学表达式

**测试用例**：
- `(100 - 25) * 4`
- `3.14 * 5^2`
- `100 / (2 + 3)`

**问题**：如果输入非法表达式会怎样？

---

### 练习3：测试记忆功能

**目标**：理解对话记忆的作用

**步骤**：
1. 运行 `python examples/agent_with_memory.py`
2. 观察 Agent 如何记住之前的对话

**问题**：
- Agent 是如何记住"我有 100 元钱"的？
- 如果去掉 `memory` 参数，会发生什么？

---

## 二、进阶练习

### 练习4：创建新工具

**目标**：学会自定义工具

**步骤**：
1. 在 `tools/` 目录下创建新文件 `date_tool.py`
2. 实现一个获取当前日期的工具
3. 在 `tools/__init__.py` 中导出新工具
4. 修改示例代码，使用新工具

**参考代码结构**：
```python
from langchain.tools import Tool
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

date_tool = Tool(
    name="get_current_date",
    func=get_current_date,
    description="用于获取当前日期和时间。当用户询问现在的时间时使用此工具。"
)
```

---

### 练习5：修改 Agent 配置

**目标**：理解 Agent 参数的作用

**步骤**：
1. 修改 `core_agent.py` 中的 `temperature` 参数
2. 测试不同值（0.1, 0.5, 1.0）的效果
3. 观察输出的变化

**问题**：
- `temperature=0` 时输出有什么特点？
- `temperature=1` 时输出有什么特点？
- 你认为什么场景适合高温度？什么场景适合低温度？

---

### 练习6：添加更多工具

**目标**：扩展 Agent 的能力

**步骤**：
1. 创建一个天气查询工具（可以使用模拟数据）
2. 创建一个翻译工具
3. 将这些工具添加到 Agent 中

**提示**：可以使用免费的天气 API 或创建模拟数据

---

## 三、挑战练习

### 练习7：实现代码生成 Agent

**目标**：创建一个能编写代码的 Agent

**要求**：
1. 创建一个代码编写工具
2. Agent 可以根据用户需求生成 Python 代码
3. 代码可以保存到文件中

**思路**：
- 使用 `write_file` 工具保存生成的代码
- 让 Agent 先分析需求，再生成代码

---

### 练习8：多 Agent 协作

**目标**：创建多个 Agent 协作完成任务

**要求**：
1. 创建一个"任务分配 Agent"
2. 根据任务类型分配给不同的专业 Agent
3. 例如：数学任务交给"计算 Agent"，写作任务交给"写作 Agent"

**思路**：
- 主 Agent 负责分析任务类型
- 根据类型调用不同的子 Agent
- 汇总结果返回给用户

---

## 四、思考问题

1. Agent 和普通的 Chatbot 有什么区别？
2. 在什么情况下 Agent 需要调用工具？
3. 如果工具返回错误，Agent 应该如何处理？
4. 你觉得 Agent 未来可以应用在哪些场景？

---

## 五、扩展阅读

- LangChain 官方文档：https://python.langchain.com/
- OpenAI API 文档：https://platform.openai.com/docs/
- Agent 相关论文："Chain of Thought"、"ReAct"