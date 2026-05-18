# LangChain Agent 实战教程

## 一、什么是 Agent？

### 1.1 直觉理解

想象一下，你有一个聪明的助手：
- 它能理解你的问题
- 它知道什么时候需要使用计算器
- 它知道什么时候需要上网查资料
- 它能记住你们之前的对话

这个助手就是 **Agent**！

### 1.2 专业定义

Agent 是一个能够：
1. **感知**（Perceive）：理解用户的问题
2. **决策**（Decide）：决定是否需要使用工具
3. **执行**（Act）：调用工具获取信息
4. **反思**（Reflect）：根据工具结果给出最终回答

### 1.3 Agent 的工作流程

```
用户提问 → Agent分析 → 决定是否使用工具 → 执行工具 → 总结回答
                          ↓
                     直接回答（如果不需要工具）
```

---

## 二、项目结构

```
20260516-agent实战/
├── .env.example          # 环境变量模板
├── requirements.txt      # 依赖清单
├── agent/
│   └── core_agent.py     # 核心 Agent 实现
├── tools/
│   ├── __init__.py       # 工具导出
│   ├── calculator.py     # 计算器工具
│   ├── web_search.py     # 网页搜索工具
│   └── file_tools.py     # 文件读写工具
├── examples/
│   ├── basic_usage.py    # 基础用法示例
│   └── agent_with_memory.py  # 记忆功能示例
└── docs/
    └── tutorial.md       # 本教程
```

---

## 三、核心组件解析

### 3.1 SimpleAgent 类

`SimpleAgent` 是我们的核心类，包含以下关键组件：

#### 3.1.1 LLM（大语言模型）
```python
self.llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
```

**关键点：**
- `model_name`：指定使用的模型
- `temperature`：控制输出的随机性（0=确定性，1=随机性）
- `api_key`：你的 OpenAI API 密钥

#### 3.1.2 Memory（记忆）
```python
self.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

**关键点：**
- `ConversationBufferMemory`：简单的对话历史存储
- Agent 可以记住之前的对话内容

#### 3.1.3 Tools（工具）
```python
self.tools = tools or []
```

**关键点：**
- 工具是 Agent 可以调用的外部能力
- 我们已经实现了计算器、搜索、文件读写工具

#### 3.1.4 Agent 初始化
```python
self.agent = initialize_agent(
    tools=self.tools,
    llm=self.llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=self.memory,
    verbose=True
)
```

**关键点：**
- `AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION`：支持对话和工具调用
- `verbose=True`：打印详细的思考过程

---

## 四、工具系统

### 4.1 什么是工具？

工具是 Agent 可以调用的函数，用于完成特定任务。

### 4.2 工具的结构

每个工具需要三个要素：
1. **name**：工具名称（用于 Agent 识别）
2. **func**：工具函数（实际执行的逻辑）
3. **description**：工具描述（告诉 Agent 何时使用此工具）

### 4.3 现有工具

| 工具名称 | 功能 | 使用场景 |
|---------|------|---------|
| `calculator` | 数学计算 | 需要计算数值时 |
| `web_search` | 网页搜索 | 需要最新信息时 |
| `read_file` | 读取文件 | 需要查看文件内容时 |
| `write_file` | 写入文件 | 需要保存内容时 |

---

## 五、动手实践

### 5.1 准备工作

1. 安装依赖：
```bash
cd /root/projects/202605-ai-study/src/demos/20260516-agent实战
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key
```

### 5.2 运行示例

```bash
# 运行基础示例
python examples/basic_usage.py

# 运行记忆功能示例
python examples/agent_with_memory.py
```

### 5.3 观察输出

当运行示例时，你会看到类似这样的输出：
```
> Entering new AgentExecutor chain...
> I need to calculate 25 * 4 + 18 / 3. I'll use the calculator tool.
> Action: calculator
> Action Input: 25 * 4 + 18 / 3
> Observation: 计算结果: 106.0
> Thought: 计算完成，我现在可以回答用户的问题了。
> Final Answer: 计算结果是 106.0
> 
> 结果: 计算结果是 106.0
```

---

## 六、知识关联

```
Agent
├── LLM (大语言模型) ── 理解问题、生成回答
├── Memory (记忆) ── 保存对话历史
└── Tools (工具)
    ├── Calculator ── 数学计算
    ├── Web Search ── 获取最新信息
    └── File Tools ── 文件读写
```

---

## 七、常见问题

### Q1：为什么需要 Agent？
A：单纯的 LLM 有知识截止日期，且不擅长精确计算。Agent 通过调用工具来弥补这些不足。

### Q2：Agent 是如何决定使用哪个工具的？
A：LLM 根据工具的 `description` 字段来判断何时使用哪个工具。

### Q3：Memory 的作用是什么？
A：Memory 让 Agent 能够进行多轮对话，记住之前的上下文信息。

---

## 八、练习题目

### 基础练习
1. 修改 `temperature` 参数，观察输出有什么变化
2. 尝试让 Agent 计算更复杂的数学表达式

### 进阶练习
1. 创建一个新工具（例如：日期工具、天气工具）
2. 修改 Agent 的提示词，让它更符合你的需求

### 挑战练习
1. 实现一个能够自动编写 Python 代码的 Agent
2. 创建一个多 Agent 协作系统