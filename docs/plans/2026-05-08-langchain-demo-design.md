# LangChain 入门教程项目设计文档

## 项目概述

### 目标
创建一个通俗易懂的 LangChain 演示项目，帮助 Python 初学者理解 LangChain 的基本概念和核心功能。

### 目标用户
- Python 初学者
- 刚接触 AI 开发的学习者
- 需要详细解释和逐步引导的用户

### 核心功能演示
1. 模型调用（Model I/O）
2. 提示词模板（Prompts）
3. 链式调用（Chains）
4. 记忆组件（Memory）

## 项目结构

```
20260508-demo4-LangChainDemo1/
├── README.md                      # 项目说明、快速开始指南
├── .env.example                   # 环境变量配置示例
├── LangChain入门教程.ipynb        # 主教程 Notebook（核心）
└── images/                        # 教程中使用的图片（可选）
    └── langchain-architecture.png
```

### 依赖管理
使用项目根目录的 `requirements.txt`，添加 jupyter 依赖：
```
langchain==1.2.17
langchain-openai>=0.1.0
python-dotenv>=1.0.0
tiktoken>=0.6.0
jupyter>=1.0.0
```

### LLM 服务
使用 OpenAI API，需要在 `.env` 文件中配置 `OPENAI_API_KEY`。

## Notebook 内容结构

### 第一章：环境准备与快速开始（10-15 分钟）
**学习目标**：让用户快速跑通第一个 LangChain 程序

**内容**：
1. 什么是 LangChain？
   - 简单比喻：LangChain 就像是"AI 应用的乐高积木"
   - 核心价值：简化 LLM 应用开发
   
2. 环境检查
   - 检查 Python 版本
   - 检查依赖是否安装
   - 检查 API Key 是否配置

3. 第一个 LangChain 程序
   ```python
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI()
   response = llm.invoke("你好，请用一句话介绍 LangChain")
   print(response.content)
   ```

4. 运行结果展示和代码解释

### 第二章：模型调用（Model I/O）（20-25 分钟）
**学习目标**：掌握 LangChain 中模型调用的各种方式

**内容**：
1. 基础概念
   - LLM vs Chat Model 的区别
   - 为什么 LangChain 使用 Chat Model？

2. 创建模型实例
   - 基本创建方式
   - 常用参数（temperature, model_name 等）
   - 参数对输出的影响

3. 调用方式
   - `invoke()` - 单次调用
   - `batch()` - 批量调用
   - `stream()` - 流式输出

4. 实践练习
   - 练习1：调整 temperature 参数
   - 练习2：使用批量调用生成创意标题

### 第三章：提示词模板（20-25 分钟）
**学习目标**：学会使用模板创建可复用的提示词

**内容**：
1. 为什么需要提示词模板？
   - 硬编码提示词的缺点
   - 模板的优势

2. 基础模板使用
   - `PromptTemplate` - 简单字符串模板
   - `ChatPromptTemplate` - 聊天消息模板
   - 变量替换

3. 模板组合
   - 系统消息 + 用户消息
   - 多个模板的组合使用

4. 实践练习
   - 练习1：创建"翻译助手"模板
   - 练习2：创建"代码解释器"模板

### 第四章：链式调用（25-30 分钟）
**学习目标**：理解如何将多个组件串联起来

**内容**：
1. 什么是链？
   - 比喻：链就像流水线
   - LCEL 简介

2. 简单链
   - 使用 `|` 操作符连接组件
   - Prompt → Model → Output 流程

3. 复杂链
   - 多步骤处理链
   - 条件分支链

4. 实践练习
   - 练习1：创建"文章摘要生成链"
   - 练习2：创建"多语言翻译链"

### 第五章：记忆组件（20-25 分钟）
**学习目标**：让 AI 记住对话历史

**内容**：
1. 为什么需要记忆？
   - LLM 的无状态特性
   - 记忆组件的作用

2. 记忆类型
   - `ConversationBufferMemory` - 完整记忆
   - `ConversationBufferWindowMemory` - 窗口记忆

3. 在链中使用记忆
   - 将记忆集成到对话链
   - 多轮对话示例

4. 实践练习
   - 练习：创建能记住上下文的聊天机器人

### 第六章：综合实战（30-40 分钟）
**学习目标**：将所有组件组合成完整应用

**内容**：
1. 项目目标
   - 构建"智能学习助手"
   - 功能：回答问题、记住对话、提供个性化建议

2. 项目实现
   - 设计提示词模板
   - 创建模型实例
   - 添加记忆组件
   - 组装成完整链

3. 测试和优化
   - 运行测试对话
   - 观察记忆效果
   - 调整参数优化

4. 扩展思路
   - 如何添加更多功能
   - 如何部署应用

## 代码风格和注释规范

### 注释风格
1. **代码块注释** - 在代码前用中文说明目的
2. **行内注释** - 解释关键代码行
3. **知识点说明** - 在 Markdown 单元格中详细解释
4. **错误处理示例** - 展示常见错误和解决方法
5. **输出结果展示** - 在代码后展示预期输出

### Markdown 单元格风格
- 章节标题使用 emoji 图标
- 重点提示使用 `> 💡 **提示**` 格式
- 警告信息使用 `> ⚠️ **注意**` 格式

## README.md 内容
- 项目简介和目标
- 前置要求
- 安装步骤（详细）
- 学习路径说明
- 常见问题解答
- 相关资源链接

## 实现优先级
1. 更新项目根目录的 requirements.txt
2. 创建项目目录结构
3. 创建 .env.example 文件
4. 创建 README.md 文件
5. 创建主教程 Notebook（LangChain入门教程.ipynb）
6. 安装依赖并测试

## 成功标准
- 用户能够按照 README 中的步骤成功运行项目
- Notebook 中的所有代码都能正常运行
- 注释清晰易懂，适合初学者
- 学习路径循序渐进，符合初学者认知
