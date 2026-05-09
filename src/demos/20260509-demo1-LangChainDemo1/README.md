# LangChain 入门教程（改进版）

基于阿里云百炼大模型的 LangChain 入门教程，提供完整的错误处理和日志记录功能。

## 项目简介

本项目是 LangChain 入门教程的改进版本，主要改进包括：

- ✅ **技术栈变更**：使用阿里云百炼大模型替代 OpenAI API
- ✅ **文件格式变更**：从 Jupyter Notebook (.ipynb) 转换为标准 Python 文件 (.py)
- ✅ **模块化设计**：按章节拆分为独立的 Python 模块
- ✅ **完整的错误处理**：所有代码包含详细的异常处理和友好提示
- ✅ **详细的日志记录**：记录每个步骤的执行情况，便于调试和学习

## 快速开始

### 环境要求

- Python >= 3.8
- 项目根目录的 `requirements.txt` 已包含所需依赖

### 安装步骤

1. **安装依赖**（在项目根目录）

```bash
cd /0-gaoth/projects/202605/202605-ai-study
pip install -r requirements.txt
```

2. **配置环境变量**

```bash
cd src/demos/20260509-demo1-LangChainDemo1
cp .env.example .env
```

编辑 `.env` 文件，填入你的阿里云百炼 API 配置：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen3.5-122b-a10b
```

### 运行方式

#### 方式 1：交互式模式

```bash
python main.py
```

显示章节菜单，选择要运行的章节。

#### 方式 2：运行指定章节

```bash
python main.py --chapter 1  # 运行第一章
python main.py --chapter 2  # 运行第二章
python main.py -c 3         # 简写形式
```

#### 方式 3：运行所有章节

```bash
python main.py --all
```

#### 方式 4：直接运行章节文件

```bash
python chapter1_env_setup.py  # 运行第一章
python chapter2_model_io.py   # 运行第二章
python chapter3_prompts.py    # 运行第三章
python chapter4_chains.py     # 运行第四章
python chapter5_memory.py     # 运行第五章
python chapter6_integration.py # 运行第六章
```

## 学习路径

### 第一章：环境准备与快速开始（10-15 分钟）

**学习目标**：
- 了解 LangChain 是什么
- 检查环境配置
- 运行第一个 LangChain 程序

**核心内容**：
- Python 版本检查
- 依赖安装检查
- API Key 配置验证
- 第一个简单调用

### 第二章：模型调用（Model I/O）（20-25 分钟）

**学习目标**：
- 学习 LLM 和 Chat Model 的区别
- 掌握不同的调用方式
- 理解模型参数的作用

**核心内容**：
- invoke() - 单次调用
- batch() - 批量调用
- stream() - 流式输出
- temperature 参数对比

### 第三章：提示词模板（20-25 分钟）

**学习目标**：
- 理解模板的作用
- 学习模板的基本使用
- 掌握模板组合技巧

**核心内容**：
- PromptTemplate - 简单字符串模板
- ChatPromptTemplate - 聊天消息模板
- 翻译助手模板实战
- 代码解释器模板实战

### 第四章：链式调用（25-30 分钟）

**学习目标**：
- 理解链的概念
- 学习 LCEL 语法
- 构建简单和复杂的处理链

**核心内容**：
- LCEL 语法（| 操作符）
- 简单链：prompt | llm
- 完整链：prompt | llm | parser
- 文章摘要生成链
- 多语言翻译链

### 第五章：记忆组件（20-25 分钟）

**学习目标**：
- 理解为什么需要记忆
- 学习不同的记忆类型
- 实现多轮对话

**核心内容**：
- ConversationBufferMemory - 完整记忆
- ConversationBufferWindowMemory - 窗口记忆
- 多轮对话演示
- 聊天机器人实战

### 第六章：综合实战（30-40 分钟）

**学习目标**：
- 综合运用所有知识
- 构建一个智能学习助手
- 学习扩展和优化方法

**核心内容**：
- 完整应用架构
- 组件集成
- 多轮对话测试
- 功能扩展思路

## 项目结构

```
20260509-demo1-LangChainDemo1/
├── README.md                    # 项目说明文档
├── .env.example                 # 环境变量模板
├── .env                         # 实际环境变量（不提交到git）
├── config.py                    # 配置管理模块
├── logger.py                    # 日志配置模块
├── main.py                      # 主程序入口
├── chapter1_env_setup.py        # 第一章：环境准备
├── chapter2_model_io.py         # 第二章：模型调用
├── chapter3_prompts.py          # 第三章：提示词模板
├── chapter4_chains.py           # 第四章：链式调用
├── chapter5_memory.py           # 第五章：记忆组件
└── chapter6_integration.py      # 第六章：综合实战
```

## 常见问题

### 1. 配置问题

**Q: 提示 ".env 文件不存在"**

A: 请复制 `.env.example` 为 `.env` 并填写配置：
```bash
cp .env.example .env
```

**Q: 提示 "OPENAI_API_KEY 未配置"**

A: 请在 `.env` 文件中设置 `OPENAI_API_KEY`，从阿里云百炼控制台获取。

### 2. 运行错误

**Q: 提示 "ModuleNotFoundError: No module named 'langchain'"**

A: 请在项目根目录安装依赖：
```bash
pip install -r requirements.txt
```

**Q: API 调用失败**

A: 可能的原因：
1. API Key 无效
2. 网络连接问题
3. API 服务不可用

请检查：
- API Key 是否正确
- 网络连接是否正常
- API 服务是否可用

### 3. API 问题

**Q: 如何获取阿里云百炼 API Key？**

A: 
1. 访问阿里云百炼控制台
2. 创建应用并获取 API Key
3. 将 API Key 填入 `.env` 文件

**Q: 支持哪些模型？**

A: 本教程使用 `qwen3.5-122b-a10b` 模型，你也可以在阿里云百炼控制台查看其他可用模型。

## 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [项目源码](https://github.com/langchain-ai/langchain)

## 学习建议

1. **按顺序学习**：章节之间有递进关系，建议按顺序学习
2. **动手实践**：每个章节都有演示代码，务必动手运行
3. **修改参数**：尝试修改代码中的参数，观察输出变化
4. **记录问题**：遇到不理解的地方，记录下来，可以反复阅读

## 许可证

本项目仅供学习使用。

---

祝你学习愉快！🚀
