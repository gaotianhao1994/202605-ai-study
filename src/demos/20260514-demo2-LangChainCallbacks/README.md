# LangChain Callbacks 回调系统教程

基于阿里云百炼大模型的 LangChain Callbacks 回调系统教程，提供完整的错误处理和日志记录功能。

## 项目简介

本项目聚焦于 LangChain 的 Callbacks 回调系统，采用渐进式教学结构，从概念到实战，帮助你掌握回调机制的核心知识。

- ✅ **聚焦回调系统**：深入讲解 Callbacks 机制的概念、原理和应用
- ✅ **渐进式教学**：从概念 → 内置处理器 → 自定义处理器 → 异步回调 → 综合实战
- ✅ **第一性原理**：每个知识点都包含"是什么→为什么→追问"的深度思考
- ✅ **完整错误处理**：所有代码包含详细的异常处理和友好提示
- ✅ **详细的日志记录**：记录每个步骤的执行情况，便于调试和学习

## 快速开始

### 环境要求

- Python >= 3.8
- 项目根目录的 `requirements.txt` 已包含所需依赖

### 安装步骤

1. **安装依赖**（在项目根目录）

```bash
pip install -r requirements.txt
```

2. **配置环境变量**

```bash
cd src/demos/20260514-demo2-LangChainCallbacks
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
python chapter1_callback_concepts.py  # 第一章：回调概念
python chapter2_builtin_handlers.py   # 第二章：内置处理器
python chapter3_custom_handler.py     # 第三章：自定义处理器
python chapter4_async_callbacks.py    # 第四章：异步回调
python chapter5_integration.py        # 第五章：综合实战
```

## 学习路径

### 第一章：回调系统概念与环境准备（15-20 分钟）

**学习目标**：
- 理解回调机制的概念与必要性
- 理解回调与直接调用的本质区别
- 搭建环境并运行第一个带回调的程序

**核心内容**：
- 回调是什么："你先做，做完告诉我"的模式
- 回调的生命周期：on_llm_start → on_llm_end / on_llm_error
- 不带回调 vs 带回调的调用对比

### 第二章：内置回调处理器（20-25 分钟）

**学习目标**：
- 掌握 LangChain 内置回调处理器
- 理解不同处理器的适用场景

**核心内容**：
- StdOutCallbackHandler — 控制台输出
- 回调配置方式：callbacks 参数 vs with_config() 方法
- 回调传播机制：父组件的回调自动传播给子组件

### 第三章：自定义回调处理器（25-30 分钟）

**学习目标**：
- 掌握继承 BaseCallbackHandler 创建自定义处理器
- 理解各生命周期方法的触发时机和参数
- 实现日志、监控、审计等常见场景

**核心内容**：
- BaseCallbackHandler 核心方法详解
- TimingCallbackHandler — 调用计时
- TokenCountCallbackHandler — Token 计数
- AuditLogCallbackHandler — 审计日志

### 第四章：异步回调与多回调组合（20-25 分钟）

**学习目标**：
- 理解同步回调 vs 异步回调的区别
- 掌握 AsyncCallbackHandler 的使用
- 学会组合多个回调处理器

**核心内容**：
- AsyncCallbackHandler 异步回调
- 多回调组合：计时 + 计数 + 日志
- with_config() 链式配置

### 第五章：综合实战（30-40 分钟）

**学习目标**：
- 综合运用所有回调知识
- 构建完整的 LLM 调用监控与审计系统
- 学习扩展和优化方法

**核心内容**：
- MetricsCallback — 性能指标收集
- AlertCallback — 异常告警
- LoggingCallback — 结构化日志
- 扩展思路：Prometheus、OpenTelemetry、成本控制

## 项目结构

```
20260514-demo2-LangChainCallbacks/
├── README.md                         # 项目说明文档
├── .env.example                      # 环境变量模板
├── config.py                         # 配置管理模块
├── logger.py                         # 日志配置模块
├── main.py                           # 主程序入口
├── chapter1_callback_concepts.py     # 第一章：回调概念
├── chapter2_builtin_handlers.py      # 第二章：内置处理器
├── chapter3_custom_handler.py        # 第三章：自定义处理器
├── chapter4_async_callbacks.py       # 第四章：异步回调
└── chapter5_integration.py           # 第五章：综合实战
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

### 3. 回调相关问题

**Q: 回调没有被触发？**

A: 请检查：
1. 回调是否正确传入 `config={"callbacks": [...]}` 参数
2. 回调方法名是否正确（注意大小写）
3. 是否使用了正确的回调基类（同步 vs 异步）

**Q: 异步回调报错？**

A: 请确保：
1. 使用 `ainvoke` 而非 `invoke` 进行异步调用
2. 继承 `AsyncCallbackHandler` 而非 `BaseCallbackHandler`
3. 在 async 函数中运行异步代码

## 相关资源

- [LangChain 官方文档 - Callbacks](https://python.langchain.com/docs/concepts/callbacks/)
- [LangChain 官方文档](https://python.langchain.com/)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)

## 学习建议

1. **按顺序学习**：章节之间有递进关系，建议按顺序学习
2. **动手实践**：每个章节都有演示代码，务必动手运行
3. **修改参数**：尝试修改回调处理器中的参数，观察输出变化
4. **自己实现**：尝试实现一个新的回调处理器，加深理解

## 许可证

本项目仅供学习使用。

---

祝你学习愉快！
