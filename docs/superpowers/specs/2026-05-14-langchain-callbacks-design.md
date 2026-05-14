# LangChain Callbacks 回调系统学习项目 — 设计文档

## 概述

在 `src/demos/` 目录下创建一个新的学习项目 `20260514-demo2-LangChainCallbacks`，聚焦于 LangChain 的 Callbacks 回调系统，采用 5 章渐进式教学结构。

## 项目信息

- **文件夹名称**: `20260514-demo2-LangChainCallbacks`
- **路径**: `src/demos/20260514-demo2-LangChainCallbacks/`
- **命名规范**: 遵循 `YYYYMMDD-demoN-TopicName` 格式
- **技术栈**: 阿里云百炼 API + langchain + langchain-openai

## 项目结构

```
20260514-demo2-LangChainCallbacks/
├── README.md                         # 项目说明文档
├── .env.example                      # 环境变量模板
├── config.py                         # 配置管理模块
├── logger.py                         # 日志配置模块
├── main.py                           # 主程序入口（交互式菜单）
├── chapter1_callback_concepts.py     # 第一章：回调系统概念与环境准备
├── chapter2_builtin_handlers.py      # 第二章：内置回调处理器
├── chapter3_custom_handler.py        # 第三章：自定义回调处理器
├── chapter4_async_callbacks.py       # 第四章：异步回调与多回调组合
└── chapter5_integration.py           # 第五章：综合实战
```

## 章节设计

### 第一章：回调系统概念与环境准备

**学习目标**:
- 理解回调机制的概念与必要性
- 理解回调与直接调用的本质区别
- 搭建环境并运行第一个带回调的程序

**核心内容**:
- 回调是什么："你先做，做完告诉我"的模式
- 为什么需要回调：LLM 调用耗时长、不可预测，需要在不同阶段执行自定义逻辑
- 追问：为什么不直接在调用前后加代码？— 回调是"非侵入式"的，把"做什么"和"什么时候做"解耦
- 回调生命周期图解：`on_llm_start → on_llm_new_token → on_llm_end / on_llm_error`
- 环境搭建与第一个回调示例：用 `StdOutCallbackHandler` 观察调用过程
- 对比：不带回调 vs 带回调的调用差异

### 第二章：内置回调处理器

**学习目标**:
- 掌握 LangChain 内置回调处理器
- 理解不同处理器的适用场景

**核心内容**:
- `StdOutCallbackHandler` — 控制台输出，调试利器
- `FileCallbackHandler` — 日志写入文件，持久化记录
- 回调配置方式：构造函数传入 `callbacks` 参数 vs `with_config()` 方法
- 回调传播机制：父组件的回调如何自动传播给子组件
- 实战：用内置回调处理器监控链式调用的完整执行流程

### 第三章：自定义回调处理器

**学习目标**:
- 掌握继承 `BaseCallbackHandler` 创建自定义处理器
- 理解各生命周期方法的触发时机和参数
- 实现日志、监控、审计等常见场景

**核心内容**:
- `BaseCallbackHandler` 核心方法：
  - `on_llm_start / on_llm_end / on_llm_error` — LLM 调用生命周期
  - `on_chain_start / on_chain_end / on_chain_error` — 链式调用生命周期
  - `on_tool_start / on_tool_end / on_tool_error` — 工具调用生命周期
  - `on_llm_new_token` — 流式输出的每个 token
- 追问：为什么每个组件类型都有独立的回调方法？— 不同组件的输入输出结构不同，独立方法提供类型安全的参数
- 实战示例：
  - `TimingCallbackHandler` — 记录每次 LLM 调用的耗时
  - `TokenCountCallbackHandler` — 统计 Token 使用量
  - `AuditLogCallbackHandler` — 审计日志

### 第四章：异步回调与多回调组合

**学习目标**:
- 理解同步回调 vs 异步回调的区别
- 掌握 `AsyncCallbackHandler` 的使用
- 学会组合多个回调处理器

**核心内容**:
- 为什么需要异步回调：同步回调会阻塞事件循环
- 追问：为什么阻塞事件循环是问题？— 事件循环是异步程序的"心脏"，阻塞它等于冻结整个应用
- `AsyncCallbackHandler` 与 `BaseCallbackHandler` 的方法对应关系
- 多回调组合：同时使用计时、计数、日志三个处理器
- `with_config(callbacks=[...])` 链式配置
- 回调优先级与执行顺序

### 第五章：综合实战

**学习目标**:
- 综合运用所有回调知识
- 构建完整的 LLM 调用监控与审计系统
- 学习扩展和优化方法

**核心内容**:
- 构建监控审计系统：
  - `MetricsCallback` — 性能指标收集（延迟、Token 数、成功率）
  - `AlertCallback` — 异常告警（超时、错误率过高）
  - `LoggingCallback` — 结构化日志记录
- 将回调集成到链式调用中
- 运行多场景测试并展示监控结果
- 扩展思路：对接 Prometheus/Grafana、接入分布式追踪

## 代码风格与约定

- 使用阿里云百炼 API（`ChatOpenAI` + `openai_api_base`）
- 每个章节文件包含 `main()` 函数和 `if __name__ == '__main__'` 入口
- 完整的错误处理（try/except + logger）
- 中文注释和输出
- `config.py` 和 `logger.py` 复用已有模式
- 遵循第一性原理思维：每个知识点包含"是什么→为什么→追问"

## 依赖

- 使用项目根目录已有的 `requirements.txt`，无需新增依赖
- `langchain` 和 `langchain-openai` 已包含回调系统所需的所有模块
