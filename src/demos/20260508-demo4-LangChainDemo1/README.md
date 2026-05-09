# LangChain 入门教程

## 📖 项目简介

这是一个为 Python 初学者设计的 LangChain 入门教程项目。通过 Jupyter Notebook 的形式，带你从零开始学习 LangChain 的核心功能。

### 你将学到什么？

- ✅ 如何调用 OpenAI 等 LLM 模型
- ✅ 如何使用提示词模板
- ✅ 如何构建链式调用
- ✅ 如何实现对话记忆功能
- ✅ 如何组合这些组件构建完整应用

## 🚀 快速开始

### 前置要求

- Python 3.8 或更高版本
- OpenAI API Key（需要注册 OpenAI 账号）

### 安装步骤

1. **克隆或下载项目**
   ```bash
   cd /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1
   ```

2. **安装依赖**
   ```bash
   pip install -r ../../../requirements.txt
   ```

3. **配置环境变量**
   ```bash
   # 复制环境变量模板
   cp .env.example .env
   
   # 编辑 .env 文件，填入你的 OpenAI API Key
   # OPENAI_API_KEY=sk-your-api-key-here
   ```

4. **启动 Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **打开教程**
   在浏览器中打开 `LangChain入门教程.ipynb`

## 📚 学习路径

### 第一章：环境准备与快速开始（10-15 分钟）
- 了解 LangChain 是什么
- 检查环境配置
- 运行第一个 LangChain 程序

### 第二章：模型调用（20-25 分钟）
- 学习 LLM 和 Chat Model 的区别
- 掌握不同的调用方式
- 理解模型参数的作用

### 第三章：提示词模板（20-25 分钟）
- 理解模板的作用
- 学习模板的基本使用
- 掌握模板组合技巧

### 第四章：链式调用（25-30 分钟）
- 理解链的概念
- 学习 LCEL 语法
- 构建简单和复杂的处理链

### 第五章：记忆组件（20-25 分钟）
- 理解为什么需要记忆
- 学习不同的记忆类型
- 实现多轮对话

### 第六章：综合实战（30-40 分钟）
- 综合运用所有知识
- 构建一个智能学习助手
- 学习扩展和优化方法

**总计学习时间：约 2-3 小时**

## 🎯 学习建议

1. **按顺序学习**：章节之间有递进关系，建议按顺序学习
2. **动手实践**：每个章节都有练习题，务必动手完成
3. **修改参数**：尝试修改代码中的参数，观察输出变化
4. **记录问题**：遇到不理解的地方，记录下来，可以反复阅读

## ❓ 常见问题

### Q1: 提示 "API key not found" 错误？
**A**: 检查 `.env` 文件是否正确配置，确保 `OPENAI_API_KEY` 前面没有空格。

### Q2: 运行代码时没有响应？
**A**: 可能是网络问题或 API 调用超时，请检查网络连接，或稍后重试。

### Q3: 如何获取 OpenAI API Key？
**A**: 
1. 访问 https://platform.openai.com/
2. 注册并登录账号
3. 在 API keys 页面创建新的 Key

### Q4: 调用 API 需要付费吗？
**A**: 是的，OpenAI API 是付费服务。新账号通常有免费额度，建议先用于学习。

## 📦 项目结构

```
20260508-demo4-LangChainDemo1/
├── README.md                      # 本文件
├── .env.example                   # 环境变量模板
├── LangChain入门教程.ipynb        # 主教程
└── images/                        # 教程图片
```

## 🔗 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

## 📝 许可证

本项目仅供学习使用。
