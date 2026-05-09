# LangChain 入门教程项目实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个通俗易懂的 LangChain 演示项目，帮助 Python 初学者学习 LangChain 的核心功能。

**Architecture:** 使用 Jupyter Notebook 作为主要教学载体，按照渐进式学习路径组织内容。项目包含环境配置、模型调用、提示词模板、链式调用、记忆组件和综合实战六个章节。

**Tech Stack:** Python 3.8+, LangChain 1.2.17, OpenAI API, Jupyter Notebook

---

## 前置条件

- 项目根目录：`/0-gaoth/projects/202605/202605-ai-study`
- 目标目录：`/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1`
- 已有依赖文件：`/0-gaoth/projects/202605/202605-ai-study/requirements.txt`

---

### Task 1: 更新项目依赖

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/requirements.txt`

**Step 1: 读取当前 requirements.txt**

Read the file to see current content.

**Step 2: 添加 Jupyter 依赖**

在文件末尾添加：
```
# Jupyter Notebook for tutorials
jupyter>=1.0.0
```

**Step 3: 安装依赖**

Run: `pip install -r /0-gaoth/projects/202605/202605-ai-study/requirements.txt`
Expected: Successfully installed jupyter and dependencies

---

### Task 2: 创建项目目录结构

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/` (directory)
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/images/` (directory)

**Step 1: 创建主项目目录**

Run: `mkdir -p /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/images`
Expected: Directory created successfully

**Step 2: 验证目录创建**

Run: `ls -la /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1`
Expected: Directory listing with images subdirectory

---

### Task 3: 创建环境变量模板文件

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/.env.example`

**Step 1: 创建 .env.example 文件**

```bash
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here
```

**Step 2: 验证文件创建**

Run: `cat /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/.env.example`
Expected: File content displayed

---

### Task 4: 创建 README.md 文件

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/README.md`

**Step 1: 创建 README.md 文件**

内容见设计文档中的 README.md 部分，包含：
- 项目简介
- 快速开始
- 学习路径
- 常见问题
- 项目结构
- 相关资源

**Step 2: 验证文件创建**

Run: `head -20 /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/README.md`
Expected: First 20 lines of README displayed

---

### Task 5: 创建主教程 Notebook（第一章：环境准备）

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 创建 Notebook 文件结构**

创建 Jupyter Notebook 文件，包含：
- 元数据（kernel spec, language info）
- 第一章的 Markdown 和 Code 单元格

**Step 2: 添加第一章内容**

第一章包含：
- 标题和简介
- 环境检查代码
- 第一个 LangChain 程序
- 运行结果展示

**Step 3: 验证 Notebook 创建**

Run: `jupyter nbconvert --to script /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`
Expected: Notebook converted to Python script successfully

---

### Task 6: 添加第二章内容（模型调用）

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 添加第二章 Markdown 单元格**

包含：
- 章节标题
- 学习目标
- 基础概念说明

**Step 2: 添加模型调用示例代码**

包含：
- 创建模型实例
- 不同调用方式（invoke, batch, stream）
- 参数调整示例

**Step 3: 添加实践练习**

包含：
- 练习1：调整 temperature
- 练习2：批量调用

---

### Task 7: 添加第三章内容（提示词模板）

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 添加第三章 Markdown 单元格**

包含：
- 章节标题
- 学习目标
- 模板概念说明

**Step 2: 添加模板使用示例代码**

包含：
- PromptTemplate 使用
- ChatPromptTemplate 使用
- 变量替换示例

**Step 3: 添加实践练习**

包含：
- 练习1：翻译助手模板
- 练习2：代码解释器模板

---

### Task 8: 添加第四章内容（链式调用）

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 添加第四章 Markdown 单元格**

包含：
- 章节标题
- 学习目标
- 链的概念说明

**Step 2: 添加链式调用示例代码**

包含：
- 简单链示例
- 复杂链示例
- LCEL 语法演示

**Step 3: 添加实践练习**

包含：
- 练习1：文章摘要生成链
- 练习2：多语言翻译链

---

### Task 9: 添加第五章内容（记忆组件）

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 添加第五章 Markdown 单元格**

包含：
- 章节标题
- 学习目标
- 记忆组件概念说明

**Step 2: 添加记忆组件示例代码**

包含：
- ConversationBufferMemory 使用
- ConversationBufferWindowMemory 使用
- 多轮对话示例

**Step 3: 添加实践练习**

包含：
- 练习：创建有记忆的聊天机器人

---

### Task 10: 添加第六章内容（综合实战）

**Files:**
- Modify: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/LangChain入门教程.ipynb`

**Step 1: 添加第六章 Markdown 单元格**

包含：
- 章节标题
- 学习目标
- 项目目标说明

**Step 2: 添加综合实战代码**

包含：
- 智能学习助手完整实现
- 所有组件的组合使用

**Step 3: 添加测试和优化说明**

包含：
- 测试对话示例
- 优化建议
- 扩展思路

---

### Task 11: 创建 .env 文件并测试

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/.env`

**Step 1: 创建 .env 文件**

使用用户提供的 API Key：
```
OPENAI_API_KEY=sk-5e4ce55385c042acbd69d58a7bb9a230
```

**Step 2: 启动 Jupyter Notebook**

Run: `cd /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1 && jupyter notebook`
Expected: Jupyter Notebook server starts

**Step 3: 测试第一个示例**

在 Notebook 中运行第一章的代码，验证：
- 环境检查通过
- 第一个 LangChain 程序运行成功
- 输出结果符合预期

---

### Task 12: 最终验证和文档完善

**Files:**
- Verify: All files in `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/`

**Step 1: 验证所有文件存在**

Run: `ls -la /0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1`
Expected: All files listed:
- README.md
- .env.example
- .env
- LangChain入门教程.ipynb
- images/

**Step 2: 验证 Notebook 结构**

检查 Notebook 是否包含所有六个章节。

**Step 3: 验证代码可运行**

在 Jupyter Notebook 中逐个运行所有代码单元格，确保：
- 没有语法错误
- 所有导入成功
- API 调用正常
- 输出符合预期

---

## 成功标准

- ✅ 所有文件创建成功
- ✅ 依赖安装成功
- ✅ Jupyter Notebook 可以正常启动
- ✅ Notebook 中所有代码可以运行
- ✅ 注释清晰易懂，适合初学者
- ✅ 学习路径循序渐进
- ✅ README 文档完整

## 注意事项

1. **API Key 安全**：.env 文件不应提交到版本控制
2. **代码注释**：所有代码都需要详细的中文注释
3. **错误处理**：每个示例都应包含错误处理和提示
4. **输出展示**：每个代码块后都应展示预期输出
5. **渐进式学习**：章节之间有递进关系，不要跳跃式讲解
