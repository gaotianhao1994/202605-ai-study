# 项目重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 对项目目录架构进行系统性重构，使其更加合理、清晰且具备良好的可维护性，同时保留所有demo的日期信息。

**Architecture:** 
1. 创建统一的 `src/demos/` 目录存放所有demo
2. 使用 `YYYYMMDD_name` 命名规范组织demo目录
3. 清理未使用的依赖库
4. 更新配置文件和文档

**Tech Stack:** Python 3.11, Git, pip

---

## 目录命名规范

| 规则 | 说明 | 示例 |
|------|------|------|
| 日期格式 | `YYYYMMDD_`（8位日期+下划线） | `20260508_` |
| 位置 | 日期前缀放在目录名称最前方 | `20260508_rag_splitter/` |
| 命名风格 | 目录名称使用小写蛇形命名法 | `20260507_thinking_reflection/` |

---

## 任务列表

### Task 1: 创建目录结构

**Files:**
- Create: `src/demos/`
- Create: `src/common/`
- Create: `src/api/`
- Create: `src/config/`
- Create: `tests/`
- Create: `frontend/`
- Create: `docs/plans/`
- Create: `notebooks/`
- Create: `data/`

**Step 1: 创建目录结构**

```bash
mkdir -p src/demos src/common src/api src/config tests frontend docs/plans notebooks data
```

**Step 2: 创建必要的 __init__.py 文件**

```bash
touch src/__init__.py src/demos/__init__.py src/common/__init__.py src/api/__init__.py src/config/__init__.py tests/__init__.py
```

**Step 3: 验证目录结构**

Run: `find src -type d | head -20`
Expected: 显示创建的目录结构

**Step 4: Commit**

```bash
git add src/ tests/ frontend/ docs/ notebooks/ data/
git commit -m "feat: create project directory structure"
```

---

### Task 2: 迁移 RAG 文档分割器 (20260508)

**Files:**
- Move: `20260508-demo-rag-splitter/src/` → `src/demos/20260508_rag_splitter/src/`
- Move: `20260508-demo-rag-splitter/demo.py` → `src/demos/20260508_rag_splitter/demo.py`
- Move: `20260508-demo-rag-splitter/data/` → `src/demos/20260508_rag_splitter/data/`
- Move: `20260508-demo-rag-splitter/README.md` → `src/demos/20260508_rag_splitter/README.md`

**Step 1: 创建目标目录**

```bash
mkdir -p src/demos/20260508_rag_splitter/src/splitters src/demos/20260508_rag_splitter/src/utils src/demos/20260508_rag_splitter/data
```

**Step 2: 复制文件**

```bash
cp -r 20260508-demo-rag-splitter/src/splitters/* src/demos/20260508_rag_splitter/src/splitters/
cp -r 20260508-demo-rag-splitter/src/utils/* src/demos/20260508_rag_splitter/src/utils/
cp 20260508-demo-rag-splitter/src/__init__.py src/demos/20260508_rag_splitter/src/
cp 20260508-demo-rag-splitter/demo.py src/demos/20260508_rag_splitter/
cp 20260508-demo-rag-splitter/data/sample_document.txt src/demos/20260508_rag_splitter/data/
cp 20260508-demo-rag-splitter/README.md src/demos/20260508_rag_splitter/
```

**Step 3: 更新 demo.py 中的导入路径**

```python
# 修改导入语句
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.splitters import (
    CharacterSplitter,
    TokenSplitter,
    SemanticSplitter,
    SentenceSplitter,
    read_file,
    print_chunk_info,
    display_chunks,
    compare_splitters
)
```

**Step 4: 更新 src/__init__.py 中的导入路径**

```python
# 修改相对导入为绝对导入
from .splitters import (
    CharacterSplitter,
    TokenSplitter,
    SemanticSplitter,
    SentenceSplitter,
)
from .utils import (
    read_file,
    print_chunk_info,
    display_chunks,
    compare_splitters,
)
```

**Step 5: 验证功能**

Run: `cd src/demos/20260508_rag_splitter && python demo.py`
Expected: 演示正常运行

**Step 6: Commit**

```bash
git add src/demos/20260508_rag_splitter/
git commit -m "feat: migrate RAG splitter demo"
```

---

### Task 3: 迁移思维反射框架 (20260507)

**Files:**
- Move: `20260507-demo02-thinking-reflection-framework/` → `src/demos/20260507_thinking_reflection/`

**Step 1: 创建目标目录**

```bash
mkdir -p src/demos/20260507_thinking_reflection
```

**Step 2: 复制文件**

```bash
cp 20260507-demo02-thinking-reflection-framework/*.py src/demos/20260507_thinking_reflection/
cp 20260507-demo02-thinking-reflection-framework/*.json src/demos/20260507_thinking_reflection/
cp 20260507-demo02-thinking-reflection-framework/*.md src/demos/20260507_thinking_reflection/
```

**Step 3: 更新导入路径**

```python
# 在 demo.py 中修改导入
from thinking_engine import ThinkingEngine
from self_reflection import SelfReflectionEngine
from memory_system import MemorySystem
```

**Step 4: 验证功能**

Run: `cd src/demos/20260507_thinking_reflection && python demo.py`
Expected: 演示正常运行

**Step 5: Commit**

```bash
git add src/demos/20260507_thinking_reflection/
git commit -m "feat: migrate thinking reflection framework"
```

---

### Task 4: 迁移简单 demo (20260507, 20260508)

**Files:**
- Move: `20260507-demo/` → `src/demos/20260507_demo/`
- Move: `20260508-demo/` → `src/demos/20260508_demo/`

**Step 1: 创建目标目录**

```bash
mkdir -p src/demos/20260507_demo src/demos/20260508_demo
```

**Step 2: 复制文件**

```bash
cp 20260507-demo/*.py src/demos/20260507_demo/
cp 20260508-demo/*.py src/demos/20260508_demo/
```

**Step 3: 验证功能**

Run: `python src/demos/20260508_demo/hello_world.py`
Expected: 输出 "Hello World"

**Step 4: Commit**

```bash
git add src/demos/20260507_demo/ src/demos/20260508_demo/
git commit -m "feat: migrate simple demos"
```

---

### Task 5: 清理未使用的依赖

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

**Step 1: 分析当前依赖使用情况**

根据代码扫描，项目实际使用的依赖：
- tiktoken (必需)
- sentence-transformers (必需)
- numpy (必需)

**Step 2: 更新 requirements.txt**

```txt
# Core dependencies
tiktoken>=0.6.0
sentence-transformers>=2.7.0
numpy>=1.26.0

# Development dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

**Step 3: 更新 pyproject.toml**

```toml
[project]
name = "ai-study-demos"
version = "1.0.0"
description = "AI学习演示项目集合"
requires-python = ">=3.8"
authors = [
    {name = "Developer", email = "developer@example.com"}
]
dependencies = [
    "tiktoken>=0.6.0",
    "sentence-transformers>=2.7.0",
    "numpy>=1.26.0",
]

[tool.setuptools]
package-dir = {"": "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]

[tool.setuptools.package-data]
"*" = ["*.txt", "*.md", "*.json"]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

**Step 4: 验证配置**

Run: `pip check`
Expected: 无错误输出

**Step 5: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "feat: clean up unused dependencies"
```

---

### Task 6: 更新根目录 README.md

**Files:**
- Modify: `README.md`

**Step 1: 更新项目说明**

```markdown
# AI学习演示项目

## 项目简介

这是一个用于 AI 学习和研究的项目仓库，包含各种 AI 相关的演示程序和实验代码。

## 目录结构

```
.
├── src/                    # 主源码目录
│   ├── demos/              # Demo示例集合
│   │   ├── 20260507_demo/              # 2026年5月7日创建的演示
│   │   ├── 20260507_thinking_reflection/ # 思维反射框架
│   │   ├── 20260508_demo/              # 2026年5月8日创建的演示
│   │   └── 20260508_rag_splitter/      # RAG文档分割器
│   ├── common/             # 通用工具模块
│   ├── api/                # API服务（预留）
│   └── config/             # 配置管理
├── tests/                  # 测试目录
├── frontend/               # 前端代码（预留）
├── docs/                   # 文档目录
├── notebooks/              # Jupyter Notebook目录
├── data/                   # 数据目录
├── requirements.txt        # 依赖清单
├── pyproject.toml          # Python项目配置
└── README.md               # 项目说明
```

## 环境要求

- Python 3.8+

## 安装和配置

### 1. 创建虚拟环境

```bash
python -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或使用 pyproject.toml
pip install .

# 安装开发依赖
pip install ".[dev]"
```

## 运行演示

### RAG文档分割器演示

```bash
cd src/demos/20260508_rag_splitter
python demo.py
```

### 思维反射框架演示

```bash
cd src/demos/20260507_thinking_reflection
python demo.py
```

## 依赖管理

### requirements.txt vs pyproject.toml

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `requirements.txt` | 简单依赖声明 | 快速开发、CI/CD |
| `pyproject.toml` | 标准化配置 | 包发布、复杂项目 |

### 常见操作

```bash
# 安装依赖
pip install -r requirements.txt

# 更新依赖
pip install --upgrade -r requirements.txt

# 导出依赖清单
pip freeze > requirements.txt
```

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目仅供学习使用。
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with new structure"
```

---

### Task 7: 更新 .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: 更新忽略规则**

```gitignore
# Virtual environment
venv/
.env/
.env.local

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Build artifacts
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment variables
.env
.secret

# Frontend
frontend/node_modules/
frontend/dist/
frontend/.next/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "feat: update gitignore rules"
```

---

### Task 8: 删除旧demo目录

**Files:**
- Delete: `20260507-demo/`
- Delete: `20260507-demo02-thinking-reflection-framework/`
- Delete: `20260508-demo/`
- Delete: `20260508-demo-rag-splitter/`

**Step 1: 删除旧目录**

```bash
rm -rf 20260507-demo/ 20260507-demo02-thinking-reflection-framework/ 20260508-demo/ 20260508-demo-rag-splitter/
```

**Step 2: 验证删除**

Run: `ls -la | grep "202605"`
Expected: 无旧demo目录输出

**Step 3: Commit**

```bash
git add .
git commit -m "cleanup: remove old demo directories"
```

---

### Task 9: 运行测试验证

**Files:**
- Test: 所有demo

**Step 1: 运行所有demo验证**

```bash
# 测试 RAG 分割器
cd src/demos/20260508_rag_splitter && python demo.py

# 测试思维反射框架
cd src/demos/20260507_thinking_reflection && python demo.py

# 测试简单demo
python src/demos/20260508_demo/hello_world.py
```

**Step 2: 验证结果**

所有demo应正常运行，无错误输出。

**Step 3: Commit**

```bash
git commit -m "test: verify all demos work correctly"
```

---

### Task 10: 清理 .trae 目录中过时的文档

**Files:**
- Delete: `.trae/documents/sentence_splitter_plan.md`
- Delete: `.trae/documents/rag_document_splitter_plan.md`

**Step 1: 检查并删除过时文档**

```bash
rm -f .trae/documents/sentence_splitter_plan.md .trae/documents/rag_document_splitter_plan.md
```

**Step 2: Commit**

```bash
git add .trae/
git commit -m "cleanup: remove obsolete plan documents"
```

---

## 预期输出

### 重构后的目录结构

```
202605-ai-study/
├── .git/
├── .gitignore
├── .trae/
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── plans/
│       └── 2026-05-08-project-restructure.md
├── src/
│   ├── __init__.py
│   ├── common/
│   │   └── __init__.py
│   ├── demos/
│   │   ├── __init__.py
│   │   ├── 20260507_demo/
│   │   ├── 20260507_thinking_reflection/
│   │   ├── 20260508_demo/
│   │   └── 20260508_rag_splitter/
│   │       ├── demo.py
│   │       ├── README.md
│   │       ├── data/
│   │       └── src/
│   │           ├── __init__.py
│   │           ├── splitters/
│   │           └── utils/
│   ├── api/
│   │   └── __init__.py
│   └── config/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── frontend/
├── notebooks/
└── data/
```

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 导入路径错误 | 高 | 功能异常 | 更新导入语句后逐项测试 |
| 依赖冲突 | 中 | 运行失败 | 使用虚拟环境隔离 |
| 配置文件失效 | 中 | 包安装失败 | 更新后测试构建 |
| 测试遗漏 | 中 | 潜在bug | 重构后运行所有demo验证 |

---

## 完成标准

1. ✅ 所有demo迁移完成，日期信息保留
2. ✅ 目录结构符合命名规范
3. ✅ 依赖清理完成，仅保留必需依赖
4. ✅ 所有demo运行正常
5. ✅ 配置文件更新完成
6. ✅ 文档更新完成
