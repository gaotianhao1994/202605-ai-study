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
