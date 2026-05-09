# LangChain 入门教程改进版实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个模块化的 LangChain 教程项目，使用阿里云百炼大模型，提供完整的错误处理和日志记录功能。

**Architecture:** 采用按章节分模块的设计，每个章节一个独立的 Python 文件。使用 config.py 管理配置，logger.py 提供日志功能，main.py 作为主入口。所有代码包含详细的错误处理和注释。

**Tech Stack:** Python 3.8+, LangChain 1.2.17, 阿里云百炼 API, python-dotenv

---

## 前置条件

- 项目根目录：`/0-gaoth/projects/202605/202605-ai-study`
- 目标目录：`/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1`
- 已有依赖文件：`/0-gaoth/projects/202605/202605-ai-study/requirements.txt`
- 已有配置信息：`/0-gaoth/projects/202605/202605-ai-study/src/demos/20260508-demo4-LangChainDemo1/.env`

---

### Task 1: 创建项目目录结构

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/` (directory)

**Step 1: 创建项目目录**

Run: `mkdir -p /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1`
Expected: Directory created successfully

**Step 2: 验证目录创建**

Run: `ls -la /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1`
Expected: Directory listing (empty)

---

### Task 2: 创建配置管理模块

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/config.py`

**Step 1: 创建 config.py 文件**

```python
"""
配置管理模块

负责加载环境变量和提供配置信息
支持阿里云百炼 API 配置

作者：AI Study Project
日期：2026-05-09
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv
from pathlib import Path


class ConfigError(Exception):
    """配置错误异常"""
    pass


class Config:
    """
    配置管理类
    
    负责加载和管理环境变量配置
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            env_file: .env 文件路径，默认为当前目录下的 .env
        """
        self.env_file = env_file
        self._api_key: Optional[str] = None
        self._api_base: Optional[str] = None
        self._model_name: Optional[str] = None
        
        self.load_env()
    
    def load_env(self) -> None:
        """
        加载环境变量文件
        
        Raises:
            FileNotFoundError: .env 文件不存在
        """
        if self.env_file:
            env_path = Path(self.env_file)
        else:
            env_path = Path(__file__).parent / '.env'
        
        if not env_path.exists():
            raise FileNotFoundError(
                f".env 文件不存在: {env_path}\n"
                "请复制 .env.example 为 .env 并填写配置"
            )
        
        load_dotenv(env_path)
        
        self._api_key = os.getenv('OPENAI_API_KEY')
        self._api_base = os.getenv('OPENAI_API_BASE')
        self._model_name = os.getenv('OPENAI_MODEL_NAME')
    
    def get_api_key(self) -> str:
        """
        获取 API Key
        
        Returns:
            API Key 字符串
            
        Raises:
            ConfigError: API Key 未配置
        """
        if not self._api_key:
            raise ConfigError(
                "OPENAI_API_KEY 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_KEY"
            )
        return self._api_key
    
    def get_api_base(self) -> str:
        """
        获取 API Base URL
        
        Returns:
            API Base URL 字符串
            
        Raises:
            ConfigError: API Base 未配置
        """
        if not self._api_base:
            raise ConfigError(
                "OPENAI_API_BASE 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_BASE"
            )
        return self._api_base
    
    def get_model_name(self) -> str:
        """
        获取模型名称
        
        Returns:
            模型名称字符串
            
        Raises:
            ConfigError: 模型名称未配置
        """
        if not self._model_name:
            raise ConfigError(
                "OPENAI_MODEL_NAME 未配置\n"
                "请在 .env 文件中设置 OPENAI_MODEL_NAME"
            )
        return self._model_name
    
    def validate(self) -> bool:
        """
        验证配置是否完整
        
        Returns:
            True 如果所有必需配置都存在
            
        Raises:
            ConfigError: 配置不完整
        """
        try:
            self.get_api_key()
            self.get_api_base()
            self.get_model_name()
            return True
        except ConfigError as e:
            raise ConfigError(f"配置验证失败: {e}")
    
    def get_model_config(self) -> Dict[str, str]:
        """
        获取完整的模型配置字典
        
        Returns:
            包含 model, openai_api_base, openai_api_key 的字典
        """
        return {
            'model': self.get_model_name(),
            'openai_api_base': self.get_api_base(),
            'openai_api_key': self.get_api_key()
        }
    
    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        api_key_preview = f"{self._api_key[:10]}..." if self._api_key else "未设置"
        return (
            f"Config(\n"
            f"  api_key='{api_key_preview}',\n"
            f"  api_base='{self._api_base}',\n"
            f"  model_name='{self._model_name}'\n"
            f")"
        )


def get_config(env_file: Optional[str] = None) -> Config:
    """
    获取配置实例（便捷函数）
    
    Args:
        env_file: .env 文件路径
        
    Returns:
        Config 实例
    """
    return Config(env_file)


if __name__ == '__main__':
    try:
        config = get_config()
        print("✅ 配置加载成功")
        print(config)
        config.validate()
        print("✅ 配置验证通过")
    except Exception as e:
        print(f"❌ 配置错误: {e}")
```

**Step 2: 验证文件创建**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/config.py`
Expected: 配置错误提示（因为还没有 .env 文件）

---

### Task 3: 创建日志配置模块

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/logger.py`

**Step 1: 创建 logger.py 文件**

```python
"""
日志配置模块

提供统一的日志记录功能
支持控制台和文件输出

作者：AI Study Project
日期：2026-05-09
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回 logger 实例
    
    Args:
        name: logger 名称
        level: 日志级别，默认 DEBUG
        log_file: 日志文件路径（可选）
        format_string: 自定义日志格式（可选）
    
    Returns:
        配置好的 logger 实例
    
    Example:
        >>> logger = setup_logger('my_module')
        >>> logger.info('这是一条信息')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已配置的 logger 实例
    
    如果 logger 不存在，会自动创建一个默认配置的 logger
    
    Args:
        name: logger 名称
        
    Returns:
        logger 实例
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerContext:
    """
    日志上下文管理器
    
    用于临时修改日志级别
    
    Example:
        >>> with LoggerContext(logger, logging.WARNING):
        ...     logger.debug('这条不会显示')
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        初始化上下文管理器
        
        Args:
            logger: logger 实例
            level: 临时日志级别
        """
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """进入上下文，修改日志级别"""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复日志级别"""
        self.logger.setLevel(self.old_level)
        return False


if __name__ == '__main__':
    logger = setup_logger('test_logger', log_file='logs/test.log')
    
    logger.debug('这是一条调试信息')
    logger.info('这是一条普通信息')
    logger.warning('这是一条警告信息')
    logger.error('这是一条错误信息')
    
    print("\n✅ 日志模块测试完成")
```

**Step 2: 验证文件创建**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/logger.py`
Expected: 日志输出到控制台和 logs/test.log 文件

---

### Task 4: 创建环境变量文件

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/.env.example`
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/.env`

**Step 1: 创建 .env.example 模板文件**

```bash
# 阿里云百炼 API 配置
# API Key - 从阿里云百炼控制台获取
OPENAI_API_KEY=your-api-key-here

# 阿里云百炼 API 端点
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 使用的模型名称
OPENAI_MODEL_NAME=qwen3.5-122b-a10b
```

**Step 2: 创建实际的 .env 文件**

从原项目复制配置：
```bash
# 阿里云百炼 API 配置
# API Key
OPENAI_API_KEY=sk-5e4ce55385c042acbd69d58a7bb9a230
# 阿里云百炼 API 端点
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
# 使用的模型
OPENAI_MODEL_NAME=qwen3.5-122b-a10b
```

**Step 3: 验证配置文件**

Run: `cat /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/.env`
Expected: 显示配置内容

---

### Task 5: 创建第一章模块 - 环境准备

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter1_env_setup.py`

**Step 1: 创建 chapter1_env_setup.py 文件**

```python
"""
第一章：环境准备与快速开始

学习目标：
- 了解 LangChain 是什么
- 检查环境配置
- 运行第一个 LangChain 程序

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, ConfigError
from logger import setup_logger

logger = setup_logger('chapter1_env_setup')


def check_python_version() -> bool:
    """
    检查 Python 版本是否 >= 3.8
    
    Returns:
        True 如果版本符合要求
    """
    logger.info("检查 Python 版本...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python 版本: {sys.version}")
    print(f"Python 版本号: {version_str}")
    print(f"Python 解释器路径: {sys.executable}")
    
    if version >= (3, 8):
        logger.info(f"✅ Python 版本符合要求: {version_str}")
        print("✅ Python 版本符合要求")
        return True
    else:
        logger.error(f"❌ Python 版本过低: {version_str}")
        print("❌ Python 版本过低，请升级到 3.8 或更高版本")
        return False


def check_dependencies() -> bool:
    """
    检查依赖是否安装
    
    Returns:
        True 如果所有依赖都已安装
    """
    logger.info("检查依赖安装...")
    
    dependencies = {
        'langchain': 'langchain',
        'langchain_openai': 'langchain-openai',
        'dotenv': 'python-dotenv'
    }
    
    all_installed = True
    
    for module_name, package_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            logger.debug(f"✅ {package_name} 已安装: {version}")
            print(f"✅ {package_name} 已安装: {version}")
        except ImportError:
            logger.error(f"❌ {package_name} 未安装")
            print(f"❌ {package_name} 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n请安装缺失的依赖:")
        print("pip install langchain==1.2.17 langchain-openai python-dotenv")
    
    return all_installed


def check_api_key() -> bool:
    """
    检查 API Key 配置
    
    Returns:
        True 如果 API Key 已配置
    """
    logger.info("检查 API Key 配置...")
    
    try:
        config = get_config()
        api_key = config.get_api_key()
        
        key_preview = f"{api_key[:10]}..."
        
        logger.info(f"✅ API Key 已配置: {key_preview}")
        print(f"✅ API Key 已配置")
        print(f"   API Key 前缀: {key_preview}")
        
        return True
        
    except (FileNotFoundError, ConfigError) as e:
        logger.error(f"❌ API Key 配置错误: {e}")
        print(f"❌ API Key 配置错误")
        print(f"\n请按照以下步骤配置:")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中填入你的 OPENAI_API_KEY")
        print("3. 重新运行此程序")
        
        return False


def run_first_program() -> bool:
    """
    运行第一个 LangChain 程序
    
    Returns:
        True 如果程序运行成功
    """
    logger.info("运行第一个 LangChain 程序...")
    
    print("\n" + "=" * 60)
    print("第一个 LangChain 程序")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI
        
        config = get_config()
        model_config = config.get_model_config()
        
        logger.debug(f"创建模型实例: {model_config['model']}")
        
        llm = ChatOpenAI(
            **model_config,
            temperature=0.7
        )
        
        prompt = "你好，请用一句话介绍 LangChain"
        
        logger.info(f"调用模型，提示词: {prompt}")
        print(f"\n提示词: {prompt}")
        print("\nAI 正在思考...\n")
        
        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("AI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 程序运行成功！耗时 {elapsed_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 程序运行失败: {e}", exc_info=True)
        print(f"\n❌ 程序运行失败: {e}")
        print("\n可能的原因:")
        print("1. API Key 无效")
        print("2. 网络连接问题")
        print("3. API 服务不可用")
        
        return False


def main():
    """
    主函数：运行所有环境检查
    
    执行步骤：
    1. 检查 Python 版本
    2. 检查依赖安装
    3. 检查 API Key 配置
    4. 运行第一个程序
    """
    print("=" * 60)
    print("📚 第一章：环境准备与快速开始")
    print("=" * 60)
    
    logger.info("开始环境检查...")
    
    checks = [
        ("Python 版本", check_python_version),
        ("依赖安装", check_dependencies),
        ("API Key 配置", check_api_key),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{'=' * 60}")
        print(f"检查项: {name}")
        print("=" * 60)
        
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"检查 {name} 时发生错误: {e}", exc_info=True)
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("运行第一个 LangChain 程序")
        print("=" * 60)
        run_first_program()
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！环境配置完成，可以开始学习了！")
        print("=" * 60)
    else:
        print("\n❌ 部分检查未通过，请先解决上述问题")
    
    logger.info("环境检查完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第一章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter1_env_setup.py`
Expected: 环境检查通过，第一个程序运行成功

---

### Task 6: 创建第二章模块 - 模型调用

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter2_model_io.py`

**Step 1: 创建 chapter2_model_io.py 文件**

```python
"""
第二章：模型调用（Model I/O）

学习目标：
- 学习 LLM 和 Chat Model 的区别
- 掌握不同的调用方式
- 理解模型参数的作用

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI

logger = setup_logger('chapter2_model_io')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """
    创建模型实例
    
    Args:
        temperature: 创造性程度，范围 0-1
                    0 = 保守，1 = 创造性
    
    Returns:
        ChatOpenAI: 配置好的模型实例
        
    Raises:
        ValueError: temperature 超出范围
    """
    if temperature < 0 or temperature > 1:
        raise ValueError(f"temperature 必须在 0-1 之间，当前值: {temperature}")
    
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info(f"模型实例创建成功: {model_config['model']}")
    return llm


def demo_invoke():
    """
    演示单次调用
    
    使用 invoke() 方法进行单次模型调用
    """
    print("\n" + "=" * 60)
    print("演示 1: invoke() - 单次调用")
    print("=" * 60)
    
    logger.info("开始演示 invoke() 单次调用")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = "什么是机器学习？"
        
        logger.info(f"调用模型，提示词: {prompt}")
        print(f"\n提示词: {prompt}")
        print("\nAI 正在思考...\n")
        
        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("AI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 单次调用成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 单次调用失败: {e}", exc_info=True)
        print(f"\n❌ 单次调用失败: {e}")


def demo_batch():
    """
    演示批量调用
    
    使用 batch() 方法一次处理多个请求
    """
    print("\n" + "=" * 60)
    print("演示 2: batch() - 批量调用")
    print("=" * 60)
    
    logger.info("开始演示 batch() 批量调用")
    
    try:
        llm = create_model(temperature=0.7)
        
        questions = [
            "什么是 Python？",
            "什么是 JavaScript？",
            "什么是 Go？"
        ]
        
        logger.info(f"批量调用模型，问题数量: {len(questions)}")
        print(f"\n批量处理 {len(questions)} 个问题...\n")
        
        start_time = time.time()
        responses = llm.batch(questions)
        elapsed_time = time.time() - start_time
        
        logger.info(f"批量调用成功，耗时 {elapsed_time:.2f}s")
        
        for i, (question, response) in enumerate(zip(questions, responses), 1):
            print(f"\n{'=' * 60}")
            print(f"问题 {i}: {question}")
            print("=" * 60)
            print(response.content)
        
        print(f"\n✅ 批量调用成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 批量调用失败: {e}", exc_info=True)
        print(f"\n❌ 批量调用失败: {e}")


def demo_stream():
    """
    演示流式输出
    
    使用 stream() 方法实时显示输出
    """
    print("\n" + "=" * 60)
    print("演示 3: stream() - 流式输出")
    print("=" * 60)
    
    logger.info("开始演示 stream() 流式输出")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = "请写一个关于人工智能的短故事"
        
        logger.info(f"流式调用模型，提示词: {prompt}")
        print(f"\n提示词: {prompt}")
        print("\nAI 正在回答（流式输出）:")
        print("=" * 60)
        
        start_time = time.time()
        
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        logger.info(f"流式输出完成，耗时 {elapsed_time:.2f}s")
        print(f"\n✅ 流式输出成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 流式输出失败: {e}", exc_info=True)
        print(f"\n❌ 流式输出失败: {e}")


def demo_temperature():
    """
    演示 temperature 参数效果
    
    对比不同 temperature 值的输出差异
    """
    print("\n" + "=" * 60)
    print("演示 4: temperature 参数对比")
    print("=" * 60)
    
    logger.info("开始演示 temperature 参数效果")
    
    try:
        prompt = "写一个关于咖啡的广告语"
        
        temperatures = [0.0, 1.0]
        
        for temp in temperatures:
            print(f"\n{'=' * 60}")
            print(f"temperature={temp} ({'保守' if temp == 0 else '创造性'})")
            print("=" * 60)
            
            logger.info(f"使用 temperature={temp} 调用模型")
            
            llm = create_model(temperature=temp)
            
            start_time = time.time()
            response = llm.invoke(prompt)
            elapsed_time = time.time() - start_time
            
            logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
            
            print(f"\n提示词: {prompt}")
            print("\nAI 的回答:")
            print("-" * 60)
            print(response.content)
            print("-" * 60)
        
        print("\n✅ temperature 参数演示完成")
        print("\n💡 提示: temperature=0 输出更确定，temperature=1 输出更有创造性")
        
    except Exception as e:
        logger.error(f"❌ temperature 演示失败: {e}", exc_info=True)
        print(f"\n❌ temperature 演示失败: {e}")


def main():
    """
    主函数：运行所有模型调用演示
    
    执行顺序：
    1. 单次调用演示
    2. 批量调用演示
    3. 流式输出演示
    4. temperature 参数演示
    """
    print("=" * 60)
    print("🤖 第二章：模型调用（Model I/O）")
    print("=" * 60)
    
    logger.info("开始第二章演示...")
    
    demos = [
        ("单次调用", demo_invoke),
        ("批量调用", demo_batch),
        ("流式输出", demo_stream),
        ("temperature 参数", demo_temperature),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第二章演示完成！")
    print("=" * 60)
    
    logger.info("第二章演示完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第二章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter2_model_io.py`
Expected: 所有模型调用演示成功运行

---

### Task 7: 创建第三章模块 - 提示词模板

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter3_prompts.py`

**Step 1: 创建 chapter3_prompts.py 文件**

```python
"""
第三章：提示词模板（Prompts）

学习目标：
- 理解模板的作用
- 学习模板的基本使用
- 掌握模板组合技巧

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

logger = setup_logger('chapter3_prompts')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info(f"模型实例创建成功")
    return llm


def demo_prompt_template():
    """
    演示 PromptTemplate
    
    简单字符串模板的使用
    """
    print("\n" + "=" * 60)
    print("演示 1: PromptTemplate - 简单字符串模板")
    print("=" * 60)
    
    logger.info("开始演示 PromptTemplate")
    
    try:
        template = "请用一句话介绍{subject}"
        
        logger.debug(f"创建模板: {template}")
        prompt = PromptTemplate.from_template(template)
        
        final_prompt = prompt.format(subject="Python")
        
        logger.info(f"生成的提示词: {final_prompt}")
        print(f"\n模板: {template}")
        print(f"变量: subject='Python'")
        print(f"\n生成的提示词:")
        print("-" * 60)
        print(final_prompt)
        print("-" * 60)
        
        llm = create_model(temperature=0.7)
        
        logger.info("调用模型...")
        start_time = time.time()
        response = llm.invoke(final_prompt)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ PromptTemplate 演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ PromptTemplate 演示失败: {e}", exc_info=True)
        print(f"\n❌ PromptTemplate 演示失败: {e}")


def demo_chat_prompt_template():
    """
    演示 ChatPromptTemplate
    
    聊天消息模板的使用，包含系统消息和用户消息
    """
    print("\n" + "=" * 60)
    print("演示 2: ChatPromptTemplate - 聊天消息模板")
    print("=" * 60)
    
    logger.info("开始演示 ChatPromptTemplate")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}，请用专业但易懂的语言回答问题。"),
            ("user", "{question}")
        ])
        
        logger.debug(f"创建聊天模板，包含系统消息和用户消息")
        
        messages = prompt.format_messages(
            role="Python 专家",
            question="什么是装饰器？"
        )
        
        logger.info("生成的消息:")
        for msg in messages:
            logger.info(f"  {msg.type}: {msg.content}")
        
        print("\n模板消息:")
        print("-" * 60)
        for msg in messages:
            print(f"{msg.type}: {msg.content}")
        print("-" * 60)
        
        llm = create_model(temperature=0.7)
        
        logger.info("调用模型...")
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        logger.info(f"模型调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ ChatPromptTemplate 演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ ChatPromptTemplate 演示失败: {e}", exc_info=True)
        print(f"\n❌ ChatPromptTemplate 演示失败: {e}")


def demo_translation_template():
    """
    演示翻译助手模板
    
    创建一个可以翻译多种语言的模板
    """
    print("\n" + "=" * 60)
    print("演示 3: 翻译助手模板")
    print("=" * 60)
    
    logger.info("开始演示翻译助手模板")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的翻译助手。"),
            ("user", "请将以下{source_lang}翻译成{target_lang}：{text}")
        ])
        
        logger.debug("创建翻译模板")
        
        test_cases = [
            {
                "source_lang": "英文",
                "target_lang": "中文",
                "text": "Hello, World!"
            },
            {
                "source_lang": "中文",
                "target_lang": "英文",
                "text": "你好，世界！"
            }
        ]
        
        llm = create_model(temperature=0.3)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"翻译案例 {i}")
            print("=" * 60)
            
            messages = prompt.format_messages(**case)
            
            logger.info(f"翻译: {case['source_lang']} -> {case['target_lang']}")
            print(f"\n原文 ({case['source_lang']}): {case['text']}")
            
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            logger.info(f"翻译成功，耗时 {elapsed_time:.2f}s")
            
            print(f"\n译文 ({case['target_lang']}):")
            print("-" * 60)
            print(response.content)
            print("-" * 60)
        
        print("\n✅ 翻译助手模板演示成功！")
        
    except Exception as e:
        logger.error(f"❌ 翻译助手模板演示失败: {e}", exc_info=True)
        print(f"\n❌ 翻译助手模板演示失败: {e}")


def demo_code_explainer():
    """
    演示代码解释器模板
    
    创建一个可以解释代码的模板
    """
    print("\n" + "=" * 60)
    print("演示 4: 代码解释器模板")
    print("=" * 60)
    
    logger.info("开始演示代码解释器模板")
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{language}专家，请用简单易懂的语言解释代码。"),
            ("user", "请解释以下代码：\n{code}")
        ])
        
        logger.debug("创建代码解释模板")
        
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        messages = prompt.format_messages(
            language="Python",
            code=code
        )
        
        print(f"\n代码:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        
        llm = create_model(temperature=0.3)
        
        logger.info("调用模型解释代码...")
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        logger.info(f"代码解释成功，耗时 {elapsed_time:.2f}s")
        
        print("\n代码解释:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 代码解释器模板演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 代码解释器模板演示失败: {e}", exc_info=True)
        print(f"\n❌ 代码解释器模板演示失败: {e}")


def main():
    """
    主函数：运行所有提示词模板演示
    
    执行顺序：
    1. PromptTemplate 演示
    2. ChatPromptTemplate 演示
    3. 翻译助手模板演示
    4. 代码解释器模板演示
    """
    print("=" * 60)
    print("📝 第三章：提示词模板（Prompts）")
    print("=" * 60)
    
    logger.info("开始第三章演示...")
    
    demos = [
        ("PromptTemplate", demo_prompt_template),
        ("ChatPromptTemplate", demo_chat_prompt_template),
        ("翻译助手模板", demo_translation_template),
        ("代码解释器模板", demo_code_explainer),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第三章演示完成！")
    print("=" * 60)
    
    logger.info("第三章演示完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第三章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter3_prompts.py`
Expected: 所有提示词模板演示成功运行

---

### Task 8: 创建第四章模块 - 链式调用

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter4_chains.py`

**Step 1: 创建 chapter4_chains.py 文件**

```python
"""
第四章：链式调用（Chains）

学习目标：
- 理解链的概念
- 学习 LCEL 语法
- 构建简单和复杂的处理链

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter4_chains')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info(f"模型实例创建成功")
    return llm


def demo_simple_chain():
    """
    演示简单链
    
    使用 LCEL 语法（| 操作符）连接组件
    """
    print("\n" + "=" * 60)
    print("演示 1: 简单链 - LCEL 语法")
    print("=" * 60)
    
    logger.info("开始演示简单链")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
        
        logger.debug("创建链: prompt | llm")
        chain = prompt | llm
        
        logger.info("调用链，topic='程序员'")
        print("\n链的组成: prompt | llm")
        print("\n输入: {'topic': '程序员'}")
        
        start_time = time.time()
        response = chain.invoke({"topic": "程序员"})
        elapsed_time = time.time() - start_time
        
        logger.info(f"链调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n✅ 简单链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 简单链演示失败: {e}", exc_info=True)
        print(f"\n❌ 简单链演示失败: {e}")


def demo_chain_with_parser():
    """
    演示带输出解析器的链
    
    添加 StrOutputParser 将输出转换为字符串
    """
    print("\n" + "=" * 60)
    print("演示 2: 带输出解析器的链")
    print("=" * 60)
    
    logger.info("开始演示带输出解析器的链")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
        output_parser = StrOutputParser()
        
        logger.debug("创建链: prompt | llm | output_parser")
        chain = prompt | llm | output_parser
        
        logger.info("调用链，topic='人工智能'")
        print("\n链的组成: prompt | llm | output_parser")
        print("\n输入: {'topic': '人工智能'}")
        
        start_time = time.time()
        result = chain.invoke({"topic": "人工智能"})
        elapsed_time = time.time() - start_time
        
        logger.info(f"链调用成功，耗时 {elapsed_time:.2f}s")
        
        print("\nAI 的回答（字符串格式）:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        print(f"\n✅ 带输出解析器的链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 带输出解析器的链演示失败: {e}", exc_info=True)
        print(f"\n❌ 带输出解析器的链演示失败: {e}")


def demo_summary_chain():
    """
    演示文章摘要生成链
    
    创建一个可以生成文章摘要的链
    """
    print("\n" + "=" * 60)
    print("演示 3: 文章摘要生成链")
    print("=" * 60)
    
    logger.info("开始演示文章摘要生成链")
    
    try:
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请为以下文章生成一个简洁的摘要（不超过50字）：\n\n{article}"
        )
        
        logger.debug("创建摘要生成链")
        chain = prompt | llm | StrOutputParser()
        
        article = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
这些任务包括学习、推理、问题解决、感知和语言理解。AI 技术已经广泛应用于各个领域，
包括医疗诊断、金融分析、自动驾驶汽车和智能家居设备。
        """
        
        logger.info("调用摘要生成链")
        print("\n原文:")
        print("-" * 60)
        print(article)
        print("-" * 60)
        
        start_time = time.time()
        summary = chain.invoke({"article": article})
        elapsed_time = time.time() - start_time
        
        logger.info(f"摘要生成成功，耗时 {elapsed_time:.2f}s")
        
        print("\n生成的摘要:")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        print(f"\n✅ 文章摘要生成链演示成功！耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ 文章摘要生成链演示失败: {e}", exc_info=True)
        print(f"\n❌ 文章摘要生成链演示失败: {e}")


def demo_translation_chain():
    """
    演示多语言翻译链
    
    创建一个可以翻译多种语言的链
    """
    print("\n" + "=" * 60)
    print("演示 4: 多语言翻译链")
    print("=" * 60)
    
    logger.info("开始演示多语言翻译链")
    
    try:
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请将以下文本翻译成{target_language}：\n\n{text}"
        )
        
        logger.debug("创建翻译链")
        chain = prompt | llm | StrOutputParser()
        
        test_cases = [
            {"target_language": "英文", "text": "你好，世界！"},
            {"target_language": "日文", "text": "你好，世界！"}
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"翻译案例 {i}")
            print("=" * 60)
            
            logger.info(f"翻译到 {case['target_language']}")
            print(f"\n原文: {case['text']}")
            print(f"目标语言: {case['target_language']}")
            
            start_time = time.time()
            result = chain.invoke(case)
            elapsed_time = time.time() - start_time
            
            logger.info(f"翻译成功，耗时 {elapsed_time:.2f}s")
            
            print(f"\n译文:")
            print("-" * 60)
            print(result)
            print("-" * 60)
        
        print("\n✅ 多语言翻译链演示成功！")
        
    except Exception as e:
        logger.error(f"❌ 多语言翻译链演示失败: {e}", exc_info=True)
        print(f"\n❌ 多语言翻译链演示失败: {e}")


def main():
    """
    主函数：运行所有链式调用演示
    
    执行顺序：
    1. 简单链演示
    2. 带输出解析器的链演示
    3. 文章摘要生成链演示
    4. 多语言翻译链演示
    """
    print("=" * 60)
    print("🔗 第四章：链式调用（Chains）")
    print("=" * 60)
    
    logger.info("开始第四章演示...")
    
    demos = [
        ("简单链", demo_simple_chain),
        ("带输出解析器的链", demo_chain_with_parser),
        ("文章摘要生成链", demo_summary_chain),
        ("多语言翻译链", demo_translation_chain),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第四章演示完成！")
    print("=" * 60)
    
    logger.info("第四章演示完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第四章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter4_chains.py`
Expected: 所有链式调用演示成功运行

---

### Task 9: 创建第五章模块 - 记忆组件

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter5_memory.py`

**Step 1: 创建 chapter5_memory.py 文件**

```python
"""
第五章：记忆组件（Memory）

学习目标：
- 理解为什么需要记忆
- 学习不同的记忆类型
- 实现多轮对话

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain

logger = setup_logger('chapter5_memory')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info(f"模型实例创建成功")
    return llm


def demo_buffer_memory():
    """
    演示 ConversationBufferMemory
    
    保存完整的对话历史
    """
    print("\n" + "=" * 60)
    print("演示 1: ConversationBufferMemory - 完整记忆")
    print("=" * 60)
    
    logger.info("开始演示 ConversationBufferMemory")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferMemory()
        
        logger.debug("创建对话链，使用 ConversationBufferMemory")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        conversations = [
            ("你好，我叫小明", "第一轮对话"),
            ("你还记得我的名字吗？", "第二轮对话")
        ]
        
        for user_input, round_name in conversations:
            print(f"\n{'=' * 60}")
            print(f"{round_name}")
            print("=" * 60)
            
            logger.info(f"用户输入: {user_input}")
            print(f"用户: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
        
        print(f"\n✅ ConversationBufferMemory 演示成功！")
        print("\n💡 提示: AI 记住了第一轮对话中的名字")
        
    except Exception as e:
        logger.error(f"❌ ConversationBufferMemory 演示失败: {e}", exc_info=True)
        print(f"\n❌ ConversationBufferMemory 演示失败: {e}")


def demo_window_memory():
    """
    演示 ConversationBufferWindowMemory
    
    只保存最近 N 轮对话
    """
    print("\n" + "=" * 60)
    print("演示 2: ConversationBufferWindowMemory - 窗口记忆")
    print("=" * 60)
    
    logger.info("开始演示 ConversationBufferWindowMemory")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferWindowMemory(k=2)
        
        logger.debug("创建对话链，使用 ConversationBufferWindowMemory(k=2)")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        conversations = [
            ("我喜欢吃苹果", "第一轮对话"),
            ("我喜欢吃香蕉", "第二轮对话"),
            ("我喜欢吃什么水果？", "第三轮对话")
        ]
        
        for user_input, round_name in conversations:
            print(f"\n{'=' * 60}")
            print(f"{round_name}")
            print("=" * 60)
            
            logger.info(f"用户输入: {user_input}")
            print(f"用户: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
        
        print(f"\n✅ ConversationBufferWindowMemory 演示成功！")
        print("\n💡 提示: AI 只记得最近 2 轮对话")
        
    except Exception as e:
        logger.error(f"❌ ConversationBufferWindowMemory 演示失败: {e}", exc_info=True)
        print(f"\n❌ ConversationBufferWindowMemory 演示失败: {e}")


def demo_chatbot():
    """
    演示有记忆的聊天机器人
    
    创建一个能记住对话历史的聊天机器人
    """
    print("\n" + "=" * 60)
    print("演示 3: 有记忆的聊天机器人")
    print("=" * 60)
    
    logger.info("开始演示有记忆的聊天机器人")
    
    try:
        llm = create_model(temperature=0.7)
        
        memory = ConversationBufferMemory()
        
        logger.debug("创建聊天机器人")
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        print("\n聊天机器人已启动！")
        print("=" * 60)
        
        test_inputs = [
            "你好，我想学习 Python",
            "你能推荐一些学习资源吗？",
            "谢谢！我刚才说我想学什么？"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n【第 {i} 轮对话】")
            print(f"用户: {user_input}")
            
            logger.info(f"用户输入: {user_input}")
            
            start_time = time.time()
            response = conversation.predict(input=user_input)
            elapsed_time = time.time() - start_time
            
            logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
            print(f"AI: {response}")
            print("-" * 60)
        
        print("\n✅ 聊天机器人演示成功！")
        print("\n💡 提示: AI 记住了整个对话历史")
        
    except Exception as e:
        logger.error(f"❌ 聊天机器人演示失败: {e}", exc_info=True)
        print(f"\n❌ 聊天机器人演示失败: {e}")


def main():
    """
    主函数：运行所有记忆组件演示
    
    执行顺序：
    1. ConversationBufferMemory 演示
    2. ConversationBufferWindowMemory 演示
    3. 有记忆的聊天机器人演示
    """
    print("=" * 60)
    print("💾 第五章：记忆组件（Memory）")
    print("=" * 60)
    
    logger.info("开始第五章演示...")
    
    demos = [
        ("ConversationBufferMemory", demo_buffer_memory),
        ("ConversationBufferWindowMemory", demo_window_memory),
        ("有记忆的聊天机器人", demo_chatbot),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第五章演示完成！")
    print("=" * 60)
    
    logger.info("第五章演示完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第五章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter5_memory.py`
Expected: 所有记忆组件演示成功运行

---

### Task 10: 创建第六章模块 - 综合实战

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter6_integration.py`

**Step 1: 创建 chapter6_integration.py 文件**

```python
"""
第六章：综合实战

学习目标：
- 综合运用所有知识
- 构建一个智能学习助手
- 学习扩展和优化方法

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

logger = setup_logger('chapter6_integration')


class StudyAssistant:
    """
    智能学习助手类
    
    集成了模型调用、提示词模板、链式调用和记忆组件
    """
    
    def __init__(self, temperature: float = 0.7):
        """
        初始化学习助手
        
        Args:
            temperature: 创造性程度，范围 0-1
        """
        logger.info("初始化智能学习助手...")
        
        config = get_config()
        model_config = config.get_model_config()
        
        logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
        
        self.llm = ChatOpenAI(
            **model_config,
            temperature=temperature
        )
        
        self.memory = ConversationBufferMemory()
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        logger.info("智能学习助手初始化完成")
    
    def chat(self, user_input: str) -> str:
        """
        进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI 的回复
        """
        logger.info(f"用户输入: {user_input}")
        
        start_time = time.time()
        response = self.conversation.predict(input=user_input)
        elapsed_time = time.time() - start_time
        
        logger.info(f"AI 回复成功，耗时 {elapsed_time:.2f}s")
        
        return response
    
    def clear_memory(self):
        """清空对话记忆"""
        logger.info("清空对话记忆")
        self.memory.clear()


def create_study_assistant(temperature: float = 0.7) -> StudyAssistant:
    """
    创建智能学习助手
    
    Args:
        temperature: 创造性程度
        
    Returns:
        StudyAssistant 实例
    """
    return StudyAssistant(temperature=temperature)


def test_study_assistant():
    """
    测试智能学习助手
    
    模拟多轮对话，测试助手的记忆和回答能力
    """
    print("\n" + "=" * 60)
    print("测试智能学习助手")
    print("=" * 60)
    
    logger.info("开始测试智能学习助手")
    
    try:
        assistant = create_study_assistant(temperature=0.7)
        
        print("\n✅ 智能学习助手已创建！")
        print("=" * 60)
        
        test_conversations = [
            "你好！我想学习 Python，应该从哪里开始？",
            "变量是什么？能举个例子吗？",
            "函数怎么定义？",
            "你刚才说的变量和函数有什么关系？"
        ]
        
        for i, user_input in enumerate(test_conversations, 1):
            print(f"\n【第 {i} 轮对话】")
            print(f"学生: {user_input}")
            
            response = assistant.chat(user_input)
            
            print(f"\n学习助手: {response}")
            print("-" * 60)
        
        print("\n✅ 智能学习助手测试完成！")
        print("\n💡 提示: 助手记住了整个对话历史，能够连贯地回答问题")
        
    except Exception as e:
        logger.error(f"❌ 智能学习助手测试失败: {e}", exc_info=True)
        print(f"\n❌ 智能学习助手测试失败: {e}")


def demo_full_integration():
    """
    完整集成演示
    
    展示如何将所有组件组合成完整应用
    """
    print("\n" + "=" * 60)
    print("完整集成演示")
    print("=" * 60)
    
    logger.info("开始完整集成演示")
    
    print("\n智能学习助手架构:")
    print("-" * 60)
    print("1. 配置管理 (config.py)")
    print("   - 加载环境变量")
    print("   - 提供 API 配置")
    print()
    print("2. 模型调用 (ChatOpenAI)")
    print("   - 连接阿里云百炼 API")
    print("   - 调用大语言模型")
    print()
    print("3. 记忆组件 (ConversationBufferMemory)")
    print("   - 保存对话历史")
    print("   - 实现多轮对话")
    print()
    print("4. 对话链 (ConversationChain)")
    print("   - 组合模型和记忆")
    print("   - 提供统一的对话接口")
    print("-" * 60)
    
    test_study_assistant()
    
    print("\n" + "=" * 60)
    print("扩展思路")
    print("=" * 60)
    
    print("\n功能扩展:")
    print("1. 添加知识库：使用 LangChain 的检索功能，让 AI 能查询特定文档")
    print("2. 多轮对话优化：使用更好的记忆策略，如摘要记忆")
    print("3. 个性化学习路径：根据学生的学习进度推荐内容")
    
    print("\n技术扩展:")
    print("1. 使用不同的 LLM：如 GPT-4、Claude 等")
    print("2. 添加工具调用：让 AI 能执行代码、搜索网络等")
    print("3. 部署为 Web 应用：使用 Streamlit、Gradio 等框架")
    
    logger.info("完整集成演示完成")


def main():
    """
    主函数：运行综合实战
    
    执行步骤：
    1. 创建智能学习助手
    2. 测试学习助手
    3. 展示扩展思路
    """
    print("=" * 60)
    print("🎯 第六章：综合实战")
    print("=" * 60)
    
    logger.info("开始第六章演示...")
    
    demo_full_integration()
    
    print("\n" + "=" * 60)
    print("🎉 第六章演示完成！")
    print("=" * 60)
    
    logger.info("第六章演示完成")


if __name__ == '__main__':
    main()
```

**Step 2: 验证第六章运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter6_integration.py`
Expected: 综合实战演示成功运行

---

### Task 11: 创建主程序入口

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/main.py`

**Step 1: 创建 main.py 文件**

```python
"""
LangChain 入门教程 - 主程序入口

提供命令行界面，选择运行哪个章节的演示

作者：AI Study Project
日期：2026-05-09
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('main')


def show_menu():
    """显示章节菜单"""
    print("\n" + "=" * 60)
    print("📚 LangChain 入门教程")
    print("=" * 60)
    print("\n章节列表:")
    print("-" * 60)
    print("1. 环境准备与快速开始")
    print("2. 模型调用（Model I/O）")
    print("3. 提示词模板（Prompts）")
    print("4. 链式调用（Chains）")
    print("5. 记忆组件（Memory）")
    print("6. 综合实战")
    print("0. 退出")
    print("-" * 60)


def run_chapter(chapter: int):
    """
    运行指定章节
    
    Args:
        chapter: 章节编号（1-6）
    """
    logger.info(f"运行第 {chapter} 章")
    
    chapters = {
        1: 'chapter1_env_setup',
        2: 'chapter2_model_io',
        3: 'chapter3_prompts',
        4: 'chapter4_chains',
        5: 'chapter5_memory',
        6: 'chapter6_integration'
    }
    
    if chapter not in chapters:
        logger.error(f"无效的章节编号: {chapter}")
        print(f"❌ 无效的章节编号: {chapter}")
        return
    
    module_name = chapters[chapter]
    
    try:
        print(f"\n{'=' * 60}")
        print(f"正在运行第 {chapter} 章...")
        print("=" * 60)
        
        module = __import__(module_name)
        module.main()
        
    except Exception as e:
        logger.error(f"运行第 {chapter} 章失败: {e}", exc_info=True)
        print(f"\n❌ 运行第 {chapter} 章失败: {e}")


def run_all_chapters():
    """运行所有章节"""
    logger.info("运行所有章节")
    
    print("\n" + "=" * 60)
    print("📚 运行所有章节")
    print("=" * 60)
    
    for chapter in range(1, 7):
        run_chapter(chapter)
        
        if chapter < 6:
            input("\n按 Enter 键继续下一章...")
    
    print("\n" + "=" * 60)
    print("🎉 所有章节运行完成！")
    print("=" * 60)


def interactive_mode():
    """交互式模式"""
    logger.info("启动交互式模式")
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择章节 (0-6): ").strip()
            
            if choice == '0':
                logger.info("用户选择退出")
                print("\n👋 再见！")
                break
            
            chapter = int(choice)
            
            if 1 <= chapter <= 6:
                run_chapter(chapter)
            else:
                print("❌ 请输入 0-6 之间的数字")
        
        except ValueError:
            logger.warning("用户输入无效")
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            print("\n\n👋 再见！")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LangChain 入门教程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                  # 交互式模式
  python main.py --chapter 1      # 运行第一章
  python main.py --all            # 运行所有章节
        """
    )
    
    parser.add_argument(
        '--chapter', '-c',
        type=int,
        choices=range(1, 7),
        help='运行指定章节 (1-6)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='运行所有章节'
    )
    
    args = parser.parse_args()
    
    logger.info("程序启动")
    
    try:
        if args.all:
            run_all_chapters()
        elif args.chapter:
            run_chapter(args.chapter)
        else:
            interactive_mode()
    
    except Exception as e:
        logger.error(f"程序运行失败: {e}", exc_info=True)
        print(f"\n❌ 程序运行失败: {e}")
        sys.exit(1)
    
    logger.info("程序结束")


if __name__ == '__main__':
    main()
```

**Step 2: 验证主程序运行**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/main.py --chapter 1`
Expected: 第一章运行成功

---

### Task 12: 创建 README.md 文档

**Files:**
- Create: `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/README.md`

**Step 1: 创建 README.md 文件**

```markdown
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
```

**Step 2: 验证 README 创建**

Run: `head -50 /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/README.md`
Expected: 显示 README 前 50 行

---

### Task 13: 最终验证

**Files:**
- Verify: All files in `/0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/`

**Step 1: 验证所有文件存在**

Run: `ls -la /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1`
Expected: 所有文件列表：
- README.md
- .env.example
- .env
- config.py
- logger.py
- main.py
- chapter1_env_setup.py
- chapter2_model_io.py
- chapter3_prompts.py
- chapter4_chains.py
- chapter5_memory.py
- chapter6_integration.py

**Step 2: 运行第一章验证**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/chapter1_env_setup.py`
Expected: 环境检查通过，第一个程序运行成功

**Step 3: 运行主程序验证**

Run: `python /0-gaoth/projects/202605/202605-ai-study/src/demos/20260509-demo1-LangChainDemo1/main.py --chapter 2`
Expected: 第二章运行成功

---

## 成功标准

- ✅ 所有文件创建成功
- ✅ 配置管理模块正常工作
- ✅ 日志记录功能完整
- ✅ 每个章节都能独立运行
- ✅ 错误处理完善，提示友好
- ✅ 注释清晰，适合初学者
- ✅ API 调用正常，使用阿里云百炼
- ✅ README 文档完整

## 注意事项

1. **API Key 安全**：.env 文件不应提交到版本控制
2. **错误提示友好**：所有错误都应提供解决建议
3. **日志详细**：便于调试和学习
4. **代码可读性**：优先考虑初学者的理解
5. **渐进式学习**：章节之间有递进关系
