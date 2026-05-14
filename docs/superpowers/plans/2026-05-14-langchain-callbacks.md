# LangChain Callbacks 回调系统学习项目 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/demos/` 下创建 `20260514-demo2-LangChainCallbacks` 学习项目，5 章渐进式教学 LangChain Callbacks 回调系统。

**Architecture:** 章节式 Python 项目，每章一个 .py 文件，共享 config.py 和 logger.py，使用阿里云百炼 API，遵循第一性原理思维规则（是什么→为什么→追问）。

**Tech Stack:** Python 3.8+, langchain, langchain-openai, python-dotenv, 阿里云百炼 API

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/demos/20260514-demo2-LangChainCallbacks/README.md` | 项目说明文档 |
| `src/demos/20260514-demo2-LangChainCallbacks/.env.example` | 环境变量模板 |
| `src/demos/20260514-demo2-LangChainCallbacks/config.py` | 配置管理（复用已有模式） |
| `src/demos/20260514-demo2-LangChainCallbacks/logger.py` | 日志配置（复用已有模式） |
| `src/demos/20260514-demo2-LangChainCallbacks/main.py` | 主程序入口（交互式菜单） |
| `src/demos/20260514-demo2-LangChainCallbacks/chapter1_callback_concepts.py` | 第一章：回调概念与环境准备 |
| `src/demos/20260514-demo2-LangChainCallbacks/chapter2_builtin_handlers.py` | 第二章：内置回调处理器 |
| `src/demos/20260514-demo2-LangChainCallbacks/chapter3_custom_handler.py` | 第三章：自定义回调处理器 |
| `src/demos/20260514-demo2-LangChainCallbacks/chapter4_async_callbacks.py` | 第四章：异步回调与多回调组合 |
| `src/demos/20260514-demo2-LangChainCallbacks/chapter5_integration.py` | 第五章：综合实战 |

---

### Task 1: 创建项目目录与基础设施文件

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/.env.example`
- Create: `src/demos/20260514-demo2-LangChainCallbacks/config.py`
- Create: `src/demos/20260514-demo2-LangChainCallbacks/logger.py`

- [ ] **Step 1: 创建项目目录**

```bash
mkdir -p src/demos/20260514-demo2-LangChainCallbacks
```

- [ ] **Step 2: 创建 .env.example**

```bash
cat > src/demos/20260514-demo2-LangChainCallbacks/.env.example << 'EOF'
# 阿里云百炼 API 配置
# API Key - 从阿里云百炼控制台获取
OPENAI_API_KEY=your-api-key-here

# 阿里云百炼 API 端点
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 使用的模型名称
OPENAI_MODEL_NAME=qwen3.5-122b-a10b
EOF
```

- [ ] **Step 3: 创建 config.py**

```python
"""
配置管理模块

负责加载环境变量和提供配置信息
支持阿里云百炼 API 配置

作者：AI Study Project
日期：2026-05-14
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
        self.env_file = env_file
        self._api_key: Optional[str] = None
        self._api_base: Optional[str] = None
        self._model_name: Optional[str] = None
        
        self.load_env()
    
    def load_env(self) -> None:
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
        if not self._api_key:
            raise ConfigError(
                "OPENAI_API_KEY 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_KEY"
            )
        return self._api_key
    
    def get_api_base(self) -> str:
        if not self._api_base:
            raise ConfigError(
                "OPENAI_API_BASE 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_BASE"
            )
        return self._api_base
    
    def get_model_name(self) -> str:
        if not self._model_name:
            raise ConfigError(
                "OPENAI_MODEL_NAME 未配置\n"
                "请在 .env 文件中设置 OPENAI_MODEL_NAME"
            )
        return self._model_name
    
    def validate(self) -> bool:
        try:
            self.get_api_key()
            self.get_api_base()
            self.get_model_name()
            return True
        except ConfigError as e:
            raise ConfigError(f"配置验证失败: {e}")
    
    def get_model_config(self) -> Dict[str, str]:
        return {
            'model': self.get_model_name(),
            'openai_api_base': self.get_api_base(),
            'openai_api_key': self.get_api_key()
        }
    
    def __repr__(self) -> str:
        api_key_preview = f"{self._api_key[:10]}..." if self._api_key else "未设置"
        return (
            f"Config(\n"
            f"  api_key='{api_key_preview}',\n"
            f"  api_base='{self._api_base}',\n"
            f"  model_name='{self._model_name}'\n"
            f")"
        )


def get_config(env_file: Optional[str] = None) -> Config:
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

- [ ] **Step 4: 创建 logger.py**

```python
"""
日志配置模块

提供统一的日志记录功能
支持控制台和文件输出

作者：AI Study Project
日期：2026-05-14
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
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


if __name__ == '__main__':
    logger = setup_logger('test_logger', log_file='logs/test.log')
    
    logger.debug('这是一条调试信息')
    logger.info('这是一条普通信息')
    logger.warning('这是一条警告信息')
    logger.error('这是一条错误信息')
    
    print("\n✅ 日志模块测试完成")
```

- [ ] **Step 5: 验证基础设施文件**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -c "import sys; sys.path.insert(0, 'src/demos/20260514-demo2-LangChainCallbacks'); from logger import setup_logger; logger = setup_logger('test'); logger.info('Logger OK')"
```

Expected: 输出包含 `Logger OK` 的日志行

- [ ] **Step 6: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/.env.example src/demos/20260514-demo2-LangChainCallbacks/config.py src/demos/20260514-demo2-LangChainCallbacks/logger.py
git commit -m "feat: add LangChain Callbacks project infrastructure files"
```

---

### Task 2: 创建第一章 — 回调系统概念与环境准备

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/chapter1_callback_concepts.py`

- [ ] **Step 1: 创建 chapter1_callback_concepts.py**

```python
"""
第一章：回调系统概念与环境准备

学习目标：
- 理解回调机制的概念与必要性
- 理解回调与直接调用的本质区别
- 搭建环境并运行第一个带回调的程序

作者：AI Study Project
日期：2026-05-14
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler

logger = setup_logger('chapter1_callback_concepts')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    logger.debug(f"创建模型实例: model={model_config['model']}, temperature={temperature}")
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info("模型实例创建成功")
    return llm


def explain_callback_concept():
    """
    讲解回调机制的概念
    
    ### 知识点：回调（Callback）
    是什么？ 回调是一种"你先做，做完告诉我"的模式。
    调用者不需要主动轮询结果，而是预先注册一个函数，
    当特定事件发生时，系统自动调用这个函数。
    
    为什么？ LLM 调用耗时长（几秒到几十秒）、不可预测，
    我们需要在调用的不同阶段（开始、进行中、结束、出错）
    执行自定义逻辑，比如记录日志、显示进度、触发告警。
    
    追问：为什么不直接在调用前后加代码？
    - 直接加代码是"侵入式"的 → 每次新需求都要改业务代码
    - 回调是"非侵入式"的 → 把"做什么"和"什么时候做"解耦
    - 回调可以随时添加/移除，不影响核心业务逻辑
    """
    print("\n" + "=" * 60)
    print("📖 回调机制的概念")
    print("=" * 60)
    
    print("""
### 知识点：回调（Callback）

是什么？
  回调是一种"你先做，做完告诉我"的模式。
  调用者预先注册一个函数，当特定事件发生时，系统自动调用它。

为什么？
  LLM 调用耗时长（几秒到几十秒）、不可预测，
  我们需要在调用的不同阶段执行自定义逻辑：
  - 开始时：记录开始时间
  - 进行中：显示生成进度
  - 结束时：记录结果和耗时
  - 出错时：触发告警

追问：为什么不直接在调用前后加代码？
  - 直接加代码是"侵入式"的 → 每次新需求都要改业务代码
  - 回调是"非侵入式"的 → 把"做什么"和"什么时候做"解耦
  - 回调可以随时添加/移除，不影响核心业务逻辑

类比：
  直接调用 = 你站在门口等快递
  回调     = 你留了电话，快递到了打给你
""")
    
    logger.info("回调概念讲解完成")


def show_callback_lifecycle():
    """
    展示回调的生命周期
    
    ### 知识点：回调生命周期
    是什么？ LangChain 定义了一组标准的事件触发点，
    每个事件对应一个回调方法。
    
    为什么？ 标准化的生命周期让开发者可以在精确的时间点
    插入自定义逻辑，而不需要理解内部实现。
    
    追问：为什么是这几个事件？
    - on_llm_start/end/error：覆盖了调用的完整生命周期
    - on_llm_new_token：流式输出需要逐 token 处理
    - on_chain_start/end/error：链式调用需要追踪执行流
    """
    print("\n" + "=" * 60)
    print("📖 回调的生命周期")
    print("=" * 60)
    
    print("""
### 知识点：回调生命周期

LLM 调用生命周期：
  on_llm_start  → LLM 开始调用
  on_llm_new_token → 生成新 token（流式输出时）
  on_llm_end    → LLM 调用完成
  on_llm_error  → LLM 调用出错

链式调用生命周期：
  on_chain_start → 链开始执行
  on_chain_end   → 链执行完成
  on_chain_error → 链执行出错

工具调用生命周期：
  on_tool_start → 工具开始执行
  on_tool_end   → 工具执行完成
  on_tool_error → 工具执行出错

追问：为什么每个组件类型都有独立的回调方法？
  不同组件的输入输出结构不同：
  - LLM 的输入是 prompt，输出是消息
  - Chain 的输入是字典，输出也是字典
  - Tool 的输入是字符串，输出也是字符串
  独立方法可以提供类型安全的参数，而不是用通用字典
""")
    
    logger.info("回调生命周期讲解完成")


def demo_without_callback():
    """
    演示不带回调的 LLM 调用
    
    对比：不带回调时，调用过程是"黑盒"
    """
    print("\n" + "=" * 60)
    print("演示 1: 不带回调的 LLM 调用")
    print("=" * 60)
    
    logger.info("开始演示不带回调的调用")
    
    try:
        llm = create_model(temperature=0.7)
        
        prompt = "请用一句话解释什么是回调函数"
        
        print(f"\n提示词: {prompt}")
        print("\n（调用中，无法看到内部过程...）\n")
        
        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed_time = time.time() - start_time
        
        print("AI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n耗时: {elapsed_time:.2f}s")
        print("\n⚠️ 注意：我们只能看到最终结果，无法了解调用过程")
        
        logger.info(f"不带回调的调用完成，耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_with_callback():
    """
    演示带回调的 LLM 调用
    
    使用 StdOutCallbackHandler 观察调用过程
    """
    print("\n" + "=" * 60)
    print("演示 2: 带回调的 LLM 调用（StdOutCallbackHandler）")
    print("=" * 60)
    
    logger.info("开始演示带回调的调用")
    
    try:
        handler = StdOutCallbackHandler()
        
        llm = create_model(temperature=0.7)
        
        prompt = "请用一句话解释什么是回调函数"
        
        print(f"\n提示词: {prompt}")
        print("\n（现在可以看到调用的内部过程 ↓）\n")
        
        start_time = time.time()
        response = llm.invoke(prompt, config={"callbacks": [handler]})
        elapsed_time = time.time() - start_time
        
        print("\n" + "-" * 60)
        print("AI 的回答:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\n耗时: {elapsed_time:.2f}s")
        print("\n✅ 对比：带回调时，我们可以看到调用的完整过程！")
        
        logger.info(f"带回调的调用完成，耗时 {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_callback_vs_direct():
    """
    对比回调与直接调用的差异
    
    ### 知识点：回调 vs 直接调用
    是什么？ 回调是"声明式"的，直接调用是"命令式"的。
    为什么？ 回调解耦了"做什么"和"什么时候做"，
    让代码更灵活、更可维护。
    """
    print("\n" + "=" * 60)
    print("演示 3: 回调 vs 直接调用 — 对比")
    print("=" * 60)
    
    print("""
对比总结：

┌──────────────┬──────────────────┬──────────────────┐
│              │ 直接调用          │ 回调              │
├──────────────┼──────────────────┼──────────────────┤
│ 代码位置     │ 嵌入业务代码中    │ 独立的处理器类    │
│ 添加新逻辑   │ 修改业务代码      │ 添加新回调处理器  │
│ 可复用性     │ 低               │ 高               │
│ 可组合性     │ 困难             │ 轻松组合多个回调  │
│ 侵入性       │ 高               │ 低               │
└──────────────┴──────────────────┴──────────────────┘

追问：为什么"可组合性"很重要？
  在生产环境中，你可能同时需要：
  - 日志记录回调
  - 性能监控回调
  - 错误告警回调
  用回调，只需把它们放进 callbacks 列表；
  用直接调用，你得在业务代码里到处加逻辑。
""")
    
    logger.info("回调对比讲解完成")


def main():
    """
    主函数：运行第一章所有演示
    
    执行顺序：
    1. 回调概念讲解
    2. 回调生命周期讲解
    3. 不带回调的调用演示
    4. 带回调的调用演示
    5. 回调 vs 直接调用对比
    """
    print("=" * 60)
    print("📞 第一章：回调系统概念与环境准备")
    print("=" * 60)
    
    logger.info("开始第一章演示...")
    
    demos = [
        ("回调概念讲解", explain_callback_concept),
        ("回调生命周期", show_callback_lifecycle),
        ("不带回调的调用", demo_without_callback),
        ("带回调的调用", demo_with_callback),
        ("回调 vs 直接调用", demo_callback_vs_direct),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 第一章演示完成！")
    print("=" * 60)
    
    logger.info("第一章演示完成")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 验证第一章文件语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter1_callback_concepts.py
```

Expected: 无输出（编译通过）

- [ ] **Step 3: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/chapter1_callback_concepts.py
git commit -m "feat: add chapter 1 - callback concepts and environment setup"
```

---

### Task 3: 创建第二章 — 内置回调处理器

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/chapter2_builtin_handlers.py`

- [ ] **Step 1: 创建 chapter2_builtin_handlers.py**

```python
"""
第二章：内置回调处理器

学习目标：
- 掌握 LangChain 内置回调处理器
- 理解不同处理器的适用场景
- 学习回调的配置方式和传播机制

作者：AI Study Project
日期：2026-05-14
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter2_builtin_handlers')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info("模型实例创建成功")
    return llm


def demo_stdout_handler():
    """
    演示 StdOutCallbackHandler
    
    ### 知识点：StdOutCallbackHandler
    是什么？ 将回调事件输出到控制台的内置处理器。
    为什么？ 最简单的调试方式，无需额外配置即可观察调用过程。
    追问：为什么调试时优先用 StdOutCallbackHandler？
    - 零配置，开箱即用
    - 输出格式清晰，包含完整的调用信息
    - 不影响业务逻辑，调试完可以轻松移除
    """
    print("\n" + "=" * 60)
    print("演示 1: StdOutCallbackHandler — 控制台输出")
    print("=" * 60)
    
    logger.info("开始演示 StdOutCallbackHandler")
    
    try:
        handler = StdOutCallbackHandler()
        llm = create_model(temperature=0.7)
        
        print("\n使用 StdOutCallbackHandler 观察调用过程 ↓\n")
        
        response = llm.invoke(
            "请用一句话说明回调处理器的作用",
            config={"callbacks": [handler]}
        )
        
        print(f"\n回答: {response.content}")
        print("\n✅ StdOutCallbackHandler 将调用过程输出到了控制台")
        
        logger.info("StdOutCallbackHandler 演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_stdout_handler_with_chain():
    """
    演示 StdOutCallbackHandler 在链式调用中的使用
    
    展示回调如何自动传播到链中的每个组件
    """
    print("\n" + "=" * 60)
    print("演示 2: StdOutCallbackHandler 在链式调用中")
    print("=" * 60)
    
    logger.info("开始演示链式调用中的 StdOutCallbackHandler")
    
    try:
        handler = StdOutCallbackHandler()
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请将以下文本翻译成英文：\n\n{text}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        print("\n链结构: prompt | llm | parser")
        print("使用 StdOutCallbackHandler 观察整个链的执行 ↓\n")
        
        result = chain.invoke(
            {"text": "回调处理器是 LangChain 中的中间件机制"},
            config={"callbacks": [handler]}
        )
        
        print(f"\n翻译结果: {result}")
        print("\n✅ 回调自动传播到了链中的每个组件")
        
        logger.info("链式调用回调演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_callback_config_methods():
    """
    演示回调的两种配置方式
    
    ### 知识点：回调配置方式
    是什么？ LangChain 提供两种方式传入回调：
    1. 构造函数 callbacks 参数
    2. with_config() 方法
    
    为什么？ 两种方式适用于不同场景：
    - 构造函数：回调固定不变时使用
    - with_config()：需要动态组合回调时使用
    
    追问：为什么 with_config() 更灵活？
    - with_config() 返回新的 Runnable，不修改原始对象
    - 可以链式调用，逐步添加配置
    - 符合不可变对象的设计原则
    """
    print("\n" + "=" * 60)
    print("演示 3: 回调的两种配置方式")
    print("=" * 60)
    
    logger.info("开始演示回调配置方式")
    
    try:
        handler = StdOutCallbackHandler()
        llm = create_model(temperature=0.7)
        
        print("\n--- 方式 1: 通过 config 参数传入 ---\n")
        
        response1 = llm.invoke(
            "请说'方式1成功'",
            config={"callbacks": [handler]}
        )
        print(f"结果: {response1.content}")
        
        print("\n--- 方式 2: 通过 with_config() 方法 ---\n")
        
        llm_with_callback = llm.with_config(callbacks=[handler])
        response2 = llm_with_callback.invoke("请说'方式2成功'")
        print(f"结果: {response2.content}")
        
        print("""
对比总结：

方式 1: config={"callbacks": [handler]}
  - 每次调用时指定
  - 适合临时使用不同回调

方式 2: llm.with_config(callbacks=[handler])
  - 创建带回调的新实例
  - 适合多次使用同一组回调
  - 不修改原始 llm 对象
""")
        
        logger.info("回调配置方式演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_callback_propagation():
    """
    演示回调的传播机制
    
    ### 知识点：回调传播
    是什么？ 父组件的回调会自动传播给子组件。
    为什么？ 链式调用中，用户只需要在最外层设置回调，
    不需要为每个子组件单独设置。
    追问：为什么自动传播是重要的？
    - 减少重复配置，降低出错概率
    - 保证整个调用链的监控完整性
    - 用户不需要知道链的内部结构
    """
    print("\n" + "=" * 60)
    print("演示 4: 回调的传播机制")
    print("=" * 60)
    
    logger.info("开始演示回调传播机制")
    
    try:
        handler = StdOutCallbackHandler()
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请用三个关键词概括：{topic}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        print("""
回调传播规则：
  - 在最外层（chain）设置回调
  - 回调自动传播到 prompt、llm、parser 所有子组件
  - 无需为每个组件单独设置回调
""")
        
        print("只在 chain 层面设置回调 ↓\n")
        
        chain_with_callback = chain.with_config(callbacks=[handler])
        
        result = chain_with_callback.invoke({"topic": "人工智能"})
        
        print(f"\n结果: {result}")
        print("\n✅ 回调从 chain 自动传播到了 prompt、llm 和 parser")
        
        logger.info("回调传播机制演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def main():
    """
    主函数：运行第二章所有演示
    
    执行顺序：
    1. StdOutCallbackHandler 基础使用
    2. StdOutCallbackHandler 在链式调用中
    3. 回调的两种配置方式
    4. 回调的传播机制
    """
    print("=" * 60)
    print("🔧 第二章：内置回调处理器")
    print("=" * 60)
    
    logger.info("开始第二章演示...")
    
    demos = [
        ("StdOutCallbackHandler 基础", demo_stdout_handler),
        ("链式调用中的回调", demo_stdout_handler_with_chain),
        ("回调配置方式", demo_callback_config_methods),
        ("回调传播机制", demo_callback_propagation),
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

- [ ] **Step 2: 验证语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter2_builtin_handlers.py
```

Expected: 无输出（编译通过）

- [ ] **Step 3: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/chapter2_builtin_handlers.py
git commit -m "feat: add chapter 2 - built-in callback handlers"
```

---

### Task 4: 创建第三章 — 自定义回调处理器

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/chapter3_custom_handler.py`

- [ ] **Step 1: 创建 chapter3_custom_handler.py**

```python
"""
第三章：自定义回调处理器

学习目标：
- 掌握继承 BaseCallbackHandler 创建自定义处理器
- 理解各生命周期方法的触发时机和参数
- 实现日志、监控、审计等常见场景

作者：AI Study Project
日期：2026-05-14
"""

import sys
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter3_custom_handler')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info("模型实例创建成功")
    return llm


def explain_base_callback_handler():
    """
    讲解 BaseCallbackHandler 的核心方法
    
    ### 知识点：BaseCallbackHandler
    是什么？ LangChain 提供的回调基类，定义了所有可覆盖的生命周期方法。
    为什么？ 继承它只需覆盖你关心的方法，其他方法默认空实现。
    追问：为什么用继承而不是接口？
    - Python 没有强制接口机制
    - 继承提供了默认空实现，开发者只需覆盖关心的方法
    - 这是"模板方法模式"的经典应用
    """
    print("\n" + "=" * 60)
    print("📖 BaseCallbackHandler 核心方法")
    print("=" * 60)
    
    print("""
### 知识点：BaseCallbackHandler

是什么？
  LangChain 提供的回调基类，定义了所有可覆盖的生命周期方法。
  继承它只需覆盖你关心的方法，其他方法默认空实现。

为什么？
  不是每个回调处理器都需要处理所有事件。
  基类提供默认空实现，开发者只需覆盖关心的方法。

追问：为什么用继承而不是接口？
  - Python 没有强制接口机制
  - 继承提供了默认空实现，开发者只需覆盖关心的方法
  - 这是"模板方法模式"的经典应用

核心方法一览：

LLM 相关：
  on_llm_start(serialized, prompts, **kwargs)
    → LLM 开始调用时触发
  on_llm_end(response, **kwargs)
    → LLM 调用完成时触发
  on_llm_error(error, **kwargs)
    → LLM 调用出错时触发
  on_llm_new_token(token, **kwargs)
    → 流式输出每个新 token 时触发

Chain 相关：
  on_chain_start(serialized, inputs, **kwargs)
    → 链开始执行时触发
  on_chain_end(outputs, **kwargs)
    → 链执行完成时触发
  on_chain_error(error, **kwargs)
    → 链执行出错时触发

Tool 相关：
  on_tool_start(serialized, input_str, **kwargs)
    → 工具开始执行时触发
  on_tool_end(output, **kwargs)
    → 工具执行完成时触发
  on_tool_error(error, **kwargs)
    → 工具执行出错时触发
""")
    
    logger.info("BaseCallbackHandler 讲解完成")


class TimingCallbackHandler(BaseCallbackHandler):
    """
    计时回调处理器
    
    记录每次 LLM 调用的耗时
    
    ### 知识点：为什么需要计时回调？
    是什么？ 在 on_llm_start 记录开始时间，在 on_llm_end 计算耗时。
    为什么？ LLM 调用延迟直接影响用户体验，需要监控。
    追问：为什么不直接在业务代码里计时？
    - 回调自动覆盖所有调用点，不会遗漏
    - 业务代码不需要修改，计时逻辑独立维护
    """
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self.call_times: List[float] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times[run_id] = time.time()
        print(f"  ⏱️  [TimingCallback] LLM 调用开始 (run_id: {run_id[:8]}...)")
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._start_times.pop(run_id, None)
        
        if start_time:
            elapsed = time.time() - start_time
            self.call_times.append(elapsed)
            print(f"  ⏱️  [TimingCallback] LLM 调用完成，耗时: {elapsed:.2f}s")
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times.pop(run_id, None)
        print(f"  ⏱️  [TimingCallback] LLM 调用出错: {error}")
    
    def get_summary(self) -> str:
        if not self.call_times:
            return "暂无调用记录"
        
        avg = sum(self.call_times) / len(self.call_times)
        return (
            f"调用次数: {len(self.call_times)}, "
            f"平均耗时: {avg:.2f}s, "
            f"最小: {min(self.call_times):.2f}s, "
            f"最大: {max(self.call_times):.2f}s"
        )


class TokenCountCallbackHandler(BaseCallbackHandler):
    """
    Token 计数回调处理器
    
    统计 LLM 调用的 Token 使用量
    
    ### 知识点：为什么需要 Token 计数？
    是什么？ 在 on_llm_end 中从 response 提取 token 使用信息。
    为什么？ Token 用量直接关联 API 费用，需要追踪。
    追问：为什么 token 信息在 response 的 llm_output 里？
    - OpenAI API 在响应中返回 token 统计
    - LangChain 将其透传到 LLMResult.llm_output
    - 不同模型提供商的 token 计算方式不同，统一放在这里
    """
    
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        self.call_count += 1
        
        for generation in response.flatten():
            info = generation.generation_info or {}
            token_usage = info.get("token_usage", {})
            
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total = token_usage.get("total_tokens", 0)
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total
            
            print(f"  🔢  [TokenCount] 本次: prompt={prompt_tokens}, "
                  f"completion={completion_tokens}, total={total}")
    
    def get_summary(self) -> str:
        return (
            f"调用次数: {self.call_count}, "
            f"总 Prompt Tokens: {self.total_prompt_tokens}, "
            f"总 Completion Tokens: {self.total_completion_tokens}, "
            f"总 Tokens: {self.total_tokens}"
        )


class AuditLogCallbackHandler(BaseCallbackHandler):
    """
    审计日志回调处理器
    
    记录谁在什么时候调用了什么，用于审计追踪
    
    ### 知识点：为什么需要审计日志？
    是什么？ 记录每次 LLM 调用的输入、输出和时间戳。
    为什么？ 合规要求、安全审计、问题排查都需要完整的调用记录。
    追问：为什么审计日志用回调而不是在业务代码中记录？
    - 回调保证每次调用都被记录，不会遗漏
    - 审计逻辑与业务逻辑解耦，互不影响
    - 可以独立修改审计格式，不影响业务代码
    """
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        entry = {
            "event": "llm_start",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": serialized.get("name", "unknown"),
            "prompt_preview": prompts[0][:50] + "..." if prompts else "",
        }
        self.audit_log.append(entry)
        print(f"  📋  [AuditLog] 记录开始调用: model={entry['model']}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        entry = {
            "event": "llm_end",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.audit_log.append(entry)
        print(f"  📋  [AuditLog] 记录调用完成")
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        entry = {
            "event": "llm_error",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": type(error).__name__,
            "error_message": str(error)[:100],
        }
        self.audit_log.append(entry)
        print(f"  📋  [AuditLog] 记录调用错误: {entry['error_type']}")
    
    def print_log(self):
        print("\n  审计日志记录:")
        print("  " + "-" * 50)
        for i, entry in enumerate(self.audit_log, 1):
            print(f"  {i}. [{entry['timestamp']}] {entry['event']}")
            if "model" in entry:
                print(f"     模型: {entry['model']}")
            if "prompt_preview" in entry:
                print(f"     提示词预览: {entry['prompt_preview']}")
            if "error_type" in entry:
                print(f"     错误类型: {entry['error_type']}")
        print("  " + "-" * 50)


def demo_timing_handler():
    """演示计时回调处理器"""
    print("\n" + "=" * 60)
    print("演示 1: TimingCallbackHandler — 调用计时")
    print("=" * 60)
    
    logger.info("开始演示计时回调处理器")
    
    try:
        timing_handler = TimingCallbackHandler()
        llm = create_model(temperature=0.7)
        
        prompts = [
            "请用一句话介绍 Python",
            "请用一句话介绍 LangChain",
        ]
        
        for prompt in prompts:
            print(f"\n提示词: {prompt}")
            response = llm.invoke(
                prompt,
                config={"callbacks": [timing_handler]}
            )
            print(f"回答: {response.content[:100]}...")
        
        print(f"\n计时统计: {timing_handler.get_summary()}")
        print("\n✅ TimingCallbackHandler 成功记录了每次调用的耗时")
        
        logger.info("计时回调演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_token_count_handler():
    """演示 Token 计数回调处理器"""
    print("\n" + "=" * 60)
    print("演示 2: TokenCountCallbackHandler — Token 计数")
    print("=" * 60)
    
    logger.info("开始演示 Token 计数回调处理器")
    
    try:
        token_handler = TokenCountCallbackHandler()
        llm = create_model(temperature=0.7)
        
        prompts = [
            "请用一句话介绍机器学习",
            "请用一句话介绍深度学习",
        ]
        
        for prompt in prompts:
            print(f"\n提示词: {prompt}")
            response = llm.invoke(
                prompt,
                config={"callbacks": [token_handler]}
            )
            print(f"回答: {response.content[:100]}...")
        
        print(f"\nToken 统计: {token_handler.get_summary()}")
        print("\n✅ TokenCountCallbackHandler 成功统计了 Token 使用量")
        
        logger.info("Token 计数演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_audit_log_handler():
    """演示审计日志回调处理器"""
    print("\n" + "=" * 60)
    print("演示 3: AuditLogCallbackHandler — 审计日志")
    print("=" * 60)
    
    logger.info("开始演示审计日志回调处理器")
    
    try:
        audit_handler = AuditLogCallbackHandler()
        llm = create_model(temperature=0.7)
        
        prompt = "请用一句话解释什么是审计日志"
        print(f"\n提示词: {prompt}")
        
        response = llm.invoke(
            prompt,
            config={"callbacks": [audit_handler]}
        )
        
        print(f"回答: {response.content[:100]}...")
        
        audit_handler.print_log()
        print("\n✅ AuditLogCallbackHandler 成功记录了审计日志")
        
        logger.info("审计日志演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_custom_handler_with_chain():
    """演示自定义回调处理器在链式调用中的使用"""
    print("\n" + "=" * 60)
    print("演示 4: 自定义回调处理器 + 链式调用")
    print("=" * 60)
    
    logger.info("开始演示自定义回调处理器与链式调用")
    
    try:
        timing_handler = TimingCallbackHandler()
        token_handler = TokenCountCallbackHandler()
        
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请将以下文本翻译成英文：\n\n{text}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        print("\n同时使用计时和 Token 计数两个回调 ↓\n")
        
        result = chain.invoke(
            {"text": "回调处理器让代码更灵活、更可维护"},
            config={"callbacks": [timing_handler, token_handler]}
        )
        
        print(f"\n翻译结果: {result}")
        print(f"\n计时统计: {timing_handler.get_summary()}")
        print(f"Token 统计: {token_handler.get_summary()}")
        print("\n✅ 多个自定义回调处理器在链式调用中协同工作")
        
        logger.info("自定义回调与链式调用演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def main():
    """
    主函数：运行第三章所有演示
    
    执行顺序：
    1. BaseCallbackHandler 讲解
    2. TimingCallbackHandler 演示
    3. TokenCountCallbackHandler 演示
    4. AuditLogCallbackHandler 演示
    5. 自定义回调 + 链式调用
    """
    print("=" * 60)
    print("🛠️ 第三章：自定义回调处理器")
    print("=" * 60)
    
    logger.info("开始第三章演示...")
    
    demos = [
        ("BaseCallbackHandler 讲解", explain_base_callback_handler),
        ("计时回调处理器", demo_timing_handler),
        ("Token 计数回调处理器", demo_token_count_handler),
        ("审计日志回调处理器", demo_audit_log_handler),
        ("自定义回调 + 链式调用", demo_custom_handler_with_chain),
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

- [ ] **Step 2: 验证语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter3_custom_handler.py
```

Expected: 无输出（编译通过）

- [ ] **Step 3: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/chapter3_custom_handler.py
git commit -m "feat: add chapter 3 - custom callback handlers"
```

---

### Task 5: 创建第四章 — 异步回调与多回调组合

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/chapter4_async_callbacks.py`

- [ ] **Step 1: 创建 chapter4_async_callbacks.py**

```python
"""
第四章：异步回调与多回调组合

学习目标：
- 理解同步回调 vs 异步回调的区别
- 掌握 AsyncCallbackHandler 的使用
- 学会组合多个回调处理器

作者：AI Study Project
日期：2026-05-14
"""

import sys
import time
import asyncio
from typing import Any, Dict, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter4_async_callbacks')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info("模型实例创建成功")
    return llm


def explain_async_callback():
    """
    讲解异步回调的概念
    
    ### 知识点：异步回调
    是什么？ AsyncCallbackHandler 是 BaseCallbackHandler 的异步版本，
    所有回调方法都是 async 定义的。
    
    为什么？ 同步回调在 LLM 调用期间会阻塞事件循环，
    异步回调不会。在异步应用（如 Web 服务）中，同步回调会成为性能瓶颈。
    
    追问：为什么阻塞事件循环是问题？
    - 事件循环是异步程序的"心脏"，负责调度所有协程
    - 阻塞它等于冻结了整个应用的所有并发任务
    - 在 Web 服务中，这意味着一个慢回调会阻塞所有用户请求
    """
    print("\n" + "=" * 60)
    print("📖 异步回调的概念")
    print("=" * 60)
    
    print("""
### 知识点：异步回调

是什么？
  AsyncCallbackHandler 是 BaseCallbackHandler 的异步版本，
  所有回调方法都是 async 定义的。

为什么？
  同步回调在 LLM 调用期间会阻塞事件循环，
  异步回调不会。在异步应用（如 Web 服务）中，
  同步回调会成为性能瓶颈。

追问：为什么阻塞事件循环是问题？
  - 事件循环是异步程序的"心脏"，负责调度所有协程
  - 阻塞它等于冻结了整个应用的所有并发任务
  - 在 Web 服务中，一个慢回调会阻塞所有用户请求

对比：

┌──────────────┬────────────────────┬────────────────────┐
│              │ 同步回调            │ 异步回调            │
├──────────────┼────────────────────┼────────────────────┤
│ 方法定义     │ def on_llm_start   │ async def on_llm_start │
│ 事件循环     │ 阻塞               │ 不阻塞              │
│ 适用场景     │ 脚本、CLI          │ Web 服务、API       │
│ 并发安全     │ 无需考虑           │ 需要考虑            │
└──────────────┴────────────────────┴────────────────────┘
""")
    
    logger.info("异步回调概念讲解完成")


class AsyncTimingCallbackHandler(AsyncCallbackHandler):
    """
    异步计时回调处理器
    
    与同步版本功能相同，但不会阻塞事件循环
    """
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self.call_times: List[float] = []
    
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times[run_id] = time.time()
        print(f"  ⏱️  [AsyncTiming] LLM 异步调用开始 (run_id: {run_id[:8]}...)")
    
    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._start_times.pop(run_id, None)
        
        if start_time:
            elapsed = time.time() - start_time
            self.call_times.append(elapsed)
            print(f"  ⏱️  [AsyncTiming] LLM 异步调用完成，耗时: {elapsed:.2f}s")
    
    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times.pop(run_id, None)
        print(f"  ⏱️  [AsyncTiming] LLM 异步调用出错: {error}")
    
    def get_summary(self) -> str:
        if not self.call_times:
            return "暂无调用记录"
        
        avg = sum(self.call_times) / len(self.call_times)
        return (
            f"调用次数: {len(self.call_times)}, "
            f"平均耗时: {avg:.2f}s, "
            f"最小: {min(self.call_times):.2f}s, "
            f"最大: {max(self.call_times):.2f}s"
        )


class AsyncLoggingCallbackHandler(AsyncCallbackHandler):
    """
    异步日志回调处理器
    
    异步记录调用事件
    """
    
    def __init__(self):
        self.log_entries: List[str] = []
    
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] LLM 开始: {prompts[0][:30]}..."
        self.log_entries.append(entry)
        print(f"  📝  [AsyncLogging] {entry}")
    
    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] LLM 完成"
        self.log_entries.append(entry)
        print(f"  📝  [AsyncLogging] {entry}")
    
    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] Chain 开始"
        self.log_entries.append(entry)
        print(f"  📝  [AsyncLogging] {entry}")
    
    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] Chain 完成"
        self.log_entries.append(entry)
        print(f"  📝  [AsyncLogging] {entry}")


def demo_async_callback():
    """
    演示异步回调处理器
    
    使用 ainvoke 异步调用 LLM
    """
    print("\n" + "=" * 60)
    print("演示 1: AsyncCallbackHandler — 异步计时")
    print("=" * 60)
    
    logger.info("开始演示异步回调处理器")
    
    async def run():
        timing_handler = AsyncTimingCallbackHandler()
        llm = create_model(temperature=0.7)
        
        prompt = "请用一句话介绍异步编程"
        print(f"\n提示词: {prompt}")
        print("使用异步回调处理器 ↓\n")
        
        response = await llm.ainvoke(
            prompt,
            config={"callbacks": [timing_handler]}
        )
        
        print(f"\n回答: {response.content[:100]}...")
        print(f"\n计时统计: {timing_handler.get_summary()}")
        print("\n✅ AsyncCallbackHandler 在异步调用中正常工作")
    
    try:
        asyncio.run(run())
        logger.info("异步回调演示完成")
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_async_with_chain():
    """
    演示异步回调在链式调用中的使用
    """
    print("\n" + "=" * 60)
    print("演示 2: 异步回调 + 链式调用")
    print("=" * 60)
    
    logger.info("开始演示异步回调与链式调用")
    
    async def run():
        timing_handler = AsyncTimingCallbackHandler()
        logging_handler = AsyncLoggingCallbackHandler()
        
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请用一句话概括：{topic}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        print("\n同时使用异步计时和异步日志回调 ↓\n")
        
        result = await chain.ainvoke(
            {"topic": "异步编程的优势"},
            config={"callbacks": [timing_handler, logging_handler]}
        )
        
        print(f"\n结果: {result}")
        print(f"\n计时统计: {timing_handler.get_summary()}")
        print(f"日志条数: {len(logging_handler.log_entries)}")
        print("\n✅ 异步回调在链式调用中协同工作")
    
    try:
        asyncio.run(run())
        logger.info("异步回调与链式调用演示完成")
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_multiple_callbacks():
    """
    演示多回调组合
    
    ### 知识点：多回调组合
    是什么？ 将多个回调处理器放入 callbacks 列表，它们按顺序依次执行。
    为什么？ 生产环境中通常需要同时记录日志、监控性能、审计追踪。
    追问：为什么回调按顺序执行而不是并行？
    - 回调通常很轻量（记录日志、更新计数器），串行执行开销可忽略
    - 串行执行保证顺序可预测，便于调试
    - 如果某个回调出错，不会影响其他回调
    """
    print("\n" + "=" * 60)
    print("演示 3: 多回调组合")
    print("=" * 60)
    
    logger.info("开始演示多回调组合")
    
    try:
        from chapter3_custom_handler import TimingCallbackHandler, TokenCountCallbackHandler
        
        timing_handler = TimingCallbackHandler()
        token_handler = TokenCountCallbackHandler()
        audit_handler = AuditLogCallbackHandler()
        
        llm = create_model(temperature=0.7)
        
        print("""
组合三个回调处理器：
  1. TimingCallbackHandler — 计时
  2. TokenCountCallbackHandler — Token 计数
  3. AuditLogCallbackHandler — 审计日志
""")
        
        prompt = "请用一句话解释回调组合的意义"
        print(f"提示词: {prompt}\n")
        
        response = llm.invoke(
            prompt,
            config={"callbacks": [timing_handler, token_handler, audit_handler]}
        )
        
        print(f"\n回答: {response.content[:100]}...")
        print(f"\n计时统计: {timing_handler.get_summary()}")
        print(f"Token 统计: {token_handler.get_summary()}")
        
        audit_handler.print_log()
        
        print("\n✅ 三个回调处理器协同工作，各司其职")
        
        logger.info("多回调组合演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_with_config_chain():
    """
    演示 with_config 链式配置
    
    ### 知识点：with_config 链式配置
    是什么？ 使用 with_config() 方法链式添加回调配置。
    为什么？ 可以在不同层级添加不同的回调，实现精细化的监控。
    追问：为什么不用一个大的回调列表？
    - 不同层级的关注点不同：全局监控 vs 局部调试
    - with_config 返回新对象，不修改原始对象，更安全
    - 可以按需组合，避免不必要的回调开销
    """
    print("\n" + "=" * 60)
    print("演示 4: with_config 链式配置")
    print("=" * 60)
    
    logger.info("开始演示 with_config 链式配置")
    
    try:
        from chapter3_custom_handler import TimingCallbackHandler
        
        timing_handler = TimingCallbackHandler()
        
        llm = create_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template("请用一句话介绍：{topic}")
        chain = prompt | llm | StrOutputParser()
        
        print("\n原始链 → with_config 添加计时回调 ↓\n")
        
        monitored_chain = chain.with_config(callbacks=[timing_handler])
        
        result = monitored_chain.invoke({"topic": "链式配置"})
        
        print(f"\n结果: {result}")
        print(f"计时统计: {timing_handler.get_summary()}")
        print("\n✅ with_config 创建了带回调的新链实例，原始链不受影响")
        
        logger.info("with_config 链式配置演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def main():
    """
    主函数：运行第四章所有演示
    
    执行顺序：
    1. 异步回调概念讲解
    2. AsyncCallbackHandler 演示
    3. 异步回调 + 链式调用
    4. 多回调组合
    5. with_config 链式配置
    """
    print("=" * 60)
    print("⚡ 第四章：异步回调与多回调组合")
    print("=" * 60)
    
    logger.info("开始第四章演示...")
    
    demos = [
        ("异步回调概念", explain_async_callback),
        ("AsyncCallbackHandler", demo_async_callback),
        ("异步回调 + 链式调用", demo_async_with_chain),
        ("多回调组合", demo_multiple_callbacks),
        ("with_config 链式配置", demo_with_config_chain),
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

- [ ] **Step 2: 验证语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter4_async_callbacks.py
```

Expected: 无输出（编译通过）

- [ ] **Step 3: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/chapter4_async_callbacks.py
git commit -m "feat: add chapter 4 - async callbacks and multiple callback composition"
```

---

### Task 6: 创建第五章 — 综合实战

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/chapter5_integration.py`

- [ ] **Step 1: 创建 chapter5_integration.py**

```python
"""
第五章：综合实战

学习目标：
- 综合运用所有回调知识
- 构建完整的 LLM 调用监控与审计系统
- 学习扩展和优化方法

作者：AI Study Project
日期：2026-05-14
"""

import sys
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger('chapter5_integration')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    
    logger.info("模型实例创建成功")
    return llm


class MetricsCallback(BaseCallbackHandler):
    """
    性能指标回调处理器
    
    收集延迟、Token 数、成功率等关键指标
    
    ### 知识点：为什么需要性能指标？
    是什么？ 自动收集每次 LLM 调用的性能数据。
    为什么？ 性能数据是优化的基础，没有指标就无法发现问题。
    追问：为什么用回调而不是手动统计？
    - 回调自动覆盖所有调用点，零遗漏
    - 指标收集与业务逻辑完全解耦
    - 可以随时启用/禁用，不影响业务代码
    """
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self.metrics: List[Dict[str, Any]] = []
        self.total_calls = 0
        self.success_calls = 0
        self.error_calls = 0
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times[run_id] = time.time()
        self.total_calls += 1
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._start_times.pop(run_id, None)
        
        self.success_calls += 1
        
        metric: Dict[str, Any] = {
            "status": "success",
            "latency": time.time() - start_time if start_time else -1,
        }
        
        for generation in response.flatten():
            info = generation.generation_info or {}
            token_usage = info.get("token_usage", {})
            metric["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
            metric["completion_tokens"] = token_usage.get("completion_tokens", 0)
            metric["total_tokens"] = token_usage.get("total_tokens", 0)
        
        self.metrics.append(metric)
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times.pop(run_id, None)
        
        self.error_calls += 1
        self.metrics.append({
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error)[:100],
        })
    
    def get_summary(self) -> str:
        success_metrics = [m for m in self.metrics if m["status"] == "success"]
        
        if not success_metrics:
            return "暂无成功调用记录"
        
        latencies = [m["latency"] for m in success_metrics if m["latency"] > 0]
        total_tokens = sum(m.get("total_tokens", 0) for m in success_metrics)
        
        success_rate = (self.success_calls / self.total_calls * 100
                       if self.total_calls > 0 else 0)
        
        lines = [
            f"总调用次数: {self.total_calls}",
            f"成功: {self.success_calls}, 失败: {self.error_calls}",
            f"成功率: {success_rate:.1f}%",
        ]
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            lines.extend([
                f"平均延迟: {avg_latency:.2f}s",
                f"最小延迟: {min(latencies):.2f}s",
                f"最大延迟: {max(latencies):.2f}s",
            ])
        
        lines.append(f"总 Token 数: {total_tokens}")
        
        return "\n  ".join(lines)


class AlertCallback(BaseCallbackHandler):
    """
    异常告警回调处理器
    
    当调用超时或错误率过高时触发告警
    
    ### 知识点：为什么需要告警？
    是什么？ 当指标超过阈值时自动通知。
    为什么？ 问题越早发现，影响越小。等待用户投诉是最差的监控策略。
    追问：为什么告警用回调实现？
    - 告警逻辑与业务逻辑完全分离
    - 可以独立调整告警阈值，不影响业务代码
    - 可以随时添加新的告警规则
    """
    
    def __init__(
        self,
        latency_threshold: float = 10.0,
        error_window: int = 5,
        error_rate_threshold: float = 0.5
    ):
        self.latency_threshold = latency_threshold
        self.error_window = error_window
        self.error_rate_threshold = error_rate_threshold
        
        self._start_times: Dict[str, float] = {}
        self._recent_results: List[str] = []
        self.alerts: List[str] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times[run_id] = time.time()
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._start_times.pop(run_id, None)
        
        self._recent_results.append("success")
        if len(self._recent_results) > self.error_window:
            self._recent_results.pop(0)
        
        if start_time:
            latency = time.time() - start_time
            if latency > self.latency_threshold:
                alert = f"🚨 延迟告警: {latency:.2f}s > {self.latency_threshold}s"
                self.alerts.append(alert)
                print(f"  {alert}")
        
        self._check_error_rate()
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times.pop(run_id, None)
        
        self._recent_results.append("error")
        if len(self._recent_results) > self.error_window:
            self._recent_results.pop(0)
        
        alert = f"🚨 错误告警: {type(error).__name__} - {str(error)[:50]}"
        self.alerts.append(alert)
        print(f"  {alert}")
        
        self._check_error_rate()
    
    def _check_error_rate(self):
        if len(self._recent_results) >= self.error_window:
            error_count = self._recent_results.count("error")
            error_rate = error_count / len(self._recent_results)
            
            if error_rate > self.error_rate_threshold:
                alert = (
                    f"🚨 错误率告警: 近 {self.error_window} 次调用中 "
                    f"错误率 {error_rate:.0%} > {self.error_rate_threshold:.0%}"
                )
                self.alerts.append(alert)
                print(f"  {alert}")
    
    def get_alerts_summary(self) -> str:
        if not self.alerts:
            return "无告警"
        return f"共 {len(self.alerts)} 条告警:\n  " + "\n  ".join(self.alerts)


class LoggingCallback(BaseCallbackHandler):
    """
    结构化日志回调处理器
    
    以结构化格式记录所有调用事件
    
    ### 知识点：为什么需要结构化日志？
    是什么？ 以键值对格式记录日志，而非自由文本。
    为什么？ 结构化日志可以被日志系统（ELK、Splunk）自动解析和检索。
    追问：为什么结构化比自由文本好？
    - 自由文本需要正则解析 → 脆弱、易出错
    - 结构化日志可以直接索引 → 快速检索、聚合分析
    - 是生产环境日志的最佳实践
    """
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        self.entries.append({
            "event": "llm_start",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": serialized.get("name", "unknown"),
            "prompt_length": len(prompts[0]) if prompts else 0,
        })
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        self.entries.append({
            "event": "llm_end",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generations_count": len(response.generations),
        })
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        self.entries.append({
            "event": "chain_start",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chain_name": serialized.get("name", "unknown"),
            "input_keys": list(inputs.keys()) if isinstance(inputs, dict) else [],
        })
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        self.entries.append({
            "event": "chain_end",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    
    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        self.entries.append({
            "event": "llm_error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error_type": type(error).__name__,
        })
    
    def print_structured_log(self):
        print("\n  结构化日志:")
        print("  " + "-" * 50)
        for i, entry in enumerate(self.entries, 1):
            parts = [f"{k}={v}" for k, v in entry.items()]
            print(f"  {i}. {' | '.join(parts)}")
        print("  " + "-" * 50)


def demo_monitoring_system():
    """
    演示完整的监控审计系统
    
    组合 MetricsCallback、AlertCallback、LoggingCallback
    """
    print("\n" + "=" * 60)
    print("演示 1: 完整的 LLM 调用监控审计系统")
    print("=" * 60)
    
    logger.info("开始演示完整监控审计系统")
    
    try:
        metrics = MetricsCallback()
        alerts = AlertCallback(latency_threshold=10.0)
        logging_cb = LoggingCallback()
        
        llm = create_model(temperature=0.7)
        
        test_prompts = [
            "请用一句话介绍 Python",
            "请用一句话介绍 LangChain",
            "请用一句话介绍回调机制",
        ]
        
        print("""
监控审计系统组成：
  1. MetricsCallback — 性能指标（延迟、Token、成功率）
  2. AlertCallback — 异常告警（超时、高错误率）
  3. LoggingCallback — 结构化日志
""")
        
        for prompt in test_prompts:
            print(f"\n提示词: {prompt}")
            response = llm.invoke(
                prompt,
                config={"callbacks": [metrics, alerts, logging_cb]}
            )
            print(f"回答: {response.content[:80]}...")
        
        print("\n" + "=" * 60)
        print("📊 监控报告")
        print("=" * 60)
        
        print("\n性能指标:")
        print(f"  {metrics.get_summary()}")
        
        print(f"\n告警状态:")
        print(f"  {alerts.get_alerts_summary()}")
        
        logging_cb.print_structured_log()
        
        print("\n✅ 监控审计系统正常运行，三个回调处理器协同工作")
        
        logger.info("监控审计系统演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_monitoring_with_chain():
    """
    演示监控审计系统在链式调用中的使用
    """
    print("\n" + "=" * 60)
    print("演示 2: 监控审计系统 + 链式调用")
    print("=" * 60)
    
    logger.info("开始演示监控审计系统与链式调用")
    
    try:
        metrics = MetricsCallback()
        alerts = AlertCallback(latency_threshold=10.0)
        logging_cb = LoggingCallback()
        
        llm = create_model(temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            "请将以下文本翻译成{language}：\n\n{text}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        test_cases = [
            {"language": "英文", "text": "今天天气很好"},
            {"language": "日文", "text": "人工智能改变世界"},
        ]
        
        print("\n在链式调用中使用监控审计系统 ↓\n")
        
        for case in test_cases:
            print(f"翻译到 {case['language']}: {case['text']}")
            result = chain.invoke(
                case,
                config={"callbacks": [metrics, alerts, logging_cb]}
            )
            print(f"结果: {result[:80]}...\n")
        
        print("=" * 60)
        print("📊 监控报告")
        print("=" * 60)
        
        print(f"\n性能指标:\n  {metrics.get_summary()}")
        print(f"\n告警状态:\n  {alerts.get_alerts_summary()}")
        
        logging_cb.print_structured_log()
        
        print("\n✅ 监控审计系统在链式调用中正常工作")
        
        logger.info("监控审计系统与链式调用演示完成")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        print(f"\n❌ 演示失败: {e}")


def demo_extension_ideas():
    """
    展示扩展思路
    
    ### 知识点：回调系统的扩展方向
    是什么？ 基于回调机制可以构建更强大的监控系统。
    为什么？ 生产环境的需求远超基础监控，需要持续扩展。
    追问：为什么回调是可扩展的基础？
    - 回调是"插件式"的 → 添加新功能不需要改旧代码
    - 回调是"可组合的" → 多个回调可以协同工作
    - 回调是"标准化的" → 遵循统一接口，易于集成
    """
    print("\n" + "=" * 60)
    print("📖 扩展思路")
    print("=" * 60)
    
    print("""
### 知识点：回调系统的扩展方向

是什么？
  基于回调机制可以构建更强大的监控系统。

为什么？
  生产环境的需求远超基础监控，需要持续扩展。

追问：为什么回调是可扩展的基础？
  - 回调是"插件式"的 → 添加新功能不需要改旧代码
  - 回调是"可组合的" → 多个回调可以协同工作
  - 回调是"标准化的" → 遵循统一接口，易于集成

扩展方向：

1. 对接 Prometheus + Grafana
   MetricsCallback → 暴露 /metrics 端点 → Prometheus 采集 → Grafana 可视化
   适合：需要实时仪表盘和告警规则的生产环境

2. 接入分布式追踪（OpenTelemetry）
   在 on_llm_start 创建 span，on_llm_end 关闭 span
   适合：微服务架构，需要追踪跨服务调用链

3. 成本控制
   TokenCountCallback → 累计费用 → 超预算自动降级
   适合：需要控制 API 调用成本的团队

4. A/B 测试
   在回调中记录不同 prompt 版本的效果
   适合：需要优化 prompt 策略的场景

5. 安全审计
   AuditLogCallback → 检测敏感信息泄露 → 自动脱敏
   适合：有合规要求的行业
""")
    
    logger.info("扩展思路讲解完成")


def main():
    """
    主函数：运行第五章所有演示
    
    执行顺序：
    1. 完整监控审计系统
    2. 监控审计系统 + 链式调用
    3. 扩展思路
    """
    print("=" * 60)
    print("🏗️ 第五章：综合实战")
    print("=" * 60)
    
    logger.info("开始第五章演示...")
    
    demos = [
        ("完整监控审计系统", demo_monitoring_system),
        ("监控审计 + 链式调用", demo_monitoring_with_chain),
        ("扩展思路", demo_extension_ideas),
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
    
    print("""
学习总结：

第一章：回调概念 — 理解"你先做，做完告诉我"的模式
第二章：内置处理器 — 掌握 StdOutCallbackHandler 等内置工具
第三章：自定义处理器 — 继承 BaseCallbackHandler 实现业务逻辑
第四章：异步与组合 — AsyncCallbackHandler + 多回调协同
第五章：综合实战 — 构建完整的监控审计系统

核心收获：
  ✅ 回调是 LangChain 的"中间件"机制
  ✅ 回调解耦了"做什么"和"什么时候做"
  ✅ 回调可以组合、传播、异步执行
  ✅ 基于回调可以构建强大的监控系统
""")
    
    logger.info("第五章演示完成")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 验证语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter5_integration.py
```

Expected: 无输出（编译通过）

- [ ] **Step 3: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/chapter5_integration.py
git commit -m "feat: add chapter 5 - integration with monitoring and audit system"
```

---

### Task 7: 创建 main.py 和 README.md

**Files:**
- Create: `src/demos/20260514-demo2-LangChainCallbacks/main.py`
- Create: `src/demos/20260514-demo2-LangChainCallbacks/README.md`

- [ ] **Step 1: 创建 main.py**

```python
"""
LangChain Callbacks 回调系统教程 - 主程序入口

提供命令行界面，选择运行哪个章节的演示

作者：AI Study Project
日期：2026-05-14
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
    print("📞 LangChain Callbacks 回调系统教程")
    print("=" * 60)
    print("\n章节列表:")
    print("-" * 60)
    print("1. 回调系统概念与环境准备")
    print("2. 内置回调处理器")
    print("3. 自定义回调处理器")
    print("4. 异步回调与多回调组合")
    print("5. 综合实战")
    print("0. 退出")
    print("-" * 60)


def run_chapter(chapter: int):
    """
    运行指定章节
    
    Args:
        chapter: 章节编号（1-5）
    """
    logger.info(f"运行第 {chapter} 章")
    
    chapters = {
        1: 'chapter1_callback_concepts',
        2: 'chapter2_builtin_handlers',
        3: 'chapter3_custom_handler',
        4: 'chapter4_async_callbacks',
        5: 'chapter5_integration'
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
    print("📞 运行所有章节")
    print("=" * 60)
    
    for chapter in range(1, 6):
        run_chapter(chapter)
        
        if chapter < 5:
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
            choice = input("\n请选择章节 (0-5): ").strip()
            
            if choice == '0':
                logger.info("用户选择退出")
                print("\n👋 再见！")
                break
            
            chapter = int(choice)
            
            if 1 <= chapter <= 5:
                run_chapter(chapter)
            else:
                print("❌ 请输入 0-5 之间的数字")
        
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
        description='LangChain Callbacks 回调系统教程',
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
        choices=range(1, 6),
        help='运行指定章节 (1-5)'
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

- [ ] **Step 2: 创建 README.md**

```markdown
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
```

- [ ] **Step 3: 验证 main.py 语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/main.py
```

Expected: 无输出（编译通过）

- [ ] **Step 4: 提交**

```bash
git add src/demos/20260514-demo2-LangChainCallbacks/main.py src/demos/20260514-demo2-LangChainCallbacks/README.md
git commit -m "feat: add main.py entry point and README.md"
```

---

### Task 8: 最终验证

- [ ] **Step 1: 验证所有文件存在**

```bash
ls -la src/demos/20260514-demo2-LangChainCallbacks/
```

Expected: 列出 10 个文件（README.md, .env.example, config.py, logger.py, main.py, chapter1-5）

- [ ] **Step 2: 验证所有 Python 文件语法**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/config.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/logger.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/main.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter1_callback_concepts.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter2_builtin_handlers.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter3_custom_handler.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter4_async_callbacks.py && \
python -m py_compile src/demos/20260514-demo2-LangChainCallbacks/chapter5_integration.py && \
echo "✅ All files compiled successfully"
```

Expected: `✅ All files compiled successfully`

- [ ] **Step 3: 验证模块导入链**

```bash
cd /root/projects/202605-ai-study
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'src/demos/20260514-demo2-LangChainCallbacks')
from config import get_config
from logger import setup_logger
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler, StdOutCallbackHandler
print('✅ All imports successful')
"
```

Expected: `✅ All imports successful`
