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
from typing import Any, Dict, List
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
