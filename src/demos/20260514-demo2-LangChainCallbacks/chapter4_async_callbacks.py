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
        from chapter3_custom_handler import (
            TimingCallbackHandler,
            TokenCountCallbackHandler,
            AuditLogCallbackHandler
        )

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
