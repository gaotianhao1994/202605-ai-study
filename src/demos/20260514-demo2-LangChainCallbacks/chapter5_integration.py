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
