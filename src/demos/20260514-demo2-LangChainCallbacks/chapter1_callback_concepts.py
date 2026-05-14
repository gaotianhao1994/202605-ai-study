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
