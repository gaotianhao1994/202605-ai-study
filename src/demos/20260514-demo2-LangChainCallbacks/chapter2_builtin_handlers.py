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
