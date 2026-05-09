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
