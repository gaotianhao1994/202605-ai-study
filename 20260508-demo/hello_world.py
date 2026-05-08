#!/usr/bin/env python3
"""
Hello World 演示程序
这是一个用于PR学习的简单示例
支持中英文双语问候
"""
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def greet(name: str = "World") -> str:
    """
    生成问候语
    
    Args:
        name: 要问候的名字，默认为 "World"
    
    Returns:
        问候语字符串（中英文）
    """
    logger.info(f"Generating greeting for: {name}")
    return f"Hello, {name}! / 你好，{name}！"

def main():
    """主函数"""
    logger.info("Starting the greeting demo")
    
    print("=" * 30)
    print("  Greetings Demo / 你好世界 演示程序")
    print("=" * 30)
    
    # 测试基本问候
    print("\n1. 基本问候:")
    print(greet())
    
    # 测试个性化问候
    print("\n2. 个性化问候:")
    print(greet("GitHub"))
    print(greet("Pull Request"))
    
    logger.info("Greeting demo completed successfully")
    print("\n✓ 演示完成！")

if __name__ == "__main__":
    main()
