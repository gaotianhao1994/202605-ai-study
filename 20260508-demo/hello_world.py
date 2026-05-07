#!/usr/bin/env python3
"""
Hello World 演示程序
这是一个用于PR学习的简单示例
"""

def greet(name: str = "Everyone") -> str:
    """
    生成问候语
    
    Args:
        name: 要问候的名字，默认为 "Everyone"
    
    Returns:
        问候语字符串
    """
    return f"Greetings, {name}!"

def main():
    """主函数"""
    print("=" * 30)
    print("  Greetings Demo")
    print("=" * 30)
    
    # 测试基本问候
    print("\n1. 基本问候:")
    print(greet())
    
    # 测试个性化问候
    print("\n2. 个性化问候:")
    print(greet("GitHub"))
    print(greet("Pull Request"))
    
    print("\n✓ 演示完成！")

if __name__ == "__main__":
    main()
