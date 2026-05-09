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
