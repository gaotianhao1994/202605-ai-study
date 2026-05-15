"""
RAG智能问答系统 - 主程序入口

提供命令行界面，选择运行哪个章节的演示

运行方式：
    python main.py                  # 交互式模式
    python main.py --chapter 1      # 运行第一章
    python main.py -c 3             # 简写形式
    python main.py --all            # 运行所有章节
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
    print("RAG智能问答系统 - LangChain教程")
    print("=" * 60)
    print("\n章节列表:")
    print("-" * 60)
    print("1. 文档加载与预处理")
    print("2. 文本分割策略")
    print("3. 向量嵌入与FAISS存储")
    print("4. 检索器与相似度搜索")
    print("5. 问答链构建与优化")
    print("0. 退出")
    print("-" * 60)
    print("\n提示：建议按顺序学习，章节之间有递进关系")


def run_chapter(chapter: int):
    """
    运行指定章节

    Args:
        chapter: 章节编号（1-5）
    """
    logger.info(f"运行第 {chapter} 章")

    chapters = {
        1: 'chapter1_document_loader',
        2: 'chapter2_text_splitter',
        3: 'chapter3_vector_store',
        4: 'chapter4_retriever',
        5: 'chapter5_qa_chain'
    }

    if chapter not in chapters:
        logger.error(f"无效的章节编号: {chapter}")
        print(f"无效的章节编号: {chapter}")
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
        print(f"\n运行第 {chapter} 章失败: {e}")


def run_all_chapters():
    """运行所有章节"""
    logger.info("运行所有章节")

    print("\n" + "=" * 60)
    print("运行所有章节")
    print("=" * 60)

    for chapter in range(1, 6):
        run_chapter(chapter)

        if chapter < 5:
            input("\n按 Enter 键继续下一章...")

    print("\n" + "=" * 60)
    print("所有章节运行完成！")
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
                print("\n再见！")
                break

            chapter = int(choice)

            if 1 <= chapter <= 5:
                run_chapter(chapter)
            else:
                print("请输入 0-5 之间的数字")

        except ValueError:
            logger.warning("用户输入无效")
            print("请输入有效的数字")
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            print("\n\n再见！")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='RAG智能问答系统 - LangChain教程',
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
        print(f"\n程序运行失败: {e}")
        sys.exit(1)

    logger.info("程序结束")


if __name__ == '__main__':
    main()
