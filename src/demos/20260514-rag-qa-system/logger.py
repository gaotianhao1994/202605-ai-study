"""
日志配置模块

提供统一的日志记录功能，支持控制台和文件输出

知识点：为什么需要日志？
- print 的问题：无法控制级别、无法持久化、无法过滤
- 日志的层级：DEBUG → INFO → WARNING → ERROR → CRITICAL
- RAG系统特别需要日志：检索过程、相似度分数、耗时等都需要记录
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回配置好的 Logger

    Args:
        name: 日志器名称，通常使用模块名
        level: 日志级别，默认 INFO
        log_file: 日志文件路径，为 None 则只输出到控制台

    Returns:
        配置好的 Logger 实例

    为什么日志要分级别？
    - 开发时看 DEBUG → 了解每一步的细节
    - 生产时看 INFO → 只关注关键事件
    - 出问题时看 ERROR → 快速定位故障
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    logger = setup_logger('test')

    logger.debug("调试信息 - 只在开发时显示")
    logger.info("常规信息 - 记录关键步骤")
    logger.warning("警告信息 - 可能的问题")
    logger.error("错误信息 - 需要关注的问题")
