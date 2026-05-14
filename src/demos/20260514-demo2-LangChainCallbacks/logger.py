"""
日志配置模块

提供统一的日志记录功能
支持控制台和文件输出

作者：AI Study Project
日期：2026-05-14
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回 logger 实例

    Args:
        name: logger 名称
        level: 日志级别，默认 DEBUG
        log_file: 日志文件路径（可选）
        format_string: 自定义日志格式（可选）

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    formatter = logging.Formatter(
        format_string,
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
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger


if __name__ == '__main__':
    logger = setup_logger('test_logger', log_file='logs/test.log')

    logger.debug('这是一条调试信息')
    logger.info('这是一条普通信息')
    logger.warning('这是一条警告信息')
    logger.error('这是一条错误信息')

    print("\n✅ 日志模块测试完成")
