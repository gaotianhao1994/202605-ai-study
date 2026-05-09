"""
第一章：环境准备与快速开始

学习目标：
- 了解 LangChain 是什么
- 检查环境配置
- 运行第一个 LangChain 程序

作者：AI Study Project
日期：2026-05-09
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, ConfigError
from logger import setup_logger

logger = setup_logger('chapter1_env_setup')


def check_python_version() -> bool:
    """
    检查 Python 版本是否 >= 3.8
    
    Returns:
        True 如果版本符合要求
    """
    logger.info("检查 Python 版本...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python 版本: {sys.version}")
    print(f"Python 版本号: {version_str}")
    print(f"Python 解释器路径: {sys.executable}")
    
    if version >= (3, 8):
        logger.info(f"✅ Python 版本符合要求: {version_str}")
        print("✅ Python 版本符合要求")
        return True
    else:
        logger.error(f"❌ Python 版本过低: {version_str}")
        print("❌ Python 版本过低，请升级到 3.8 或更高版本")
        return False


def check_dependencies() -> bool:
    """
    检查依赖是否安装
    
    Returns:
        True 如果所有依赖都已安装
    """
    logger.info("检查依赖安装...")
    
    dependencies = {
        'langchain': 'langchain',
        'langchain_openai': 'langchain-openai',
        'dotenv': 'python-dotenv'
    }
    
    all_installed = True
    
    for module_name, package_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            logger.debug(f"✅ {package_name} 已安装: {version}")
            print(f"✅ {package_name} 已安装: {version}")
        except ImportError:
            logger.error(f"❌ {package_name} 未安装")
            print(f"❌ {package_name} 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n请安装缺失的依赖:")
        print("pip install langchain==1.2.17 langchain-openai python-dotenv")
    
    return all_installed


def check_api_key() -> bool:
    """
    检查 API Key 配置
    
    Returns:
        True 如果 API Key 已配置
    """
    logger.info("检查 API Key 配置...")
    
    try:
        config = get_config()
        api_key = config.get_api_key()
        
        key_preview = f"{api_key[:10]}..."
        
        logger.info(f"✅ API Key 已配置: {key_preview}")
        print(f"✅ API Key 已配置")
        print(f"   API Key 前缀: {key_preview}")
        
        return True
        
    except (FileNotFoundError, ConfigError) as e:
        logger.error(f"❌ API Key 配置错误: {e}")
        print(f"❌ API Key 配置错误")
        print(f"\n请按照以下步骤配置:")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中填入你的 OPENAI_API_KEY")
        print("3. 重新运行此程序")
        
        return False


def run_first_program() -> bool:
    """
    运行第一个 LangChain 程序
    
    Returns:
        True 如果程序运行成功
    """
    logger.info("运行第一个 LangChain 程序...")
    
    print("\n" + "=" * 60)
    print("第一个 LangChain 程序")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI
        
        config = get_config()
        model_config = config.get_model_config()
        
        logger.debug(f"创建模型实例: {model_config['model']}")
        
        llm = ChatOpenAI(
            **model_config,
            temperature=0.7
        )
        
        prompt = "你好，请用一句话介绍 LangChain"
        
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
        print(f"\n✅ 程序运行成功！耗时 {elapsed_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 程序运行失败: {e}", exc_info=True)
        print(f"\n❌ 程序运行失败: {e}")
        print("\n可能的原因:")
        print("1. API Key 无效")
        print("2. 网络连接问题")
        print("3. API 服务不可用")
        
        return False


def main():
    """
    主函数：运行所有环境检查
    
    执行步骤：
    1. 检查 Python 版本
    2. 检查依赖安装
    3. 检查 API Key 配置
    4. 运行第一个程序
    """
    print("=" * 60)
    print("📚 第一章：环境准备与快速开始")
    print("=" * 60)
    
    logger.info("开始环境检查...")
    
    checks = [
        ("Python 版本", check_python_version),
        ("依赖安装", check_dependencies),
        ("API Key 配置", check_api_key),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{'=' * 60}")
        print(f"检查项: {name}")
        print("=" * 60)
        
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"检查 {name} 时发生错误: {e}", exc_info=True)
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("运行第一个 LangChain 程序")
        print("=" * 60)
        run_first_program()
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！环境配置完成，可以开始学习了！")
        print("=" * 60)
    else:
        print("\n❌ 部分检查未通过，请先解决上述问题")
    
    logger.info("环境检查完成")


if __name__ == '__main__':
    main()
