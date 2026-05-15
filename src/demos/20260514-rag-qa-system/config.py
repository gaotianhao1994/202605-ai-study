"""
配置管理模块

负责加载环境变量和提供配置信息
支持阿里云百炼 API 配置，兼容 OpenAI 接口

知识点：为什么需要配置管理？
- 将敏感信息（API Key）与代码分离 → 避免泄露风险
- 统一管理环境变量 → 一处修改，全局生效
- 支持不同环境切换 → 开发/测试/生产使用不同配置
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv
from pathlib import Path


class ConfigError(Exception):
    """配置错误异常"""
    pass


class Config:
    """
    配置管理类

    负责加载和管理环境变量配置，支持阿里云百炼 API

    为什么用类而不是直接读环境变量？
    - 封装验证逻辑 → 配置缺失时立即报错，而非运行时才发现
    - 提供类型安全 → 方法返回明确类型，避免 None 导致的隐蔽 bug
    - 单一职责 → 配置的加载、验证、获取集中管理
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        初始化配置

        Args:
            env_file: .env 文件路径，默认为当前目录下的 .env

        为什么支持自定义 env_file？
        - 测试时可以用不同的配置文件 → 隔离测试环境
        - 部署时可以指定不同路径 → 灵活适配各种部署方式
        """
        self.env_file = env_file
        self._api_key: Optional[str] = None
        self._api_base: Optional[str] = None
        self._model_name: Optional[str] = None
        self._embedding_model_name: Optional[str] = None

        self.load_env()

    def load_env(self) -> None:
        """
        加载环境变量文件

        为什么用 python-dotenv 而不是手动 export？
        - .env 文件可以和项目一起版本控制（.env.example） → 团队协作方便
        - 不污染系统环境变量 → 不同项目互不影响
        - 自动加载，无需每次手动设置 → 减少遗忘导致的错误

        Raises:
            FileNotFoundError: .env 文件不存在
        """
        if self.env_file:
            env_path = Path(self.env_file)
        else:
            env_path = Path(__file__).parent / '.env'

        if not env_path.exists():
            raise FileNotFoundError(
                f".env 文件不存在: {env_path}\n"
                "请复制 .env.example 为 .env 并填写配置"
            )

        load_dotenv(env_path)

        self._api_key = os.getenv('OPENAI_API_KEY')
        self._api_base = os.getenv('OPENAI_API_BASE')
        self._model_name = os.getenv('OPENAI_MODEL_NAME')
        self._embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-v3')

    def get_api_key(self) -> str:
        """
        获取 API Key

        Returns:
            API Key 字符串

        Raises:
            ConfigError: API Key 未配置
        """
        if not self._api_key:
            raise ConfigError(
                "OPENAI_API_KEY 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_KEY"
            )
        return self._api_key

    def get_api_base(self) -> str:
        """
        获取 API Base URL

        Returns:
            API Base URL 字符串

        Raises:
            ConfigError: API Base 未配置
        """
        if not self._api_base:
            raise ConfigError(
                "OPENAI_API_BASE 未配置\n"
                "请在 .env 文件中设置 OPENAI_API_BASE"
            )
        return self._api_base

    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            模型名称字符串

        Raises:
            ConfigError: 模型名称未配置
        """
        if not self._model_name:
            raise ConfigError(
                "OPENAI_MODEL_NAME 未配置\n"
                "请在 .env 文件中设置 OPENAI_MODEL_NAME"
            )
        return self._model_name

    def get_embedding_model_name(self) -> str:
        """
        获取嵌入模型名称

        Returns:
            嵌入模型名称字符串

        为什么嵌入模型和对话模型分开配置？
        - 嵌入模型和对话模型是不同的任务 → 需要不同的模型
        - 嵌入模型通常更小更快 → 成本更低
        - 对话模型需要推理能力 → 选择更大的模型
        """
        return self._embedding_model_name or 'text-embedding-v3'

    def validate(self) -> bool:
        """
        验证配置是否完整

        Returns:
            True 如果所有必需配置都存在

        Raises:
            ConfigError: 配置不完整

        为什么需要验证？
        - 尽早发现问题 → 配置缺失时立即报错，而非运行到一半才失败
        - 友好提示 → 告诉用户缺少什么配置，而非抛出晦涩的错误
        """
        try:
            self.get_api_key()
            self.get_api_base()
            self.get_model_name()
            return True
        except ConfigError as e:
            raise ConfigError(f"配置验证失败: {e}")

    def get_llm_config(self) -> Dict[str, str]:
        """
        获取 LLM 配置字典

        Returns:
            包含 model, openai_api_base, openai_api_key 的字典
        """
        return {
            'model': self.get_model_name(),
            'openai_api_base': self.get_api_base(),
            'openai_api_key': self.get_api_key()
        }

    def get_embedding_config(self) -> Dict[str, str]:
        """
        获取嵌入模型配置字典

        Returns:
            包含 model, openai_api_base, openai_api_key 的字典
        """
        return {
            'model': self.get_embedding_model_name(),
            'openai_api_base': self.get_api_base(),
            'openai_api_key': self.get_api_key()
        }

    def __repr__(self) -> str:
        """返回配置的字符串表示（隐藏敏感信息）"""
        api_key_preview = f"{self._api_key[:10]}..." if self._api_key else "未设置"
        return (
            f"Config(\n"
            f"  api_key='{api_key_preview}',\n"
            f"  api_base='{self._api_base}',\n"
            f"  model_name='{self._model_name}',\n"
            f"  embedding_model_name='{self._embedding_model_name}'\n"
            f")"
        )


def get_config(env_file: Optional[str] = None) -> Config:
    """
    获取配置实例（便捷函数）

    Args:
        env_file: .env 文件路径

    Returns:
        Config 实例
    """
    return Config(env_file)


if __name__ == '__main__':
    try:
        config = get_config()
        print("配置加载成功")
        print(config)
        config.validate()
        print("配置验证通过")
    except Exception as e:
        print(f"配置错误: {e}")
