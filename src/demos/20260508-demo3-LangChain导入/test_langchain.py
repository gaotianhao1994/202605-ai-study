#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangChain 安装测试脚本
"""

try:
    # 新版本 LangChain 的正确导入方式
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    
    print("✅ LangChain 核心模块导入成功")
    
    # 检查版本
    import langchain
    print(f"✅ LangChain 版本: {langchain.__version__}")
    
    print("\n🎉 LangChain 安装成功！可以开始开发了！")
    print("\n接下来你可以：")
    print("1. 创建 .env 文件，添加 OPENAI_API_KEY")
    print("2. 编写你的第一个 LangChain 应用")
    print("3. 运行 demo.py 查看示例")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n请确保已安装 LangChain:")
    print("pip install langchain==1.2.17 langchain-openai python-dotenv")