#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试阿里云百炼 API 是否正常
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=" * 60)
print("阿里云百炼 API 测试")
print("=" * 60)

# 显示配置信息
print("\n【配置信息】")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

print(f"API Key: {api_key[:10]}...")
print(f"API Base: {api_base}")
print(f"Model: {model_name}")

# 测试 API 调用
print("\n【测试 API 调用】")
try:
    from langchain_openai import ChatOpenAI
    
    # 创建模型实例，使用阿里云百炼的配置
    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=api_base,
        openai_api_key=api_key,
        temperature=0.7,
        request_timeout=30
    )
    
    print("正在调用阿里云百炼 API...")
    response = llm.invoke("你好，请用一句话介绍你自己")
    
    print("✅ API 调用成功！")
    print(f"\n【AI 回答】")
    print(response.content)
    
except Exception as e:
    print(f"❌ API 调用失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
