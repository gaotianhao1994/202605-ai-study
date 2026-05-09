#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试阿里云百炼 API（使用 OpenAI 客户端库）
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=" * 60)
print("阿里云百炼 API 测试（使用 OpenAI 客户端）")
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
    from openai import OpenAI
    
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    print("正在调用阿里云百炼 API...")
    
    # 调用 API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ],
        temperature=0.7
    )
    
    print("✅ API 调用成功！")
    print(f"\n【AI 回答】")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"❌ API 调用失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
