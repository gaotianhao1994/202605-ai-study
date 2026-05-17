#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5章：问答链构建与优化

学习目标：
1. 掌握LCEL（LangChain Expression Language）构建RAG问答链
2. 学会自定义Prompt模板控制回答风格
3. 理解带来源引用的问答实现
4. 理解端到端RAG流程
5. 实现交互式问答系统

核心问题：为什么需要问答链？
- 检索器只负责"找文档" -> 但用户需要的是"回答"
- 问答链将"检索+生成"串联 -> 自动完成：问题->检索->增强->生成
- LCEL是LangChain 1.x的链式构建方式 -> 用管道操作符组合组件

为什么langchain 1.x用LCEL替代了RetrievalQA？
- RetrievalQA是黑盒封装 -> 内部逻辑不透明，难以调试和定制
- LCEL是显式组合 -> 每个组件都可见，可以灵活替换和调试
- LCEL用管道操作符| -> 代码更简洁，数据流向更清晰
- LCEL天然支持流式输出、批处理、异步 -> RetrievalQA需要额外配置
- LCEL是函数式编程的思想 -> 组合优于继承，小组件拼装大功能

# ============================================================================
# 关于 LCEL vs RetrievalQA 的说明
# ============================================================================
#
# ⚠️ 已废弃：langchain.chains 模块
#    - RetrievalQA, LLMChain 等在 langchain 1.x 中已移除
#    - 不要使用：from langchain.chains import RetrievalQA
#
# ✅ 推荐方式：LCEL (LangChain Expression Language)
#    - 使用管道操作符 | 连接组件
#    - 示例：
#        chain = (
#            {"context": retriever | format_docs, "question": RunnablePassthrough()}
#            | prompt
#            | llm
#            | StrOutputParser()
#        )
#    - 优势：
#        1. 透明性：每个组件都可见，数据流向清晰
#        2. 灵活性：可以自由替换任何组件
#        3. 调试性：可以单独测试每个组件
#        4. 扩展性：添加组件只需用 | 连接
#        5. 现代特性：天然支持 stream(), batch(), ainvoke()
#
# ============================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger

logger = setup_logger('chapter5_qa_chain')


def _build_vectorstore():
    """
    构建或加载FAISS向量库（内部辅助函数）

    为什么需要这个辅助函数？
    - 第5章的每个demo都需要向量库 -> 避免在每个函数中重复构建逻辑
    - 优先加载已有索引 -> 节省API调用费用和时间
    - 只在索引不存在时才重新创建 -> 按需构建

    追问：为什么第5章复用辅助函数而不是直接导入第4章的？
    - 每章独立运行 -> 不依赖其他章节的模块
    - 辅助函数逻辑简单 -> 复制比引入依赖更清晰
    - 保持教程的完整性 -> 读者可以只看一章就理解全部逻辑

    Returns:
        FAISS向量库实例，失败时返回None
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from config import get_config

    config = get_config()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    if os.path.exists(index_path):
        print(f"发现已有FAISS索引，直接加载: {index_path}")
        try:
            db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"加载成功! 向量数量: {db.index.ntotal}, 维度: {db.index.d}")
            logger.info(f"加载已有FAISS索引, 向量数: {db.index.ntotal}")
            return db
        except Exception as e:
            print(f"加载已有索引失败: {e}，将重新创建")
            logger.warning(f"加载已有索引失败: {e}，将重新创建")

    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        logger.error(f"数据目录不存在: {data_dir}")
        return None

    print("未找到已有索引，开始构建向量库...")

    loader = DirectoryLoader(
        path=data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    documents = loader.load()
    print(f"加载文档数: {len(documents)}")

    chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=chinese_separators,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"分割后chunk数: {len(chunks)}")

    try:
        print("正在创建向量库（需要调用嵌入模型，请稍候）...")

        batch_size = 10
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_emb)

        import numpy as np
        embedding_array = np.array(all_embeddings, dtype=np.float32)

        db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embedding_array)),
            embedding=embeddings,
            metadatas=metadatas,
        )

        print(f"向量库创建成功! 向量数量: {db.index.ntotal}, 维度: {db.index.d}")

        db.save_local(index_path)
        print(f"向量库已保存到: {index_path}")

        logger.info(f"向量库创建并保存成功, 向量数: {db.index.ntotal}")
        return db
    except Exception as e:
        print(f"[错误] 创建向量库失败: {e}")
        logger.error(f"创建向量库失败: {e}", exc_info=True)
        return None


def _create_llm(temperature=0):
    """
    创建LLM实例（内部辅助函数）

    为什么需要这个辅助函数？
    - 多个demo都需要创建LLM -> 避免重复配置代码
    - 统一配置来源 -> 都从config获取，保证一致性
    - 支持自定义temperature -> 不同场景需要不同的创造性

    Args:
        temperature: 温度参数，控制生成的随机性
            - temperature=0: 确定性输出，适合事实性问答
            - temperature=0.7: 有一定随机性，适合创意性任务
            - temperature=1.0: 高随机性，适合头脑风暴

    追问：为什么temperature设为0？
    - RAG问答系统追求准确性 -> 不希望LLM"发挥创意"
    - temperature=0时，LLM总是选择概率最高的token -> 输出最确定
    - 事实性问答不需要创造性 -> 确定性输出更可靠
    - 方便调试和对比 -> 同样的问题总是得到同样的回答

    Returns:
        ChatOpenAI实例
    """
    from langchain_openai import ChatOpenAI
    from config import get_config

    config = get_config()

    return ChatOpenAI(
        model=config.get_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        temperature=temperature
    )


def _create_retriever(db, search_type="similarity", k=3):
    """
    创建检索器（内部辅助函数）

    为什么需要这个辅助函数？
    - 问答链需要检索器作为组件 -> 统一创建逻辑
    - 支持不同搜索类型 -> 根据场景选择合适的检索策略
    - 默认参数适合大多数场景 -> 简化调用

    Args:
        db: FAISS向量库实例
        search_type: 搜索类型，"similarity"或"mmr"
        k: 返回的文档数量

    Returns:
        检索器实例
    """
    if search_type == "mmr":
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def _format_docs(docs):
    """
    将检索到的Document列表格式化为纯文本字符串

    为什么需要这个函数？
    - LCEL链中，检索器返回的是List[Document] -> 但Prompt模板需要的是字符串
    - 这个函数充当"适配器" -> 将Document列表转换为Prompt可用的context字符串
    - 每个文档内容用换行分隔 -> LLM可以区分不同文档的内容

    追问：为什么不用JSON或其他格式？
    - 纯文本最自然 -> LLM对自然语言的阅读理解能力最强
    - JSON增加了格式噪音 -> LLM需要额外理解结构标记
    - 简单的分隔就足够 -> 文档间用换行分隔，LLM就能区分

    Args:
        docs: 检索器返回的Document列表

    Returns:
        拼接后的纯文本字符串
    """
    return "\n\n".join(doc.page_content for doc in docs)


def demo_basic_qa_chain():
    """
    基础LCEL RAG链：最简单的问答链构建

    知识点：LCEL（LangChain Expression Language）

    是什么？
    LCEL是LangChain 1.x引入的链式构建语法，用管道操作符|将组件串联起来：
    用户问题 -> 检索相关文档 -> 文档拼入Prompt -> LLM生成回答

    为什么langchain 1.x用LCEL替代了RetrievalQA？
    - RetrievalQA是黑盒封装 -> 内部逻辑不透明，难以调试和定制
    - LCEL是显式组合 -> 每个组件都可见，可以灵活替换和调试
    - LCEL用管道操作符| -> 代码更简洁，数据流向更清晰
    - LCEL天然支持流式输出、批处理、异步 -> RetrievalQA需要额外配置

    追问：LCEL管道操作符|的含义是什么？
    - | 是Python的__or__操作符 -> LangChain重载了它来实现组件串联
    - A | B 表示A的输出作为B的输入 -> 类似Unix管道: cat file | grep pattern
    - 数据从左到右流动 -> 代码的阅读顺序就是数据的处理顺序
    - 每个|连接的都是一个Runnable -> Runnable是LCEL的基本接口

    追问：为什么说LCEL比RetrievalQA更灵活？
    - RetrievalQA: qa = RetrievalQA.from_chain_type(llm, retriever) -> 黑盒
    - LCEL: chain = {"context": ..., "question": ...} | prompt | llm | parser -> 白盒
    - LCEL中每个组件都可以单独替换 -> 换Prompt、换LLM、换解析器，互不影响
    - LCEL中每个组件都可以单独调试 -> 检查中间结果，定位问题
    - LCEL可以自由组合 -> 不受预定义链类型的限制
    """
    print("=" * 60)
    print("1. 基础LCEL RAG链")
    print("=" * 60)

    print("""
LCEL RAG链的核心概念：

  问答链 = 检索器 + Prompt模板 + LLM + 输出解析器

  LCEL语法:
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

  工作流程:
    用户问题 -> RunnablePassthrough()原样传递问题
            -> retriever检索相关文档 -> format_docs转为字符串
            -> 构建{context, question}字典
            -> prompt模板填充变量
            -> llm生成回答
            -> StrOutputParser()提取纯文本

  管道操作符|的含义:
    A | B -> A的输出作为B的输入
    类比Unix管道: cat file | grep pattern | sort
    数据从左到右流动，代码的阅读顺序就是数据的处理顺序

  为什么比RetrievalQA好？
    RetrievalQA方式（已废弃）:
      qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
      -> 黑盒封装，内部逻辑不透明
      -> 难以调试，难以定制
      -> 不支持流式输出

    LCEL方式:
      chain = {"context": ..., "question": ...} | prompt | llm | parser
      -> 白盒组合，每个组件都可见
      -> 灵活替换和调试
      -> 天然支持流式输出、批处理、异步

  关键组件解释:
    RunnablePassthrough(): 原样传递输入 -> 用户的问题直接传给question变量
    retriever | format_docs: 检索文档并转为字符串 -> 传给context变量
    StrOutputParser(): 从LLM的AIMessage中提取纯文本 -> 得到干净的回答字符串
""")
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过基础问答链演示: 向量库构建失败")
        return

    llm = _create_llm(temperature=0)
    retriever = _create_retriever(db, search_type="similarity", k=3)

    print("\n--- 1.1 创建基础LCEL RAG链 ---")

    prompt = ChatPromptTemplate.from_template(
        "请根据以下参考资料回答问题。如果参考资料中没有相关信息，"
        "请回答\"根据现有资料无法回答该问题\"。\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n"
        "回答："
    )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n创建LCEL RAG链:")
    print(f"  组件1: retriever | _format_docs -> 检索文档并转为context字符串")
    print(f"  组件2: RunnablePassthrough() -> 原样传递用户问题")
    print(f"  组件3: ChatPromptTemplate -> 填充context和question变量")
    print(f"  组件4: ChatOpenAI -> 生成回答（temperature=0）")
    print(f"  组件5: StrOutputParser() -> 从AIMessage提取纯文本")

    print(f"\n  LCEL管道详解:")
    print(f"    输入: '什么是RAG？'（用户问题字符串）")
    print(f"    -> RunnablePassthrough()传递: question='什么是RAG？'")
    print(f"    -> retriever检索: 返回3个Document")
    print(f"    -> _format_docs转换: Document列表 -> context字符串")
    print(f"    -> 构建字典: {{'context': '...', 'question': '什么是RAG？'}}")
    print(f"    -> prompt填充: 将变量填入模板")
    print(f"    -> llm生成: AIMessage('RAG是...')")
    print(f"    -> StrOutputParser提取: 'RAG是...'（纯文本）")

    print("\n--- 1.2 使用LCEL链回答问题 ---")

    questions = [
        "什么是RAG？",
        "Python有哪些数据结构？",
        "LangChain的核心组件有哪些？",
    ]

    for question in questions:
        print(f"\n问题: {question}")
        try:
            answer = rag_chain.invoke(question)
            print(f"回答: {answer}")

            logger.info(f"基础LCEL链: 问题='{question}'")
        except Exception as e:
            print(f"  [错误] 问答失败: {e}")
            logger.error(f"基础LCEL链失败: {e}", exc_info=True)

    print("\n--- 1.3 LCEL vs RetrievalQA对比 ---")
    print("""
    LCEL相比RetrievalQA的优势：

    1. 透明性
       RetrievalQA: 黑盒 -> 不知道内部做了什么
       LCEL: 白盒 -> 每个组件都可见，数据流向清晰

    2. 灵活性
       RetrievalQA: 只能通过chain_type_kwargs传自定义Prompt
       LCEL: 可以自由替换任何组件 -> 换Prompt、换LLM、加后处理

    3. 调试性
       RetrievalQA: 出错时难以定位是检索问题还是生成问题
       LCEL: 可以单独测试每个组件 -> 逐步排查

    4. 扩展性
       RetrievalQA: 固定的4种chain_type -> 扩展需要继承重写
       LCEL: 自由组合 -> 添加组件只需用|连接

    5. 现代特性
       RetrievalQA: 不支持流式输出
       LCEL: 天然支持stream()、batch()、ainvoke()等

    为什么langchain 1.x移除了langchain.chains模块？
    - chains模块是旧架构的遗留 -> 与LCEL设计理念冲突
    - 维护两套API成本高 -> 统一到LCEL更高效
    - LCEL能实现chains的所有功能 -> 没有理由保留旧API
    - 社区反馈LCEL更易用 -> 顺应社区趋势
""")

    print("--- 基础LCEL RAG链 要点总结 ---")
    print("1. LCEL用管道操作符|将组件串联，代码即数据流")
    print("2. RunnablePassthrough()原样传递输入，retriever | format_docs检索并格式化")
    print("3. StrOutputParser()从LLM输出中提取纯文本")
    print("4. LCEL比RetrievalQA更透明、更灵活、更易调试")
    print("5. langchain 1.x移除了langchain.chains，统一使用LCEL")

    logger.info("基础LCEL RAG链演示完成")


def demo_custom_prompt():
    """
    自定义Prompt模板：控制回答风格和格式

    知识点：自定义Prompt模板

    是什么？
    通过自定义ChatPromptTemplate，控制LLM如何使用检索到的文档来生成回答，
    包括回答的语言风格、格式要求、约束条件等。

    为什么需要自定义Prompt？
    - 默认Prompt没有格式约束 -> 回答可能不符合业务需求
    - 默认Prompt没有"不知道"的处理 -> LLM可能编造答案（幻觉）
    - 自定义Prompt是控制LLM行为最直接的方式 -> Prompt工程是核心技能
    - 不同场景需要不同风格 -> 技术文档要精确，教学场景要通俗

    追问：为什么Prompt对RAG系统如此重要？
    - Prompt是LLM唯一的"指令输入" -> LLM的行为完全由Prompt决定
    - 好的Prompt能让LLM基于文档准确回答 -> 差的Prompt导致幻觉
    - Prompt中的{context}和{question}是变量 -> 运行时动态替换
    - Prompt工程是RAG系统调优的第一步 -> 成本最低，效果最直接

    追问：为什么要在Prompt中要求"只根据上下文回答"？
    - LLM有大量预训练知识 -> 可能混入训练数据中的信息
    - RAG的目标是基于特定文档回答 -> 不希望LLM"自由发挥"
    - 明确约束可以减少幻觉 -> LLM会尽量遵循Prompt的指示
    - 但约束不是100%有效 -> 需要结合其他手段（如温度、检索质量）
    """
    print("\n" + "=" * 60)
    print("2. 自定义Prompt模板")
    print("=" * 60)

    print("""
自定义Prompt模板的核心思想：

  默认Prompt（简单）:
    "请根据以下参考资料回答问题..."
    -> 问题: 没有格式约束，回答可能不符合需求
    -> 问题: 没有处理"文档中没有答案"的情况
    -> 问题: 没有风格控制，回答可能过于简略或冗长

  自定义Prompt（详细）:
    "请根据以下参考资料回答问题。如果资料中没有相关信息，
     请回答'根据现有资料无法回答'..."
    -> 优势: 明确的格式约束，回答更规范
    -> 优势: 处理了"不知道"的情况，减少幻觉
    -> 优势: 可以指定回答风格（简洁/详细/教师）

  Prompt模板中的变量：
    {{context}}: 检索到的文档内容（由retriever | format_docs自动填充）
    {{question}}: 用户的问题（由RunnablePassthrough()自动填充）
    -> 这两个变量名由ChatPromptTemplate.from_template()中的占位符决定
""")
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过自定义Prompt演示: 向量库构建失败")
        return

    llm = _create_llm(temperature=0)
    retriever = _create_retriever(db, search_type="similarity", k=3)

    print("\n--- 2.1 简单Prompt vs 详细Prompt ---")

    simple_prompt = ChatPromptTemplate.from_template(
        "根据以下资料回答问题：\n\n{context}\n\n问题：{question}"
    )

    detailed_prompt = ChatPromptTemplate.from_template(
        "请根据以下参考资料回答问题。要求：\n"
        "1. 只根据参考资料中的信息回答，不要编造内容\n"
        "2. 如果参考资料中没有相关信息，请回答\"根据现有资料无法回答该问题\"\n"
        "3. 回答要简洁明了，条理清晰\n"
        "4. 如果参考资料中有多个相关要点，请分条列出\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n回答："
    )

    simple_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | simple_prompt
        | llm
        | StrOutputParser()
    )

    detailed_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | detailed_prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n简单Prompt链: 创建成功")
    print(f"详细Prompt链: 创建成功")
    print(f"\n  详细Prompt特点:")
    print(f"    - 明确要求'不要编造' -> 减少幻觉")
    print(f"    - 处理'无法回答'的情况 -> 避免强行回答")
    print(f"    - 要求'分条列出' -> 回答更清晰")
    print(f"    - LCEL中只需替换prompt组件 -> 其他组件不变")

    print("\n--- 2.2 对比简单和详细Prompt的回答 ---")

    test_questions = [
        "什么是RAG？",
        "今天天气怎么样？",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")

        try:
            simple_answer = simple_chain.invoke(question)
            print(f"\n  [简单Prompt] 回答:")
            print(f"    {simple_answer}")
        except Exception as e:
            print(f"\n  [简单Prompt] 错误: {e}")

        try:
            detailed_answer = detailed_chain.invoke(question)
            print(f"\n  [详细Prompt] 回答:")
            print(f"    {detailed_answer}")
        except Exception as e:
            print(f"\n  [详细Prompt] 错误: {e}")

        logger.info(f"Prompt对比: 问题='{question}'")

    print(f"\n  分析:")
    print(f"    - 对于知识库内的问题，两种Prompt都能回答")
    print(f"    - 详细Prompt的回答更规范、更符合中文习惯")
    print(f"    - 对于知识库外的问题，详细Prompt会明确说'无法回答'")
    print(f"    - 简单Prompt可能尝试回答 -> 产生幻觉的风险更高")

    print("\n--- 2.3 不同风格的自定义Prompt ---")

    concise_template = (
        "根据以下资料简要回答问题，不超过3句话。\n\n"
        "资料：{context}\n\n问题：{question}\n\n简要回答："
    )

    detailed_style_template = (
        "你是一位专业的AI助手。请根据以下参考资料，详细回答用户的问题。\n"
        "回答要求：\n"
        "1. 先给出直接回答\n"
        "2. 再提供详细解释\n"
        "3. 如有可能，给出示例说明\n"
        "4. 最后总结要点\n\n"
        "参考资料：\n{context}\n\n问题：{question}\n\n详细回答："
    )

    teacher_template = (
        "你是一位耐心的老师，正在给学生讲解知识。请根据以下资料回答问题。\n"
        "回答要求：\n"
        "1. 用通俗易懂的语言解释\n"
        "2. 适当使用类比帮助理解\n"
        "3. 如果涉及专业术语，先解释术语含义\n"
        "4. 鼓励学生继续思考\n\n"
        "参考资料：\n{context}\n\n问题：{question}\n\n老师回答："
    )

    prompt_styles = [
        ("简洁风格", concise_template),
        ("详细风格", detailed_style_template),
        ("教师风格", teacher_template),
    ]

    question = "什么是RAG？"
    print(f"\n用不同风格的Prompt回答同一问题: '{question}'")

    for style_name, template in prompt_styles:
        try:
            prompt = ChatPromptTemplate.from_template(template)

            chain = (
                {"context": retriever | _format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)

            print(f"\n  [{style_name}]:")
            print(f"    {answer}")

            logger.info(f"Prompt风格对比: {style_name}, 问题='{question}'")
        except Exception as e:
            print(f"\n  [{style_name}] 错误: {e}")
            logger.error(f"Prompt风格对比失败: {style_name}, 错误: {e}")

    print(f"\n  分析:")
    print(f"    - 同样的检索结果，不同的Prompt产生不同风格的回答")
    print(f"    - 简洁风格: 回答短小精悍，适合快速获取信息")
    print(f"    - 详细风格: 回答全面深入，适合学习理解")
    print(f"    - 教师风格: 回答通俗易懂，适合教学场景")
    print(f"    - Prompt是控制LLM行为最直接的方式 -> 同样的数据，不同的表达")
    print(f"    - LCEL中切换风格只需替换prompt组件 -> 其他组件完全复用")

    print("\n--- 2.4 Prompt工程的最佳实践 ---")
    print("""
    编写高质量RAG Prompt的原则：

    1. 明确角色和任务
       "你是一位专业的AI助手，请根据参考资料回答问题"
       -> 告诉LLM"你是谁"和"要做什么"

    2. 约束信息来源
       "只根据参考资料回答，不要使用外部知识"
       -> 减少幻觉，保证回答基于检索结果

    3. 处理"不知道"的情况
       "如果资料中没有相关信息，请回答'无法回答'"
       -> 避免LLM强行编造答案

    4. 指定输出格式
       "请分条列出要点" / "请用表格形式展示"
       -> 让回答更规范、更易读

    5. 提供思考框架
       "先分析问题，再给出答案，最后总结"
       -> 引导LLM的结构化思考

    6. 迭代优化
       - 先写基础Prompt -> 测试效果
       - 发现问题 -> 修改Prompt -> 再测试
       - Prompt工程是实验性的 -> 需要反复迭代
""")

    print("--- 自定义Prompt 要点总结 ---")
    print("1. 自定义Prompt是控制RAG回答风格和质量的核心手段")
    print("2. ChatPromptTemplate.from_template()创建模板，{context}和{question}是变量")
    print("3. LCEL中替换prompt组件即可切换风格，其他组件不变")
    print("4. 好的Prompt要: 明确角色、约束来源、处理未知、指定格式")
    print("5. 同样的检索结果，不同的Prompt产生不同风格的回答")

    logger.info("自定义Prompt演示完成")


def demo_qa_with_sources():
    """
    带来源引用的问答：使用RunnableParallel同时获取回答和源文档

    知识点：来源引用 + RunnableParallel

    是什么？
    在RAG回答中标注信息的来源文档，让用户知道回答依据的是哪些文档，
    提高回答的可信度和可验证性。使用RunnableParallel同时获取回答和源文档。

    为什么需要来源引用？
    - LLM可能产生幻觉 -> 来源引用让用户可以验证回答的准确性
    - 提高可信度 -> "根据XX文档，答案是..."比"答案是..."更可信
    - 便于溯源 -> 用户可以查看原文获取更详细的信息
    - 合规要求 -> 某些场景（医疗、法律）必须标注信息来源

    追问：为什么来源引用对RAG系统特别重要？
    - RAG的核心价值是"基于文档回答" -> 来源引用是这个价值的证明
    - 没有来源引用，RAG的回答和LLM直接回答没有区别 -> 用户无法验证
    - 有了来源引用，用户可以判断回答是否可靠 -> 增强信任
    - 企业场景中，来源引用是合规的必要条件 -> 不可省略

    追问：为什么用RunnableParallel而不是分别调用？
    - 分别调用: 先调chain得回答，再调retriever得文档 -> 检索执行两次
    - RunnableParallel: 一次执行，同时获取回答和源文档 -> 检索只执行一次
    - 性能更好 -> 避免重复的API调用和检索计算
    - 数据一致 -> 回答和来源一定是对应的，不会出现不一致
    """
    print("\n" + "=" * 60)
    print("3. 带来源引用的问答")
    print("=" * 60)

    print("""
来源引用的核心价值：

  没有来源引用:
    用户: "什么是RAG？"
    系统: "RAG是检索增强生成技术..."
    用户心理: "这个回答可靠吗？根据什么说的？"

  有来源引用:
    用户: "什么是RAG？"
    系统: "RAG是检索增强生成技术... [来源: ai_basics.txt]"
    用户心理: "有据可查，可以信任"

  RunnableParallel的实现方式:
    rag_chain_with_sources = RunnableParallel(
        answer=(... | prompt | llm | StrOutputParser()),
        source_docs=retriever,
    )
    -> 一次调用，同时获取answer和source_docs
    -> 避免重复检索，性能更好
    -> 数据一致，回答和来源一定对应

  为什么用RunnableParallel？
    - 普通LCEL链只返回回答文本 -> 无法获取源文档
    - RunnableParallel让链同时输出多个结果 -> 回答+源文档
    - 检索只执行一次 -> 性能优于分别调用
""")
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过来源引用演示: 向量库构建失败")
        return

    llm = _create_llm(temperature=0)
    retriever = _create_retriever(db, search_type="similarity", k=3)

    print("\n--- 3.1 使用RunnableParallel获取回答和来源 ---")

    prompt = ChatPromptTemplate.from_template(
        "请根据以下参考资料回答问题，并在回答中标注信息来源。\n\n"
        "要求：\n"
        "1. 回答时在相关内容后标注来源，格式为[来源: 文件名]\n"
        "2. 如果多个来源都提到了相关信息，请都标注\n"
        "3. 只根据参考资料回答，不要编造内容\n"
        "4. 如果参考资料中没有相关信息，请回答\"根据现有资料无法回答\"\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n回答（请标注来源）："
    )

    answer_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_sources = RunnableParallel(
        answer=answer_chain,
        source_docs=retriever,
    )

    print(f"\n创建带来源的RAG链:")
    print(f"  RunnableParallel包含两个并行分支:")
    print(f"    answer: retriever | format_docs | prompt | llm | StrOutputParser")
    print(f"    source_docs: retriever（直接返回Document列表）")
    print(f"  调用方式: result = chain.invoke('问题')")
    print(f"  获取回答: result['answer'] -> 纯文本字符串")
    print(f"  获取来源: result['source_docs'] -> Document列表")

    question = "什么是RAG？"
    print(f"\n问题: {question}")

    try:
        result = rag_chain_with_sources.invoke(question)

        answer = result["answer"]
        source_docs = result["source_docs"]

        print(f"\n回答: {answer}")
        print(f"\n来源文档详情 ({len(source_docs)} 个):")

        for i, doc in enumerate(source_docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"  {i}. 文件: {source}")
            print(f"     内容预览: {content_preview}...")

        logger.info(f"来源引用演示: 问题='{question}', 来源数={len(source_docs)}")
    except Exception as e:
        print(f"  [错误] 问答失败: {e}")
        logger.error(f"来源引用问答失败: {e}", exc_info=True)

    print("\n--- 3.2 多个问题的来源引用 ---")

    questions = [
        "什么是RAG？",
        "Python有哪些数据结构？",
        "LangChain的核心组件有哪些？",
    ]

    for question in questions:
        print(f"\n问题: {question}")
        try:
            result = rag_chain_with_sources.invoke(question)

            answer = result["answer"]
            source_docs = result["source_docs"]

            sources = set()
            for doc in source_docs:
                source = os.path.basename(doc.metadata.get("source", "未知"))
                sources.add(source)

            print(f"回答: {answer}")
            print(f"参考来源文件: {', '.join(sources)}")

            logger.info(f"Prompt来源引用: 问题='{question}', 来源={sources}")
        except Exception as e:
            print(f"  [错误] 问答失败: {e}")
            logger.error(f"Prompt来源引用失败: {e}", exc_info=True)

    print("\n--- 3.3 构建格式化的来源引用输出 ---")

    def format_qa_with_sources(question, chain):
        """
        格式化带来源引用的问答输出

        为什么需要格式化函数？
        - RunnableParallel返回的是字典结构 -> 不适合直接展示给用户
        - 格式化后更清晰 -> 用户一眼就能看到回答和来源
        - 统一输出格式 -> 便于前端展示或日志记录

        Args:
            question: 用户问题
            chain: 带来源的RAG链（RunnableParallel）

        Returns:
            格式化的问答结果字符串
        """
        try:
            result = chain.invoke(question)

            answer = result["answer"]
            source_docs = result["source_docs"]

            source_info = []
            for doc in source_docs:
                source = os.path.basename(doc.metadata.get("source", "未知"))
                preview = doc.page_content[:50].replace("\n", " ")
                source_info.append({"file": source, "preview": preview})

            formatted = f"问题: {question}\n"
            formatted += f"回答: {answer}\n"
            formatted += f"来源:\n"
            for i, info in enumerate(source_info, 1):
                formatted += f"  {i}. [{info['file']}] {info['preview']}...\n"

            return formatted
        except Exception as e:
            return f"问题: {question}\n错误: {e}"

    question = "LangChain的核心组件有哪些？"
    print(f"\n格式化输出示例:")
    formatted_output = format_qa_with_sources(question, rag_chain_with_sources)
    print(formatted_output)

    print("\n--- 3.4 来源引用的实践建议 ---")
    print("""
    来源引用的最佳实践：

    1. 始终使用RunnableParallel获取源文档
       -> 即使不在回答中展示，也用于内部质量监控
       -> 可以统计来源命中率，评估检索质量

    2. 在Prompt中要求引用来源
       -> 让LLM在回答中自然地标注来源
       -> 比单独列出来源更自然、更易读

    3. 展示来源时包含文件名和内容预览
       -> 文件名让用户知道信息来自哪个文档
       -> 内容预览让用户快速判断来源是否相关

    4. 处理多来源的情况
       -> 同一个问题可能涉及多个文档
       -> 标注所有相关来源，不要遗漏

    5. 区分"有来源"和"无来源"的回答
       -> 有来源: 可信度高
       -> 无来源: 可能是LLM编造的，需要特别标注
""")

    print("--- 来源引用 要点总结 ---")
    print("1. 来源引用提高RAG回答的可信度和可验证性")
    print("2. RunnableParallel同时获取回答和源文档，避免重复检索")
    print("3. result['answer']获取回答，result['source_docs']获取源文档")
    print("4. 在Prompt中要求引用来源，让LLM自动标注")
    print("5. 来源引用是企业级RAG系统的必要功能")

    logger.info("来源引用演示完成")


def demo_end_to_end_rag():
    """
    端到端RAG流程演示：展示完整的7步RAG工作流

    知识点：端到端RAG流程

    是什么？
    从用户提问到获得回答的完整流程，包括：
    文档加载 -> 文本分割 -> 向量嵌入 -> 向量存储 -> 检索 -> Prompt增强 -> LLM生成

    为什么需要理解端到端流程？
    - 前面4章分别学习了各个组件 -> 但RAG是组件协作的系统
    - 理解端到端流程才能定位问题 -> 每个环节都可能出错
    - 优化需要全局视角 -> 只优化一个环节可能不够
    - 端到端流程是RAG系统的"全景图" -> 帮助建立整体认知

    追问：RAG流程中哪个环节最容易成为瓶颈？
    - 检索质量 -> "垃圾进，垃圾出"，检索不到相关文档，LLM无法正确回答
    - 文本分割 -> chunk太大语义模糊，chunk太小信息不足
    - Prompt设计 -> 好的Prompt引导LLM正确使用上下文
    - 通常检索质量是最大瓶颈 -> 优化检索的收益最大

    追问：为什么说RAG是"检索决定上限，生成决定下限"？
    - 检索质量决定了LLM能获取多少相关信息 -> 信息不足，LLM再强也无法正确回答
    - 生成质量决定了LLM如何利用检索到的信息 -> 信息充足但Prompt差，回答也不理想
    - 检索是上限: 检索不到的信息，LLM不可能用到
    - 生成是下限: 检索到了好信息，但LLM可能用不好
    - 因此优化顺序: 先优化检索，再优化生成
    """
    print("\n" + "=" * 60)
    print("4. 端到端RAG流程演示")
    print("=" * 60)

    print("""
端到端RAG流程全景图：

  用户提问
    |
    v
  [1. 文档加载] -- 加载知识库文档（第1章）
    |
    v
  [2. 文本分割] -- 将长文档切分为合适大小的chunk（第2章）
    |
    v
  [3. 向量嵌入] -- 将文本转换为语义向量（第3章）
    |
    v
  [4. 向量存储] -- 构建FAISS索引（第3章）
    |
    v
  [5. 检索] -- 根据问题检索最相关的文档（第4章）
    |
    v
  [6. Prompt增强] -- 将检索结果拼入Prompt（第5章）
    |
    v
  [7. LLM生成] -- 基于增强Prompt生成回答（第5章）
    |
    v
  用户获得回答

  关键洞察：
    - 每一步都影响最终回答质量 -> 任何一步出问题，结果都不好
    - 检索质量是最大瓶颈 -> "垃圾进，垃圾出"
    - Prompt是连接检索和生成的桥梁 -> 好的Prompt让LLM充分利用检索结果
""")

    print("\n--- 4.1 完整RAG流程：从文档到回答 ---")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from config import get_config

    config = get_config()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    print("\n步骤1: 文档加载")
    if not os.path.exists(data_dir):
        print(f"  [错误] 数据目录不存在: {data_dir}")
        logger.warning("端到端演示跳过: 数据目录不存在")
        return

    loader = DirectoryLoader(
        path=data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    documents = loader.load()
    print(f"  加载文档数: {len(documents)}")
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"    - {source}: {len(doc.page_content)} 字符")

    print("\n步骤2: 文本分割")
    chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=chinese_separators,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  分割后chunk数: {len(chunks)}")
    print(f"  平均长度: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")

    print("\n步骤3&4: 向量嵌入与存储")
    embeddings = OpenAIEmbeddings(
        model=config.get_embedding_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        check_embedding_ctx_length=False
    )

    if os.path.exists(index_path):
        print(f"  加载已有索引: {index_path}")
        try:
            db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"  加载成功! 向量数: {db.index.ntotal}")
        except Exception as e:
            print(f"  加载失败: {e}，重新创建")

            batch_size = 10
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            all_embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_emb = embeddings.embed_documents(batch_texts)
                all_embeddings_list.extend(batch_emb)

            import numpy as np
            embedding_array = np.array(all_embeddings_list, dtype=np.float32)
            db = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embedding_array)),
                embedding=embeddings,
                metadatas=metadatas,
            )
            db.save_local(index_path)
            print(f"  创建成功! 向量数: {db.index.ntotal}")
    else:
        print("  创建新索引...")

        batch_size = 10
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        all_embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = embeddings.embed_documents(batch_texts)
            all_embeddings_list.extend(batch_emb)

        import numpy as np
        embedding_array = np.array(all_embeddings_list, dtype=np.float32)
        db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embedding_array)),
            embedding=embeddings,
            metadatas=metadatas,
        )
        db.save_local(index_path)
        print(f"  创建成功! 向量数: {db.index.ntotal}")

    print("\n步骤5: 创建检索器")
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print(f"  检索策略: similarity, k=3")

    print("\n步骤6: 设计Prompt模板")
    rag_prompt = ChatPromptTemplate.from_template(
        "你是一个专业的知识库问答助手。请根据以下参考资料回答用户的问题。\n\n"
        "回答要求：\n"
        "1. 只根据参考资料中的信息回答，不要使用外部知识\n"
        "2. 如果参考资料中没有相关信息，请明确说明\"根据现有资料无法回答\"\n"
        "3. 回答要准确、完整、有条理\n"
        "4. 适当引用来源信息\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n回答："
    )
    print(f"  Prompt模板: 已创建（中文，含来源引用要求）")

    print("\n步骤7: 创建LLM和LCEL RAG链")
    llm = ChatOpenAI(
        model=config.get_model_name(),
        openai_api_base=config.get_api_base(),
        openai_api_key=config.get_api_key(),
        temperature=0
    )
    print(f"  LLM: {config.get_model_name()}, temperature=0")

    answer_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    qa_chain = RunnableParallel(
        answer=answer_chain,
        source_docs=retriever,
    )

    print(f"  LCEL RAG链: 创建成功（使用RunnableParallel获取回答+来源）")

    print("\n--- 4.2 端到端问答测试 ---")

    test_cases = [
        {
            "question": "什么是RAG？",
            "description": "知识库内的问题（应能准确回答）",
        },
        {
            "question": "Python有哪些数据结构？",
            "description": "知识库内的问题（应能准确回答）",
        },
        {
            "question": "LangChain的核心组件有哪些？",
            "description": "知识库内的问题（应能准确回答）",
        },
        {
            "question": "什么是量子计算？",
            "description": "知识库外的问题（应回答无法回答）",
        },
    ]

    for tc in test_cases:
        question = tc["question"]
        description = tc["description"]
        print(f"\n  问题: {question}")
        print(f"  类型: {description}")

        try:
            result = qa_chain.invoke(question)

            answer = result["answer"]
            source_docs = result["source_docs"]

            sources = set()
            for doc in source_docs:
                source = os.path.basename(doc.metadata.get("source", "未知"))
                sources.add(source)

            print(f"  回答: {answer}")
            if sources:
                print(f"  来源: {', '.join(sources)}")
            else:
                print(f"  来源: 无（知识库外问题）")

            logger.info(f"端到端测试: 问题='{question}', 来源={sources}")
        except Exception as e:
            print(f"  [错误] {e}")
            logger.error(f"端到端测试失败: 问题='{question}', 错误: {e}", exc_info=True)

    print("\n--- 4.3 RAG系统优化方向 ---")
    print("""
    端到端RAG系统的优化方向（按优先级排序）：

    1. 检索质量优化（影响最大）
       - 调整chunk_size和chunk_overlap -> 找到最佳分割粒度
       - 尝试不同的检索策略 -> similarity vs MMR
       - 优化k值和fetch_k -> 平衡信息量和噪音
       - 添加元数据过滤 -> 精准定位相关文档
       - 使用混合检索 -> 向量搜索 + 关键词搜索

    2. Prompt工程优化（成本最低）
       - 优化Prompt模板 -> 更清晰的指令
       - 添加Few-shot示例 -> 引导LLM的输出格式
       - 要求引用来源 -> 提高可信度
       - 处理"不知道"的情况 -> 减少幻觉

    3. 文档预处理优化（基础保障）
       - 清洗文档 -> 去除噪音、格式化
       - 优化分割策略 -> 语义完整的chunk
       - 添加元数据 -> 支持过滤和溯源
       - 定期更新知识库 -> 保持信息时效性

    4. 生成质量优化（锦上添花）
       - 调整temperature -> 事实性问答用0
       - 选择更强大的LLM -> 更好的推理能力
       - 添加后处理 -> 格式化、去重、校验

    优化原则：
    - 先诊断瓶颈 -> 用评估指标定位问题环节
    - 一次只改一个变量 -> 确认每个改动的效果
    - 量化评估效果 -> 不凭感觉，用数据说话
    - 持续迭代优化 -> RAG系统没有"完成"的一天
""")

    print("--- 端到端RAG 要点总结 ---")
    print("1. RAG流程: 加载->分割->嵌入->存储->检索->增强->生成")
    print("2. 每个环节都影响最终质量，检索质量是最大瓶颈")
    print("3. 优化顺序: 检索质量 > Prompt工程 > 文档预处理 > 生成质量")
    print("4. 端到端测试验证整个流程的正确性")
    print("5. 优化需要量化评估，不凭感觉，用数据说话")

    logger.info("端到端RAG演示完成")


def demo_interactive_qa():
    """
    交互式问答：用户可以输入问题，系统实时回答

    知识点：交互式RAG系统

    是什么？
    一个命令行交互式问答系统，用户输入问题，系统实时检索并生成回答，
    支持多轮问答，直到用户主动退出。

    为什么需要交互式问答？
    - 前面的demo都是预设问题 -> 无法体验真实使用场景
    - 交互式问答模拟真实产品 -> 用户自由提问
    - 便于探索和调试 -> 可以测试各种问题，观察系统表现
    - 是RAG系统的最终形态 -> 从学习到实践的桥梁

    追问：交互式问答和生产环境有什么差距？
    - 生产环境需要Web界面 -> 命令行只适合开发调试
    - 生产环境需要并发处理 -> 命令行是单用户单线程
    - 生产环境需要会话管理 -> 支持多轮对话上下文
    - 生产环境需要监控和日志 -> 追踪每个问答的质量
    - 但核心逻辑相同 -> 交互式问答是生产系统的简化版

    追问：为什么交互式问答不支持多轮对话上下文？
    - 本教程的LCEL链是无状态的 -> 每次问答独立
    - 多轮对话需要维护会话历史 -> 增加复杂度
    - 多轮对话需要处理指代消解 -> "它"指代什么？
    - LangChain提供了RunnableWithMessageHistory等组件 -> 支持多轮对话
    - 但多轮对话的RAG是更高级的话题 -> 超出本章范围
    """
    print("\n" + "=" * 60)
    print("5. 交互式问答系统")
    print("=" * 60)

    print("""
交互式问答系统：

  工作模式:
    用户输入问题 -> 系统检索+生成 -> 显示回答和来源 -> 等待下一个问题

  使用方式:
    输入问题进行提问
    输入 'quit' 或 'exit' 退出
    输入 'help' 查看帮助

  注意:
    - 每次问答独立，不保留上下文
    - 回答基于知识库文档，不使用LLM的预训练知识
    - 如果知识库中没有相关信息，系统会说明
""")
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    db = _build_vectorstore()
    if db is None:
        print("[跳过] 向量库构建失败")
        logger.warning("跳过交互式问答演示: 向量库构建失败")
        return

    llm = _create_llm(temperature=0)
    retriever = _create_retriever(db, search_type="similarity", k=3)

    interactive_prompt = ChatPromptTemplate.from_template(
        "你是一个专业的知识库问答助手。请根据以下参考资料回答用户的问题。\n\n"
        "回答要求：\n"
        "1. 只根据参考资料中的信息回答，不要使用外部知识或编造内容\n"
        "2. 如果参考资料中没有相关信息，请回答\"根据现有资料无法回答该问题\"\n"
        "3. 回答要准确、完整、有条理\n"
        "4. 适当引用来源信息，格式为[来源: 文件名]\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n回答："
    )

    answer_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | interactive_prompt
        | llm
        | StrOutputParser()
    )

    qa_chain = RunnableParallel(
        answer=answer_chain,
        source_docs=retriever,
    )

    print("\n--- 5.1 启动交互式问答 ---")
    print("交互式问答系统已启动!")
    print("输入问题进行提问，输入 'quit' 或 'exit' 退出，输入 'help' 查看帮助")
    print("-" * 60)

    sample_questions = [
        "什么是RAG？",
        "Python有哪些数据结构？",
        "LangChain的核心组件有哪些？",
    ]

    print("\n示例问题（你可以直接输入，或参考以下问题）:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")

    print()

    question_count = 0

    while True:
        try:
            user_input = input("你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互式问答")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("退出交互式问答，再见!")
            break

        if user_input.lower() == 'help':
            print("\n帮助信息:")
            print("  - 输入问题进行提问")
            print("  - 输入 'quit' 或 'exit' 退出")
            print("  - 输入 'help' 查看帮助")
            print("  - 示例问题:")
            for q in sample_questions:
                print(f"      {q}")
            print()
            continue

        question_count += 1
        print(f"\n[问题 #{question_count}] {user_input}")

        try:
            result = qa_chain.invoke(user_input)

            answer = result["answer"]
            source_docs = result["source_docs"]

            print(f"\n[回答]")
            print(f"  {answer}")

            if source_docs:
                sources = set()
                for doc in source_docs:
                    source = os.path.basename(doc.metadata.get("source", "未知"))
                    sources.add(source)
                print(f"\n[来源] {', '.join(sources)}")
            else:
                print(f"\n[来源] 无相关文档")

            print("-" * 60)

            logger.info(f"交互式问答 #{question_count}: 问题='{user_input}', 来源数={len(source_docs)}")
        except Exception as e:
            print(f"\n[错误] 问答失败: {e}")
            print("-" * 60)
            logger.error(f"交互式问答失败: 问题='{user_input}', 错误: {e}", exc_info=True)

    print(f"\n本次会话共回答了 {question_count} 个问题")
    logger.info(f"交互式问答结束, 共回答 {question_count} 个问题")


def main():
    """
    主函数：依次运行所有演示

    执行顺序的设计逻辑：
    1. 基础LCEL RAG链 -> 理解LCEL语法和RAG链的构建方式
    2. 自定义Prompt模板 -> 学会控制回答风格和质量
    3. 带来源引用的问答 -> 提高回答的可信度和可验证性
    4. 端到端RAG流程 -> 理解完整的RAG工作流
    5. 交互式问答 -> 体验真实的RAG问答系统

    为什么按这个顺序？
    - 从基础到应用 -> 先理解LCEL的基本概念，再学习高级用法
    - 从简单到复杂 -> 先用默认配置，再自定义Prompt和来源引用
    - 从组件到系统 -> 先学习单个组件，再理解端到端流程
    - 从学习到实践 -> 最后通过交互式问答将所学付诸实践
    - 每一步都为下一步打基础 -> 逻辑递进
    """
    print("=" * 60)
    print("第5章：问答链构建与优化")
    print("=" * 60)

    logger.info("开始第5章演示")

    demo_basic_qa_chain()

    demo_custom_prompt()

    demo_qa_with_sources()

    demo_end_to_end_rag()

    demo_interactive_qa()

    print("\n" + "=" * 60)
    print("第5章学习总结")
    print("=" * 60)
    print("""
核心要点回顾：

1. 什么是LCEL RAG链？
   - LCEL用管道操作符|将检索、Prompt、LLM、解析器串联
   - 工作流程: 问题 -> 检索 -> Prompt增强 -> LLM生成 -> 解析输出
   - RunnablePassthrough()原样传递输入，StrOutputParser()提取纯文本
   - LCEL替代了已废弃的RetrievalQA，更透明、更灵活

2. 自定义Prompt模板
   - Prompt是控制LLM行为最直接的方式
   - ChatPromptTemplate.from_template()创建模板，{context}和{question}是变量
   - 好的Prompt要: 明确角色、约束来源、处理未知、指定格式
   - 同样的检索结果，不同的Prompt产生不同风格的回答

3. 来源引用
   - RunnableParallel同时获取回答和源文档，避免重复检索
   - result['answer']获取回答，result['source_docs']获取源文档
   - 在Prompt中要求引用来源，让LLM自动标注
   - 来源引用提高回答的可信度和可验证性

4. 端到端RAG流程
   - 完整流程: 加载->分割->嵌入->存储->检索->增强->生成
   - 检索质量是最大瓶颈 -> "垃圾进，垃圾出"
   - 优化顺序: 检索质量 > Prompt工程 > 文档预处理 > 生成质量
   - 优化需要量化评估，用数据说话

5. 交互式问答系统
   - 模拟真实RAG产品的使用体验
   - 支持自由提问和实时回答
   - 是从学习到实践的桥梁
   - 生产环境需要Web界面、并发处理、会话管理等

RAG教程总结（第1-5章）：

  第1章: 文档加载 -> 获取原始文本数据
  第2章: 文本分割 -> 切分为合适大小的chunk
  第3章: 向量嵌入与FAISS存储 -> 将文本转化为可检索的向量
  第4章: 检索器与相似度搜索 -> 找到最相关的文档
  第5章: 问答链构建与优化 -> 将检索和生成串联成完整系统

  RAG的本质:
    让LLM基于特定文档回答问题，而不是依赖预训练知识
    -> 解决LLM的知识时效性、私有数据、幻觉三大问题

  RAG的核心流程:
    文档 -> 分割 -> 嵌入 -> 存储 -> 检索 -> 增强 -> 生成
    每一步都影响最终质量，检索质量是最大瓶颈

  继续学习方向:
    - 多轮对话RAG: 支持上下文连续问答
    - Agent RAG: 让LLM自主决定何时检索、检索什么
    - 混合检索: 向量搜索 + 关键词搜索
    - RAG评估: 使用RAGAS等框架量化评估
    - RAG优化: Query改写、重排序、自适应检索
""")

    logger.info("第5章演示完成")


if __name__ == '__main__':
    main()
