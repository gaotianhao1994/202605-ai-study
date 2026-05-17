"""
LangGraph 完全指南：从零基础到精通

学习目标：
- 理解 LangGraph 的核心概念（State, Node, Edge, Graph）
- 掌握图的构建和编译
- 学会使用条件边和循环
- 理解与 LangChain Chain 的区别

作者：AI Study Project
日期：2026-05-17
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated
from operator import add

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

logger = setup_logger('chapter4_langgraph')


def create_model(temperature: float = 0.7) -> ChatOpenAI:
    """创建模型实例"""
    config = get_config()
    model_config = config.get_model_config()
    
    llm = ChatOpenAI(
        **model_config,
        temperature=temperature
    )
    return llm


# ============================================================================
# 第一部分：理解 State（状态）
# ============================================================================

class SimpleState(TypedDict):
    """
    最简单的状态定义
    
    状态就是在图中流动的数据包
    所有节点都可以读取和修改它
    """
    message: str
    count: int


def demo_state_concept():
    """
    演示：理解状态的概念
    
    状态就像一个"共享笔记本"，每个节点都可以：
    - 读取：查看之前写的内容
    - 写入：添加新的内容
    """
    print("\n" + "=" * 70)
    print("📌 第一部分：理解 State（状态）")
    print("=" * 70)
    
    print("""
【直觉理解】
想象你在餐厅点餐，服务员有一个"订单本"：
- 你说"我要一份炒饭" → 服务员记下（状态更新）
- 厨师看订单本 → 知道要做炒饭（读取状态）
- 厨师做好 → 在订单本上打勾（状态更新）
- 收银员看订单本 → 知道收多少钱（读取状态）

这个"订单本"就是状态！

【代码定义】
class SimpleState(TypedDict):
    message: str    # 消息内容
    count: int      # 计数器

【关键点】
1. TypedDict 提供类型提示
2. 所有节点共享同一个状态
3. 节点返回的字典会合并到状态中
    """)
    
    print("✅ 状态概念演示完成！")


# ============================================================================
# 第二部分：理解 Node（节点）
# ============================================================================

def node_a(state: SimpleState) -> dict:
    """
    节点 A：增加计数
    
    节点就是一个函数：
    - 输入：当前状态
    - 输出：要更新的状态字段（部分更新）
    """
    print(f"  [节点A] 当前状态: message='{state['message']}', count={state['count']}")
    
    return {
        "message": state["message"] + " → A",
        "count": state["count"] + 1
    }


def node_b(state: SimpleState) -> dict:
    """节点 B：再次增加计数"""
    print(f"  [节点B] 当前状态: message='{state['message']}', count={state['count']}")
    
    return {
        "message": state["message"] + " → B",
        "count": state["count"] + 1
    }


def demo_node_concept():
    """
    演示：理解节点的概念
    
    节点是图中的处理单元
    """
    print("\n" + "=" * 70)
    print("📌 第二部分：理解 Node（节点）")
    print("=" * 70)
    
    print("""
【直觉理解】
节点就像工厂流水线上的"工位"：
- 工位A：检查产品质量
- 工位B：给产品贴标签
- 工位C：包装产品

每个工位：
- 接收产品（读取状态）
- 处理产品（执行逻辑）
- 传递给下一个工位（更新状态）

【节点函数的规则】
1. 必须接收 state 参数
2. 必须返回 dict（部分状态更新）
3. 返回的字典会自动合并到原状态

【示例】
def my_node(state: SimpleState) -> dict:
    # 处理逻辑
    return {"count": state["count"] + 1}  # 只更新 count
    """)
    
    print("\n现在让我们构建一个简单的图：A → B")
    
    graph = StateGraph(SimpleState)
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    app = graph.compile()
    
    print("\n初始状态: {'message': 'Start', 'count': 0}")
    print("\n执行过程:")
    
    result = app.invoke({"message": "Start", "count": 0})
    
    print(f"\n最终状态: {result}")
    print("\n✅ 节点概念演示完成！")


# ============================================================================
# 第三部分：理解 Edge（边）
# ============================================================================

class RouterState(TypedDict):
    """用于演示条件边的状态"""
    message: str
    route: str


def intent_node(state: RouterState) -> dict:
    """意图识别节点"""
    print(f"  [意图识别] 分析消息: {state['message']}")
    
    if "天气" in state["message"]:
        route = "weather"
    elif "新闻" in state["message"]:
        route = "news"
    else:
        route = "chat"
    
    print(f"  [意图识别] 识别结果: {route}")
    return {"route": route}


def weather_node(state: RouterState) -> dict:
    """天气查询节点"""
    print(f"  [天气查询] 处理天气请求")
    return {"message": "今天天气晴朗，温度25°C"}


def news_node(state: RouterState) -> dict:
    """新闻查询节点"""
    print(f"  [新闻查询] 处理新闻请求")
    return {"message": "今日头条：AI技术取得重大突破"}


def chat_node(state: RouterState) -> dict:
    """闲聊节点"""
    print(f"  [闲聊] 处理闲聊请求")
    return {"message": "你好！有什么我可以帮助你的吗？"}


def route_decision(state: RouterState) -> str:
    """
    路由函数：决定下一个节点
    
    这是条件边的核心
    """
    return state["route"]


def demo_edge_concept():
    """
    演示：理解边的概念
    
    边定义了节点之间的连接关系
    """
    print("\n" + "=" * 70)
    print("📌 第三部分：理解 Edge（边）")
    print("=" * 70)
    
    print("""
【直觉理解】
边就像道路，连接不同的地点（节点）：
- 普通道路：A → B，只能直走
- 岔路口：根据路标选择方向（条件边）
- 回头路：回到起点（循环）

【边的类型】

1. 普通边（无条件跳转）
   graph.add_edge("A", "B")
   含义：A 完成后，必定去 B

2. 条件边（根据状态决定）
   graph.add_conditional_edges(
       "A",
       route_function,  # 路由函数
       {
           "path1": "B",
           "path2": "C"
       }
   )
   含义：A 完成后，根据 route_function 的返回值决定去 B 还是 C

3. 自环边（循环）
   graph.add_edge("A", "A")
   含义：A 完成后，回到 A 继续执行
    """)
    
    print("\n现在让我们构建一个带条件边的图：")
    print("意图识别 → [天气/新闻/闲聊]")
    
    graph = StateGraph(RouterState)
    
    graph.add_node("intent", intent_node)
    graph.add_node("weather", weather_node)
    graph.add_node("news", news_node)
    graph.add_node("chat", chat_node)
    
    graph.set_entry_point("intent")
    
    graph.add_conditional_edges(
        "intent",
        route_decision,
        {
            "weather": "weather",
            "news": "news",
            "chat": "chat"
        }
    )
    
    graph.add_edge("weather", END)
    graph.add_edge("news", END)
    graph.add_edge("chat", END)
    
    app = graph.compile()
    
    test_cases = [
        {"message": "今天天气怎么样", "route": ""},
        {"message": "有什么新闻", "route": ""},
        {"message": "你好", "route": ""},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试案例 {i}: {case['message']}")
        print("=" * 60)
        
        result = app.invoke(case)
        print(f"\n最终回复: {result['message']}")
    
    print("\n✅ 边的概念演示完成！")


# ============================================================================
# 第四部分：循环与迭代
# ============================================================================

class LoopState(TypedDict):
    """用于演示循环的状态"""
    messages: Annotated[list[str], add]
    iteration: int
    is_complete: bool


def process_node(state: LoopState) -> dict:
    """处理节点"""
    iteration = state["iteration"]
    print(f"  [处理节点] 第 {iteration + 1} 次迭代")
    
    new_message = f"处理结果-{iteration + 1}"
    
    is_complete = iteration >= 2
    
    return {
        "messages": [new_message],
        "iteration": iteration + 1,
        "is_complete": is_complete
    }


def should_continue(state: LoopState) -> str:
    """决定是否继续循环"""
    if state["is_complete"]:
        return "end"
    return "continue"


def demo_loop_concept():
    """
    演示：循环与迭代
    
    这是 LangGraph 相比 Chain 的最大优势
    """
    print("\n" + "=" * 70)
    print("📌 第四部分：循环与迭代")
    print("=" * 70)
    
    print("""
【直觉理解】
循环就像"反复检查"：
- 检查作业 → 不满意 → 重写 → 再检查 → 满意 → 提交

【为什么 Chain 做不到？】
Chain 是线性的：A → B → C → D
无法回到之前的节点！

【LangGraph 的循环】
通过条件边实现：
- 条件函数返回 "continue" → 回到当前节点
- 条件函数返回 "end" → 结束

【关键技巧】
使用 Annotated[list, add] 来累积结果：
- 每次返回新的列表项
- 自动添加到原列表中
    """)
    
    print("\n构建循环图：处理 → 检查 → [继续/结束]")
    
    graph = StateGraph(LoopState)
    
    graph.add_node("process", process_node)
    
    graph.set_entry_point("process")
    
    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",
            "end": END
        }
    )
    
    app = graph.compile()
    
    print("\n初始状态: messages=[], iteration=0, is_complete=False")
    print("\n执行过程:")
    
    result = app.invoke({
        "messages": [],
        "iteration": 0,
        "is_complete": False
    })
    
    print(f"\n最终状态:")
    print(f"  messages: {result['messages']}")
    print(f"  iteration: {result['iteration']}")
    print(f"  is_complete: {result['is_complete']}")
    
    print("\n✅ 循环概念演示完成！")


# ============================================================================
# 第五部分：完整示例 - 智能客服系统
# ============================================================================

class CustomerServiceState(TypedDict):
    """智能客服系统状态"""
    user_input: str
    intent: str
    knowledge_result: str
    response: str
    messages: Annotated[list[str], add]
    need_human: bool


def cs_intent_node(state: CustomerServiceState) -> dict:
    """客服意图识别"""
    user_input = state["user_input"]
    print(f"\n  [意图识别] 分析: {user_input}")
    
    if "退款" in user_input or "投诉" in user_input:
        intent = "human"
        need_human = True
    elif "查询" in user_input or "订单" in user_input:
        intent = "query"
        need_human = False
    else:
        intent = "chat"
        need_human = False
    
    print(f"  [意图识别] 结果: {intent}")
    return {
        "intent": intent,
        "need_human": need_human,
        "messages": [f"[意图] {intent}"]
    }


def cs_knowledge_node(state: CustomerServiceState) -> dict:
    """知识库查询"""
    print(f"  [知识库] 查询相关信息...")
    
    result = "订单号: 12345, 状态: 已发货, 预计明天送达"
    
    print(f"  [知识库] 结果: {result}")
    return {
        "knowledge_result": result,
        "messages": [f"[知识库] {result}"]
    }


def cs_generate_node(state: CustomerServiceState) -> dict:
    """生成回复"""
    print(f"  [生成回复] 基于知识库结果生成...")
    
    if state["knowledge_result"]:
        response = f"根据查询结果：{state['knowledge_result']}"
    else:
        response = "您好，请问有什么可以帮助您的？"
    
    print(f"  [生成回复] 回复: {response}")
    return {
        "response": response,
        "messages": [f"[回复] {response}"]
    }


def cs_human_node(state: CustomerServiceState) -> dict:
    """转人工"""
    print(f"  [转人工] 问题复杂，转接人工客服...")
    
    return {
        "response": "您的问题已转接人工客服，请稍候...",
        "messages": ["[转人工] 已转接"]
    }


def cs_route_intent(state: CustomerServiceState) -> str:
    """根据意图路由"""
    return state["intent"]


def demo_customer_service():
    """
    完整示例：智能客服系统
    
    展示 LangGraph 的实际应用
    """
    print("\n" + "=" * 70)
    print("📌 第五部分：完整示例 - 智能客服系统")
    print("=" * 70)
    
    print("""
【系统流程】
用户输入 → 意图识别 → [查询/闲聊/转人工]
                           ↓
                    查询 → 知识库 → 生成回复
                           ↓
                      闲聊 → 生成回复
                           ↓
                    转人工 → 人工处理

【为什么用 LangGraph？】
1. 清晰的流程可视化
2. 条件分支自然表达
3. 状态自动管理
4. 易于扩展新节点
    """)
    
    graph = StateGraph(CustomerServiceState)
    
    graph.add_node("intent", cs_intent_node)
    graph.add_node("knowledge", cs_knowledge_node)
    graph.add_node("generate", cs_generate_node)
    graph.add_node("human", cs_human_node)
    
    graph.set_entry_point("intent")
    
    graph.add_conditional_edges(
        "intent",
        cs_route_intent,
        {
            "query": "knowledge",
            "chat": "generate",
            "human": "human"
        }
    )
    
    graph.add_edge("knowledge", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("human", END)
    
    app = graph.compile()
    
    test_cases = [
        "我想查询我的订单状态",
        "你好，请问你们营业时间？",
        "我要投诉，服务太差了！",
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"客服案例 {i}: {user_input}")
        print("=" * 60)
        
        result = app.invoke({
            "user_input": user_input,
            "intent": "",
            "knowledge_result": "",
            "response": "",
            "messages": [],
            "need_human": False
        })
        
        print(f"\n最终回复: {result['response']}")
        print(f"处理路径: {' → '.join(result['messages'])}")
    
    print("\n✅ 智能客服系统演示完成！")


# ============================================================================
# 第六部分：LangGraph vs LangChain Chain
# ============================================================================

def demo_comparison():
    """
    对比：LangGraph vs LangChain Chain
    
    帮助理解何时使用哪个
    """
    print("\n" + "=" * 70)
    print("📌 第六部分：LangGraph vs LangChain Chain")
    print("=" * 70)
    
    print("""
【LangChain Chain】
适用场景：简单的线性流程

示例：文章摘要
原文 → Prompt → LLM → 解析器 → 摘要

代码：
chain = prompt | llm | parser
result = chain.invoke({"text": "..."})

特点：
✅ 简单直观
✅ 代码简洁
❌ 无法分支
❌ 无法循环
❌ 状态管理有限

────────────────────────────────────────

【LangGraph】
适用场景：复杂的有状态流程

示例：智能客服
用户输入 → 意图识别 → [查询/闲聊/转人工]
                          ↓
                   查询 → 知识库 → 生成回复
                          ↓
                     闲聊 → 生成回复
                          ↓
                   转人工 → 人工处理

代码：
graph = StateGraph(State)
graph.add_node(...)
graph.add_conditional_edges(...)
app = graph.compile()
result = app.invoke({...})

特点：
✅ 支持分支
✅ 支持循环
✅ 完善的状态管理
✅ 可视化流程
❌ 代码较复杂

────────────────────────────────────────

【选择建议】

用 Chain 如果：
- 流程是线性的
- 不需要条件分支
- 不需要循环
- 快速原型开发

用 LangGraph 如果：
- 流程有分支
- 需要循环迭代
- 需要复杂状态管理
- 多 Agent 协作
- 生产级应用
    """)
    
    print("✅ 对比演示完成！")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：运行所有演示
    
    学习路径：
    1. 理解 State（状态）
    2. 理解 Node（节点）
    3. 理解 Edge（边）
    4. 循环与迭代
    5. 完整示例
    6. 对比分析
    """
    print("=" * 70)
    print("🎯 LangGraph 完全指南：从零基础到精通")
    print("=" * 70)
    
    logger.info("开始 LangGraph 演示...")
    
    demos = [
        ("理解 State", demo_state_concept),
        ("理解 Node", demo_node_concept),
        ("理解 Edge", demo_edge_concept),
        ("循环与迭代", demo_loop_concept),
        ("完整示例", demo_customer_service),
        ("对比分析", demo_comparison),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 LangGraph 完全指南演示完成！")
    print("=" * 70)
    
    print("""
【学习总结】

1. State（状态）
   - 在图中流动的数据包
   - 所有节点共享
   - 使用 TypedDict 定义

2. Node（节点）
   - 处理单元
   - 接收状态，返回更新
   - 函数形式定义

3. Edge（边）
   - 连接节点
   - 普通边：无条件跳转
   - 条件边：根据状态决定

4. Graph（图）
   - 节点 + 边的集合
   - 需要编译才能运行
   - 支持可视化和调试

【下一步学习】
- 学习 LangGraph 的 Checkpointer（状态持久化）
- 学习多 Agent 协作模式
- 学习 Human-in-the-loop（人工介入）
- 学习流式输出和异步执行
    """)
    
    logger.info("LangGraph 演示完成")


if __name__ == '__main__':
    main()
