"""
LangGraph 进阶教程：高级特性与最佳实践

学习目标：
- 理解状态持久化（Checkpointer）
- 掌握多 Agent 协作模式
- 学习 Human-in-the-loop（人工介入）
- 理解流式输出和异步执行
- 掌握图的可视化

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

logger = setup_logger('chapter4_langgraph_advanced')


# ============================================================================
# 知识图谱：LangGraph 在 AI 应用架构中的位置
# ============================================================================

def show_knowledge_graph():
    """
    展示 LangGraph 的知识关联图谱
    """
    print("\n" + "=" * 70)
    print("📊 LangGraph 知识关联图谱")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    AI 应用架构全景图                                  │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   用户请求        │
                    └────────┬─────────┘
                             │
                             ↓
         ┌───────────────────────────────────────┐
         │         应用层（Application）          │
         │                                       │
         │  ┌─────────────┐    ┌──────────────┐ │
         │  │ LangChain   │    │  LangGraph   │ │
         │  │ Chain       │    │  (本教程)    │ │
         │  │ (线性流程)  │    │  (图流程)    │ │
         │  └─────────────┘    └──────────────┘ │
         │           │                  │        │
         │           └────────┬─────────┘        │
         │                    │                  │
         └────────────────────┼──────────────────┘
                              │
                              ↓
         ┌───────────────────────────────────────┐
         │         框架层（Framework）            │
         │                                       │
         │  • LangChain Core    • LangSmith      │
         │  • LangServe         • LangGraph      │
         └───────────────────────┬───────────────┘
                                 │
                                 ↓
         ┌───────────────────────────────────────┐
         │         模型层（Model）                │
         │                                       │
         │  • OpenAI    • Claude    • Gemini    │
         │  • 本地模型   • 开源模型              │
         └───────────────────────┬───────────────┘
                                 │
                                 ↓
         ┌───────────────────────────────────────┐
         │         基础设施层（Infrastructure）   │
         │                                       │
         │  • 向量数据库    • 向量索引           │
         │  • 缓存系统      • 监控系统           │
         └───────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph 核心概念关系图                          │
└─────────────────────────────────────────────────────────────────────┘

                         ┌──────────┐
                         │  State   │ ← 核心数据结构
                         │  (状态)  │
                         └────┬─────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ↓               ↓               ↓
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Node    │   │  Edge    │   │  Graph   │
        │  (节点)  │   │  (边)    │   │  (图)    │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ↓
                   ┌────────────────┐
                   │  Compiled App  │
                   │  (可执行应用)  │
                   └────────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ↓             ↓             ↓
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ invoke() │  │ stream() │  │ batch()  │
        │ 同步执行 │  │ 流式执行 │  │ 批量执行 │
        └──────────┘  └──────────┘  └──────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph 高级特性关系图                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │   Checkpointer  │ ← 状态持久化
  │   (检查点)      │
  └────────┬────────┘
           │
           ↓
  ┌─────────────────┐      ┌─────────────────┐
  │  Memory         │ ←─── │  Conversation   │
  │  (记忆)         │      │  (对话历史)     │
  └─────────────────┘      └─────────────────┘

  ┌─────────────────┐
  │  Human-in-loop  │ ← 人工介入
  │  (人工循环)     │
  └────────┬────────┘
           │
           ├──→ 审批节点
           ├──→ 修正节点
           └──→ 反馈节点

  ┌─────────────────┐
  │  Multi-Agent    │ ← 多智能体协作
  │  (多智能体)     │
  └────────┬────────┘
           │
           ├──→ Supervisor 模式
           ├──→ Hierarchical 模式
           └──→ Collaborative 模式


【概念依赖关系】

State → Node → Edge → Graph → Compiled App

Checkpointer → Memory → Conversation
Human-in-loop → Approval/Correction/Feedback
Multi-Agent → Supervisor/Hierarchical/Collaborative


【学习路径建议】

Level 1: 基础概念
  └─→ State, Node, Edge, Graph

Level 2: 核心操作
  └─→ 编译、执行、调试

Level 3: 高级特性
  └─→ Checkpointer, Memory, Human-in-loop

Level 4: 多智能体
  └─→ Multi-Agent 协作模式

Level 5: 生产实践
  └─→ 性能优化、错误处理、监控
    """)


# ============================================================================
# 第一部分：状态持久化（Checkpointer）
# ============================================================================

def demo_checkpointer_concept():
    """
    演示：状态持久化的概念
    
    为什么需要持久化？
    - 长时间运行的任务需要保存中间状态
    - 对话历史需要保存
    - 错误恢复需要从检查点继续
    """
    print("\n" + "=" * 70)
    print("📌 第一部分：状态持久化（Checkpointer）")
    print("=" * 70)
    
    print("""
【直觉理解】
想象你在玩一个闯关游戏：
- 每过一关，游戏自动保存（创建检查点）
- 如果游戏崩溃，可以从最近的检查点继续
- 如果想回到之前的关卡，可以加载旧检查点

【为什么需要 Checkpointer？】

1. 长时间运行的任务
   - 任务可能运行几小时甚至几天
   - 需要保存中间结果
   - 系统重启后可以继续

2. 对话历史保存
   - 用户关闭页面后重新打开
   - 需要恢复之前的对话上下文
   - 保持对话连贯性

3. 错误恢复
   - 某个节点执行失败
   - 不需要从头开始
   - 从上一个检查点继续

【LangGraph 提供的 Checkpointer】

1. MemorySaver
   - 保存在内存中
   - 适用于开发和测试
   - 重启后数据丢失

2. SqliteSaver
   - 保存在 SQLite 数据库
   - 适用于单机应用
   - 数据持久化到磁盘

3. PostgresSaver
   - 保存在 PostgreSQL 数据库
   - 适用于生产环境
   - 支持分布式部署

【代码示例】

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 内存检查点
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# SQLite 检查点
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)
    
    # 执行时指定 thread_id
    config = {"configurable": {"thread_id": "conversation-1"}}
    result = app.invoke(input, config)

【关键概念：thread_id】

thread_id 用于区分不同的对话或任务：
- 每个用户一个 thread_id
- 每个任务一个 thread_id
- 相同 thread_id 共享状态

示例：
用户A: thread_id = "user-A-conversation"
用户B: thread_id = "user-B-conversation"
    """)
    
    print("✅ Checkpointer 概念演示完成！")


# ============================================================================
# 第二部分：Human-in-the-loop（人工介入）
# ============================================================================

def demo_human_in_loop_concept():
    """
    演示：Human-in-the-loop 的概念
    
    应用场景：
    - AI 生成内容需要人工审核
    - 关键决策需要人工确认
    - AI 不确定时请求人工帮助
    """
    print("\n" + "=" * 70)
    print("📌 第二部分：Human-in-the-loop（人工介入）")
    print("=" * 70)
    
    print("""
【直觉理解】
想象一个自动审批系统：
- AI 自动处理大部分简单申请
- 遇到复杂或高风险申请，暂停并等待人工审核
- 人工审核后，AI 继续处理

【为什么需要 Human-in-the-loop？】

1. 安全性
   - AI 可能做出错误决策
   - 关键操作需要人工确认
   - 例如：发送邮件、转账、删除数据

2. 质量控制
   - AI 生成的内容需要审核
   - 确保符合品牌调性
   - 避免敏感或不当内容

3. 处理边界情况
   - AI 遇到无法处理的情况
   - 请求人工帮助
   - 学习人工的处理方式

【实现方式】

方式1：interrupt_before
在特定节点前暂停

graph.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_action"]
)

方式2：interrupt_after
在特定节点后暂停

graph.compile(
    checkpointer=memory,
    interrupt_after=["generate_content"]
)

方式3：动态中断
在节点函数中决定是否中断

def my_node(state):
    if state["confidence"] < 0.8:
        return {"__interrupt__": "需要人工审核"}
    return {"result": "自动处理"}

【完整流程】

1. AI 执行到暂停点
2. 返回当前状态和等待信息
3. 人工查看并做出决定
4. 更新状态
5. AI 继续执行

【代码示例】

# 编译时设置中断点
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["send_email"]
)

# 第一次执行，会在 send_email 前暂停
result1 = app.invoke(input, config)
# 状态: {"status": "waiting_approval", ...}

# 人工审核后，继续执行
result2 = app.invoke(None, config)  # None 表示继续之前的执行
    """)
    
    print("✅ Human-in-the-loop 概念演示完成！")


# ============================================================================
# 第三部分：多 Agent 协作模式
# ============================================================================

def demo_multi_agent_concept():
    """
    演示：多 Agent 协作模式
    
    应用场景：
    - 复杂任务需要多个专业 Agent
    - 每个 Agent 负责特定领域
    - Agent 之间协作完成任务
    """
    print("\n" + "=" * 70)
    print("📌 第三部分：多 Agent 协作模式")
    print("=" * 70)
    
    print("""
【直觉理解】
想象一个公司：
- 销售部门：负责客户沟通
- 技术部门：负责技术方案
- 财务部门：负责报价和合同
- 总经理：协调各部门工作

每个部门是一个 Agent，总经理是 Supervisor。

【三种协作模式】

┌─────────────────────────────────────────────────────────────────────┐
│ 模式1：Supervisor（主管模式）                                        │
└─────────────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │  Supervisor  │ ← 协调者
         │    Agent     │
         └──────┬───────┘
                │
      ┌─────────┼─────────┐
      │         │         │
      ↓         ↓         ↓
  ┌───────┐ ┌───────┐ ┌───────┐
  │Agent1 │ │Agent2 │ │Agent3 │
  │(搜索) │ │(分析) │ │(写作) │
  └───────┘ └───────┘ └───────┘

工作流程：
1. Supervisor 接收任务
2. 分析任务，分配给合适的 Agent
3. Agent 执行并返回结果
4. Supervisor 汇总结果，决定下一步
5. 重复直到任务完成

─────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────┐
│ 模式2：Hierarchical（层级模式）                                      │
└─────────────────────────────────────────────────────────────────────┘

           ┌──────────────┐
           │ Top-level    │
           │ Supervisor   │
           └──────┬───────┘
                  │
        ┌─────────┼─────────┐
        │                   │
        ↓                   ↓
  ┌──────────┐       ┌──────────┐
  │ Mid-level│       │ Mid-level│
  │Supervisor│       │Supervisor│
  └────┬─────┘       └────┬─────┘
       │                  │
   ┌───┼───┐          ┌───┼───┐
   │   │   │          │   │   │
   ↓   ↓   ↓          ↓   ↓   ↓
  A1  A2  A3         A4  A5  A6

适用场景：
- 大型复杂项目
- 需要分层管理
- 每层有专门的协调者

─────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────┐
│ 模式3：Collaborative（协作模式）                                     │
└─────────────────────────────────────────────────────────────────────┘

     ┌───────┐
     │Agent1 │←──────┐
     └───┬───┘       │
         │           │
         ↓           │
     ┌───────┐       │
     │Agent2 │───────┤
     └───┬───┘       │
         │           │
         ↓           │
     ┌───────┐       │
     │Agent3 │───────┘
     └───────┘

特点：
- Agent 之间直接通信
- 没有中央协调者
- 更灵活，但可能混乱

【代码示例：Supervisor 模式】

from langgraph.supervisor import create_supervisor

# 定义专业 Agent
search_agent = create_agent(llm, tools=[search_tool])
analysis_agent = create_agent(llm, tools=[analysis_tool])
writing_agent = create_agent(llm, tools=[writing_tool])

# 创建 Supervisor
supervisor = create_supervisor(
    llm,
    agents=[search_agent, analysis_agent, writing_agent],
    prompt="你是一个协调者，负责分配任务给合适的 Agent"
)

# 编译并运行
app = supervisor.compile()
result = app.invoke({"task": "写一篇关于 AI 的文章"})

【实际应用场景】

1. 研究助手
   - 搜索 Agent：查找资料
   - 分析 Agent：提取关键信息
   - 写作 Agent：生成报告

2. 客服系统
   - 意图识别 Agent：理解用户需求
   - 知识库 Agent：查询答案
   - 对话 Agent：生成回复

3. 代码助手
   - 需求分析 Agent：理解需求
   - 架构设计 Agent：设计方案
   - 代码生成 Agent：编写代码
   - 测试 Agent：编写测试
    """)
    
    print("✅ 多 Agent 协作模式演示完成！")


# ============================================================================
# 第四部分：流式输出与异步执行
# ============================================================================

def demo_streaming_concept():
    """
    演示：流式输出与异步执行
    
    为什么需要流式输出？
    - LLM 生成是逐 token 的
    - 用户希望实时看到输出
    - 提升用户体验
    """
    print("\n" + "=" * 70)
    print("📌 第四部分：流式输出与异步执行")
    print("=" * 70)
    
    print("""
【直觉理解】
想象你在看直播：
- 传统方式：等主播说完一整段话，你才能听到
- 流式方式：主播说一句，你听一句，实时体验

【为什么需要流式输出？】

1. 用户体验
   - LLM 生成可能需要几秒到几十秒
   - 用户不想盯着空白屏幕等待
   - 流式输出让用户感觉更快

2. 长文本生成
   - 生成几千字的文章
   - 用户可以边看边判断
   - 如果方向不对可以提前停止

3. 实时反馈
   - 在生成过程中显示进度
   - 用户可以看到 AI 的"思考过程"
   - 增加信任感

【LangGraph 的流式模式】

1. stream() - 流式执行
   for event in app.stream(input, config):
       if event["event"] == "on_chain_start":
           print(f"开始执行: {event['name']}")
       elif event["event"] == "on_chain_end":
           print(f"执行完成: {event['name']}")
       elif event["event"] == "on_llm_stream":
           print(event["data"]["chunk"], end="", flush=True)

2. astream() - 异步流式执行
   async for event in app.astream(input, config):
       # 处理事件
       pass

3. stream_events() - 详细事件流
   async for event in app.astream_events(input, config):
       print(f"{event['event']}: {event['name']}")

【事件类型】

┌─────────────────────┬───────────────────────────────┐
│ 事件类型            │ 说明                           │
├─────────────────────┼───────────────────────────────┤
│ on_chain_start      │ 链开始执行                     │
│ on_chain_end        │ 链执行完成                     │
│ on_llm_start        │ LLM 开始生成                   │
│ on_llm_stream       │ LLM 生成 token                 │
│ on_llm_end          │ LLM 生成完成                   │
│ on_tool_start       │ 工具开始执行                   │
│ on_tool_end         │ 工具执行完成                   │
└─────────────────────┴───────────────────────────────┘

【异步执行的优势】

1. 并发处理
   - 同时处理多个请求
   - 提高吞吐量

2. 非阻塞
   - 不阻塞主线程
   - 可以做其他事情

3. 资源效率
   - 更高效地使用资源
   - 减少等待时间

【代码示例：异步流式】

import asyncio

async def process_request():
    async for event in app.astream(input, config):
        if event["event"] == "on_llm_stream":
            token = event["data"]["chunk"]
            print(token, end="", flush=True)

# 并发处理多个请求
await asyncio.gather(
    process_request(request1),
    process_request(request2),
    process_request(request3)
)
    """)
    
    print("✅ 流式输出与异步执行演示完成！")


# ============================================================================
# 第五部分：图的可视化
# ============================================================================

def demo_visualization_concept():
    """
    演示：图的可视化
    
    为什么需要可视化？
    - 理解复杂的工作流程
    - 调试和优化
    - 文档和沟通
    """
    print("\n" + "=" * 70)
    print("📌 第五部分：图的可视化")
    print("=" * 70)
    
    print("""
【直觉理解】
想象你在设计一个工厂流水线：
- 文字描述很难理解
- 画成流程图一目了然
- 可以发现问题和优化点

【LangGraph 的可视化方法】

1. get_graph() - 获取图结构
   graph = app.get_graph()
   print(graph.draw_ascii())

2. Mermaid 图
   from langgraph.graph import StateGraph
   
   graph = StateGraph(State)
   # ... 添加节点和边
   
   # 生成 Mermaid 图
   mermaid_code = graph.get_graph().draw_mermaid()
   print(mermaid_code)

3. ASCII 图（终端显示）
   ascii_graph = graph.get_graph().draw_ascii()
   print(ascii_graph)

【ASCII 图示例】

           ┌──────────┐
           │ __start__│
           └─────┬────┘
                 │
                 ↓
          ┌────────────┐
          │   intent   │
          └─────┬──────┘
                │
        ┌───────┼───────┐
        │       │       │
        ↓       ↓       ↓
   ┌────────┐┌────────┐┌────────┐
   │weather ││  news  ││  chat  │
   └───┬────┘└───┬────┘└───┬────┘
       │         │         │
       └─────────┼─────────┘
                 │
                 ↓
          ┌──────────┐
          │ __end__  │
          └──────────┘

【Mermaid 图示例】

graph TD
    A[__start__] --> B[intent]
    B --> C{route}
    C -->|weather| D[weather]
    C -->|news| E[news]
    C -->|chat| F[chat]
    D --> G[__end__]
    E --> G
    F --> G

【可视化的用途】

1. 调试
   - 检查图结构是否正确
   - 发现孤立节点或死循环
   - 理解执行路径

2. 优化
   - 发现冗余节点
   - 优化执行路径
   - 合并相似节点

3. 文档
   - 生成文档图表
   - 向团队解释流程
   - 用户手册插图

4. 监控
   - 实时显示执行状态
   - 显示当前在哪个节点
   - 性能分析
    """)
    
    print("✅ 图的可视化演示完成！")


# ============================================================================
# 第六部分：最佳实践与常见陷阱
# ============================================================================

def demo_best_practices():
    """
    演示：最佳实践与常见陷阱
    """
    print("\n" + "=" * 70)
    print("📌 第六部分：最佳实践与常见陷阱")
    print("=" * 70)
    
    print("""
【最佳实践】

1. 状态设计
   ✅ 保持状态最小化
   ✅ 使用 TypedDict 提供类型提示
   ✅ 使用 Annotated 处理累积字段
   
   ❌ 不要在状态中存储大对象
   ❌ 不要使用可变对象（如 list）而不加 Annotated

2. 节点设计
   ✅ 每个节点只做一件事
   ✅ 节点函数要幂等（可重复执行）
   ✅ 返回部分状态更新，不是完整状态
   
   ❌ 不要在节点中做 I/O 操作（应该用工具）
   ❌ 不要依赖外部状态

3. 边设计
   ✅ 使用有意义的路由键名
   ✅ 条件函数要简单明确
   ✅ 确保所有路径都有出口
   
   ❌ 不要创建复杂的嵌套条件
   ❌ 不要忘记处理所有可能的情况

4. 性能优化
   ✅ 使用异步执行提高并发
   ✅ 使用流式输出提升用户体验
   ✅ 使用 Checkpointer 保存中间结果
   
   ❌ 不要在同步代码中调用 LLM
   ❌ 不要等待整个流程完成才返回

─────────────────────────────────────────────────────────────────────

【常见陷阱】

陷阱1：状态累积导致内存爆炸
问题：每次迭代都往列表添加数据，最终内存溢出

解决：
class State(TypedDict):
    messages: Annotated[list, add]  # ✅ 正确
    # messages: list  # ❌ 错误，会替换而不是累积

陷阱2：忘记设置 END
问题：图没有出口，永远不结束

解决：
graph.add_edge("last_node", END)  # ✅ 必须有 END

陷阱3：条件边覆盖不全
问题：路由函数返回的键不在映射中

解决：
graph.add_conditional_edges(
    "node",
    route_func,
    {
        "path1": "node1",
        "path2": "node2",
        "path3": "node3",  # ✅ 确保所有路径都有
    }
)

陷阱4：状态字段命名冲突
问题：不同节点返回相同字段名，导致覆盖

解决：
class State(TypedDict):
    node1_result: str  # ✅ 使用前缀区分
    node2_result: str

陷阱5：忘记编译
问题：直接调用未编译的图

解决：
app = graph.compile()  # ✅ 必须先编译
result = app.invoke(input)

─────────────────────────────────────────────────────────────────────

【调试技巧】

1. 打印图结构
   print(app.get_graph().draw_ascii())

2. 使用 LangSmith
   设置环境变量启用追踪：
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=your_key

3. 添加调试节点
   def debug_node(state):
       print(f"当前状态: {state}")
       return state

4. 流式执行查看过程
   for event in app.stream(input):
       print(event)

5. 使用 Checkpointer 查看历史
   history = app.get_state_history(config)
   for state in history:
       print(state)
    """)
    
    print("✅ 最佳实践与常见陷阱演示完成！")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：运行所有进阶演示
    """
    print("=" * 70)
    print("🎯 LangGraph 进阶教程：高级特性与最佳实践")
    print("=" * 70)
    
    logger.info("开始 LangGraph 进阶演示...")
    
    demos = [
        ("知识图谱", show_knowledge_graph),
        ("状态持久化", demo_checkpointer_concept),
        ("Human-in-the-loop", demo_human_in_loop_concept),
        ("多 Agent 协作", demo_multi_agent_concept),
        ("流式输出", demo_streaming_concept),
        ("图的可视化", demo_visualization_concept),
        ("最佳实践", demo_best_practices),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {name} 失败: {e}", exc_info=True)
            print(f"\n❌ 演示 {name} 失败: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 LangGraph 进阶教程演示完成！")
    print("=" * 70)
    
    print("""
【学习路径总结】

第一阶段：基础掌握
├─ State, Node, Edge, Graph
├─ 编译和执行
└─ 简单的分支和循环

第二阶段：高级特性
├─ Checkpointer（状态持久化）
├─ Human-in-the-loop（人工介入）
└─ 流式输出和异步执行

第三阶段：多智能体
├─ Supervisor 模式
├─ Hierarchical 模式
└─ Collaborative 模式

第四阶段：生产实践
├─ 性能优化
├─ 错误处理
├─ 监控和调试
└─ 可视化

【推荐学习资源】

1. 官方文档
   https://langchain-ai.github.io/langgraph/

2. 示例代码
   https://github.com/langchain-ai/langgraph/tree/main/examples

3. 教程视频
   LangChain 官方 YouTube 频道

4. 社区讨论
   LangChain Discord 社区
    """)
    
    logger.info("LangGraph 进阶演示完成")


if __name__ == '__main__':
    main()
