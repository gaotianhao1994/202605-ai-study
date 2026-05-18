from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from dotenv import load_dotenv
import os

load_dotenv()

class SimpleAgent:
    def __init__(self, tools=None, model_name=None, temperature=0.1):
        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL_NAME", "qwen3.5-122b-a10b")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.tools = tools or []
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def add_tool(self, tool):
        if isinstance(tool, Tool):
            self.tools.append(tool)
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            raise ValueError("工具必须是 langchain.tools.Tool 类型")
    
    def run(self, prompt):
        try:
            response = self.agent.run(prompt)
            return response
        except Exception as e:
            return f"执行过程中发生错误: {str(e)}"
    
    def get_chat_history(self):
        return self.memory.load_memory_variables({})

if __name__ == "__main__":
    agent = SimpleAgent()
    print("SimpleAgent 初始化完成！")
    print("可用工具:", [tool.name for tool in agent.tools])