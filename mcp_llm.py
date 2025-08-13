import os
import asyncio
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


client = MultiServerMCPClient(
    {
        # "finnhub": {
        #     "command": "python",
        #     # Replace with absolute path to your math_server.py file
        #     "args": ["C:/Users/dppar/OneDrive - Virginia Tech/Desktop/Crowdalpha-DhairyaLaptop/finnhub_mcp_server.py"],
        #     "transport": "stdio",
        # },
        # "talib": {
        #     "command": "python",    
        #     # Replace with absolute path to your talib_mcp_server.py file
        #     "args": ["C:/Users/dppar/OneDrive - Virginia Tech/Desktop/Crowdalpha-DhairyaLaptop/talib_mcp_server.py"],
        #     "transport": "stdio",   
        # },
        "yfinance": {
            "command": "python",
            # Replace with absolute path to your yfinance_mcp_server.py file
            "args": ["C:/Users/dppar/OneDrive - Virginia Tech/Desktop/Crowdalpha-DhairyaLaptop/yfinance_mcp_server.py"],
            "transport": "stdio",
        },
        "reddit": {
            "command": "python",
            # Replace with absolute path to your reddit_mcp_server.py file
            "args": ["C:/Users/dppar/OneDrive - Virginia Tech/Desktop/Crowdalpha-DhairyaLaptop/reddit_mcp_server.py"],
            "transport": "stdio",
        },
    }
)


def get_model_groq():
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        reasoning_effort="low",
        api_key=groq_api_key,
        temperature=0,
        streaming=True,
    )
    return model


async def get_model_with_tools():
    tools = await client.get_tools()
    model = get_model_groq()
    model = model.bind_tools(tools)
    return model, tools


def get_tools_by_name(tools: list[tool]):
    tools_by_name = {tool.name: tool for tool in tools}
    return tools_by_name


async def tool_node(state: AgentState, config: RunnableConfig):
    outputs = []
    tools_by_name = config["metadata"]["__tools_by_name__"]
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    model = config["metadata"]["__model__"]
    system_prompt = SystemMessage(
        """You are a Financial Assistant, use the tools to answer user question, you are allowed to use mutliple tools to 
        answer user's query. You should carefully understand users needs and use tool whenever required, it's better to do
        deep analyses by getting input from different tools."""
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def create_agent_graph(model: ChatGroq, tools_by_name: dict[str, tool]):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model, metadata={"__model__": model})
    workflow.add_node("tools", tool_node, metadata={"__tools_by_name__": tools_by_name})

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    return graph


async def main():
    user_input = input("Enter your question: ")
    inputs = {
        "messages": [
            (
                "user",
                user_input
            )
        ]
    }
    model, tools = await get_model_with_tools()
    tools_by_name = get_tools_by_name(tools)
    print(f"Tools: {tools_by_name.keys()}")
    graph = create_agent_graph(model=model, tools_by_name=tools_by_name)
    async for output in graph.astream(inputs, stream_mode="updates"):
        if output.get("agent"):
            msg = output.get("agent")
            for m in msg.get("messages", []):
                if m.content:
                    print(m.content)
                elif m.additional_kwargs.get("tool_calls"):
                    for tool_call in m.additional_kwargs.get("tool_calls", []):
                        print(f"Tool call: {tool_call}")
        else:
            msg = output.get("tools", [])
            for m in msg.get("messages", []):
                if m.content:
                    print(f"Result from tool {m.name}: \n {m.content}")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
