
import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage
from mcp_llm import get_model_with_tools, create_agent_graph, get_tools_by_name

chat_histories = {}


async def get_answer_from_question(question, chat_id):
    chat_history = chat_histories.setdefault(chat_id, [])
    initial_state = {"messages": [HumanMessage(content=question)]}

    model, tools = await get_model_with_tools()
    tools_by_name = get_tools_by_name(tools)
    print(f"Tools: {tools_by_name.keys()}")
    graph = create_agent_graph(model=model, tools_by_name=tools_by_name)

    result = await graph.ainvoke(initial_state)
    ai_reply = result.get("messages", [])[-1].content

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=ai_reply))
    return ai_reply, str(chat_history)