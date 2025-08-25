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



# ---- BEGIN: compact tool result ----
import json
from typing import Any

def _truncate_list(seq, max_len=60):
    if not isinstance(seq, list): 
        return seq
    return seq[-max_len:] if len(seq) > max_len else seq

def _summarize_price_history(obj: dict, max_len=60) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    tkr = obj.get("ticker"); tf = obj.get("timeframe"); per = obj.get("period")
    c   = obj.get("candles", []); n_total = len(c); c = _truncate_list(c, max_len)
    if not c: return f"[price-history] {tkr or ''} {tf or ''} {per or ''}: 0 bars"
    first_t, last_t = c[0].get("t", "N/A"), c[-1].get("t", "N/A")
    last = c[-1]; O,H,L,C = last.get("o"), last.get("h"), last.get("l"), last.get("c")
    return (f"[price-history] {tkr} {tf} {per}: {len(c)} bars (from {n_total}) — "
            f"{first_t} → {last_t}; last OHLC: O={O} H={H} L={L} C={C}")

def _summarize_indicators(obj: dict, max_len=10) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    rows = obj.get("rows", []); n_total = len(rows); rows = _truncate_list(rows, max_len)
    if not rows: return "[indicators] 0 rows"
    last = rows[-1]; date = last.get("Date","N/A"); close = last.get("Close","N/A")
    rsi = last.get("RSI_14") or last.get("RSI"); ema20 = last.get("EMA_20")
    ema50 = last.get("EMA_50"); ema200 = last.get("EMA_200")
    macd = last.get("MACD"); macds = last.get("MACD_signal")
    return (f"[indicators] {len(rows)} rows (from {n_total}); "
            f"last={date} close={close} RSI={rsi} "
            f"EMA20={ema20} EMA50={ema50} EMA200={ema200} MACD={macd} vs {macds}")

def _summarize_tool_result(tool_name: str, tool_result: any) -> str:
    import json

    # normalize to dict if possible
    if isinstance(tool_result, str):
        try:
            parsed = json.loads(tool_result)
        except Exception:
            return tool_result[:800] + "…" if len(tool_result) > 800 else tool_result
    else:
        try:
            parsed = json.loads(json.dumps(tool_result, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            parsed = tool_result if isinstance(tool_result, dict) else {}

    # 1) show explicit tool errors
    if isinstance(parsed, dict) and "error" in parsed:
        return f"[tool:{tool_name}] error: {parsed['error']}"

    # 2) quick summary when prediction is present
    if isinstance(parsed, dict) and isinstance(parsed.get("prediction"), dict):
        pr = parsed["prediction"]
        sym = parsed.get("symbol", "?")
        ts  = parsed.get("asof", "")
        hz  = parsed.get("horizon", "?")
        if "last_price" in pr and "predicted_price" in pr:
            return (f"{sym} last={pr['last_price']:.2f}, "
                    f"pred(+{hz})={pr['predicted_price']:.2f} "
                    f"({pr.get('expected_pct', 0):.2f}%); as of {ts}; "
                    f"verdict={parsed.get('verdict','?')}")

    # 3) legacy paths you already handled
    if isinstance(parsed, dict):
        if "candles" in parsed:   return _summarize_price_history(parsed)
        if "rows" in parsed:      return _summarize_indicators(parsed)
        return f"[tool:{tool_name}] keys={list(parsed.keys())[:10]}…"
    return str(parsed)

# ---- END: compact tool result ----



client = MultiServerMCPClient(
    {
        "finnhub": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["./finnhub_mcp_server.py"],
            "transport": "stdio",
        },
        "talib": {
            "command": "python",    
            # Replace with absolute path to your talib_mcp_server.py file
            "args": ["./talib_mcp_server.py"],
            "transport": "stdio",   
        },
        "yfinance": {
            "command": "python",
            # Replace with absolute path to your yfinance_mcp_server.py file
            "args": ["./yfinance_mcp_server.py"],
            "transport": "stdio",
        },
        "reddit": {
            "command": "python",
            # Replace with absolute path to your reddit_mcp_server.py file
            "args": ["./reddit_mcp_server.py"],
            "transport": "stdio",
        },
        "mlModel_MCP": {
            "command": "python",
            # Replace with absolute path to your mlModel_MCP.py file
            "args": ["./mlModel_MCP.py"],
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
        summary = _summarize_tool_result(tool_call["name"], tool_result)
        outputs.append(
            ToolMessage(
                # content=json.dumps(tool_result),
                content=summary,
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
