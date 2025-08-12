# graph.py
from __future__ import annotations
import os, time, sys, datetime as dt, random
from typing import Literal, Optional, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# === 1) LLM (for the LLM-policy branch) ===
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("Gemini_API"),
    temperature=0.2,
)

# === 2) Import your unified adapter ===
from adapter import (
    Candle, PricesOut, TAFeatures, RedditOut, FinnhubNewsOut,
    get_prices_yf,
    compute_ta_from_phase1 as compute_ta_indicators,
    get_reddit_signals,
    get_company_news_finnhub,
)

# === 3) Graph State ===
class State(TypedDict, total=False):
    ticker: str
    date: str
    mode: Literal["paper","live"]

    prices: PricesOut
    features: TAFeatures
    reddit: RedditOut
    finnhub_news: Dict[str, Any]

    regime: Dict[str, Any]
    route: Dict[str, Any]
    decision: Dict[str, Any]
    risk: Dict[str, Any]
    execution: Dict[str, Any]
    metrics: Dict[str, Any]
    summary: str

# === 4) Nodes ===

@traceable(name="node:fetch_prices", run_type="chain")
def fetch_prices(state: State) -> State:
    p = get_prices_yf(state["ticker"])
    state["prices"] = p
    return state

@traceable(name="node:features_talib", run_type="chain")
def features_talib(state: State) -> State:
    feats = compute_ta_indicators(state["ticker"], period="12mo", interval="1d")
    state["features"] = feats
    return state

@traceable(name="node:reddit_ingest", run_type="chain")
def reddit_ingest(state: State) -> State:
    r = get_reddit_signals(state["ticker"], limit_per_sub=60, time_filter="week")
    state["reddit"] = r
    return state

@traceable(name="node:finnhub_news", run_type="chain")
def finnhub_news(state: State) -> State:
    to_d = dt.date.today()
    from_d = to_d - dt.timedelta(days=7)
    news = get_company_news_finnhub(state["ticker"], str(from_d), str(to_d))
    state["finnhub_news"] = news.model_dump()
    return state

@traceable(name="node:regime_detect", run_type="chain")
def regime_detect(state: State) -> State:
    f: TAFeatures = state["features"]
    if (f.trend_score or 0) > 0.25:
        reg = {"regime":"trend","confidence":min(1.0, 0.5 + (f.trend_score or 0))}
    elif (f.vol_score or 0) > 0.6:
        reg = {"regime":"high_vol","confidence":0.6}
    elif (f.trend_score or 0) < 0.1:
        reg = {"regime":"range","confidence":0.6}
    else:
        reg = {"regime":"uncertain","confidence":0.5}
    state["regime"] = reg
    return state

@traceable(name="node:strategy_router", run_type="chain")
def strategy_router(state: State) -> State:
    f: TAFeatures = state["features"]
    r: Optional[RedditOut] = state.get("reddit")
    trend = f.trend_score or 0.0
    rsi = f.RSI if f.RSI is not None else None
    r_conf = (r.sentiment["confidence"] if r else 0.0)
    event = (r.event_flags["earnings"] if r else False)

    if trend > 0.25:
        route = {"strategy":"momentum","reason":f"trend_score={trend:.2f}"}
    elif trend < 0.1 and rsi is not None and (rsi < 30 or rsi > 70):
        route = {"strategy":"meanrev","reason":f"range RSI={rsi:.1f}"}
    elif event or r_conf == 0.0:
        route = {"strategy":"llm_policy","reason":"event_or_low_confidence"}
    else:
        route = {"strategy":"llm_policy","reason":"default_to_llm_policy_for_learning"}

    state["route"] = route
    return state

@traceable(name="node:decide_momentum", run_type="chain")
def decide_momentum(state: State) -> State:
    f: TAFeatures = state["features"]
    size = 0.15 if (f.trend_score or 0) > 0.35 else 0.10
    state["decision"] = {"action":"BUY", "size":size, "reason":f"momentum trend_score={f.trend_score:.2f}"}
    return state

@traceable(name="node:decide_meanrev", run_type="chain")
def decide_meanrev(state: State) -> State:
    f: TAFeatures = state["features"]
    action = "BUY" if (f.RSI is not None and f.RSI < 30) else "SELL"
    state["decision"] = {"action":action, "size":0.10, "reason":f"meanrev RSI={f.RSI}"}
    return state

class LLMDecision(BaseModel):
    action: Literal["BUY","HOLD","SELL"]
    size: float = Field(ge=0.0, le=0.3)
    reason: str
    confidence: float
    stops: Optional[Dict[str, Optional[float]]] = None

@traceable(name="node:llm_decide_structured", run_type="chain")
def llm_decide_structured(state: State) -> State:
    schema = LLMDecision.model_json_schema()
    prompt = f"""
Return ONLY JSON matching this schema:
{schema}

Inputs:
- ticker: {state['ticker']}
- date: {state['date']}
- TA features: {state['features'].model_dump()}
- reddit: {state['reddit'].model_dump() if 'reddit' in state else {'count':0}}
"""
    decision = llm.with_structured_output(LLMDecision).invoke([{"role":"user","content":prompt}])
    state["decision"] = decision.model_dump()
    return state

@traceable(name="node:risk_gate", run_type="chain")
def risk_gate(state: State) -> State:
    dec = state["decision"]
    f: TAFeatures = state["features"]
    r: Optional[RedditOut] = state.get("reddit")
    violations = []

    ban = any(k in (r.topics if r else []) for k in ["halt","fraud","delist"])
    if ban and dec["action"] == "BUY":
        violations.append("ban_keyword_buy_blocked")
        dec = {"action":"HOLD","size":0.0,"reason":dec["reason"]+"; adjusted due to ban keywords"}

    if f.vol_score is not None and dec["size"] > 0.15 and f.vol_score > 0.5:
        violations.append("size_capped_by_vol")
        dec = {**dec, "size":0.15, "reason":dec["reason"]+f"; capped vol={f.vol_score:.2f}"}

    state["decision"] = dec
    state["risk"] = {"approved":True, "adjusted_decision":dec, "violations":violations}
    return state

@traceable(name="node:execute_order", run_type="chain")
def execute_order(state: State) -> State:
    dec = state["decision"]
    last_price = state["prices"].candles[-1].c if state["prices"].candles else 100.0
    if dec["action"] == "HOLD":
        exec_out = {"status":"filled","fill_price":None,"qty":0.0,"broker":"sim","order_id":None}
    else:
        qty = round(1000 * dec["size"] / max(1e-6, last_price), 4)
        slip = random.uniform(-0.02, 0.02)
        fill = last_price * (1 + slip/100)
        exec_out = {"status":"filled","fill_price":round(fill, 4),"qty":qty,"broker":"sim","order_id":f"SIM-{random.randint(10000,99999)}"}
    state["execution"] = exec_out
    return state

@traceable(name="node:post_trade_metrics", run_type="chain")
def post_trade_metrics(state: State) -> State:
    candles = state["prices"].candles
    dec = state["decision"]
    pnl = 0.0
    if dec["action"] in ("BUY","SELL") and len(candles) >= 2:
        p0 = candles[-2].c
        p1 = candles[-1].c
        ret = (p1 - p0) / max(1e-6, p0)
        pnl = (ret if dec["action"]=="BUY" else -ret) * (dec["size"]/0.3)
    state["metrics"] = {"pnl_sim": float(pnl), "latency_ms": 0, "rule_violations": len(state.get("risk", {}).get("violations", []))}
    return state

@traceable(name="node:report", run_type="chain")
def report(state: State) -> State:
    f: TAFeatures = state["features"]
    dec = state["decision"]
    reg = state["regime"]
    msg = (
        f"[{state['ticker']}] {dec['action']} size {dec['size']:.2f} via {state['route']['strategy']}; "
        f"regime={reg['regime']} (conf {reg['confidence']:.2f}); trend={f.trend_score:.2f}, vol={f.vol_score:.2f}; "
        f"pnl_sim={state['metrics']['pnl_sim']:+.4f}."
    )
    state["summary"] = msg
    return state

# === 5) Build graph & edges ===
graph_builder = StateGraph(State)
graph_builder.add_node("fetch_prices", fetch_prices)
graph_builder.add_node("features_talib", features_talib)
graph_builder.add_node("reddit_ingest", reddit_ingest)
graph_builder.add_node("finnhub_news", finnhub_news)
graph_builder.add_node("regime_detect", regime_detect)
graph_builder.add_node("strategy_router", strategy_router)
graph_builder.add_node("decide_momentum", decide_momentum)
graph_builder.add_node("decide_meanrev", decide_meanrev)
graph_builder.add_node("llm_decide_structured", llm_decide_structured)
graph_builder.add_node("risk_gate", risk_gate)
graph_builder.add_node("execute_order", execute_order)
graph_builder.add_node("post_trade_metrics", post_trade_metrics)
graph_builder.add_node("report", report)

graph_builder.add_edge(START, "fetch_prices")
graph_builder.add_edge("fetch_prices", "features_talib")
graph_builder.add_edge("features_talib", "reddit_ingest")
graph_builder.add_edge("reddit_ingest", "finnhub_news")
graph_builder.add_edge("finnhub_news", "regime_detect")
graph_builder.add_edge("regime_detect", "strategy_router")

def _route(state: State) -> str:
    return state["route"]["strategy"]

graph_builder.add_conditional_edges(
    "strategy_router",
    _route,
    {"momentum":"decide_momentum", "meanrev":"decide_meanrev", "llm_policy":"llm_decide_structured"}
)

graph_builder.add_edge("decide_momentum", "risk_gate")
graph_builder.add_edge("decide_meanrev", "risk_gate")
graph_builder.add_edge("llm_decide_structured", "risk_gate")
graph_builder.add_edge("risk_gate", "execute_order")
graph_builder.add_edge("execute_order", "post_trade_metrics")
graph_builder.add_edge("post_trade_metrics", "report")
graph_builder.add_edge("report", END)

graph = graph_builder.compile()

# === 6) Runner ===
@traceable(name="trade_run", run_type="chain")
def run_once(ticker: str, mode: Literal["paper","live"]="paper") -> State:
    init: State = {"ticker": ticker, "date": dt.date.today().isoformat(), "mode": mode}
    out: State = graph.invoke(init)
    return out

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    res = run_once(ticker)
    print(res["summary"])
