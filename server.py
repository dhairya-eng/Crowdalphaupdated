import os
import uuid
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from helpers import get_answer_from_question

from algorithmicTool import analyze_technicals_with_charts, fetch_bar
load_dotenv(find_dotenv())

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    userMessage: str
    chat_id: str = None

class BarsRequest(BaseModel):
    symbol: str = "AAPL"
    interval: str = "1d"      # e.g. "1m","5m","1d"
    lookback: str = "1y"      # e.g. "30d","1y","max"
    prepost: bool = True

class AnalyzeRequest(BarsRequest):
    horizon: int = 5       

CHART_DIR = os.path.join(os.getcwd(), "charts")
os.makedirs(CHART_DIR, exist_ok=True)
os.environ["MCP_TMP_DIR"] = CHART_DIR
app.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")

@app.get("/")
async def get_status():
    return "Server Running"

@app.post("/chat/text")
async def chat_text(request: ChatRequest):
    user_message = request.userMessage
    chat_id = request.chat_id or str(uuid.uuid4())
    chat_id_exists = bool(request.chat_id)

    answer = "Unable to get answers"
    chat_history = []
    if user_message:
        answer, chat_history = await get_answer_from_question(user_message, chat_id)

    response = {"reply": answer, "chat_history": chat_history}
    if not chat_id_exists:
        response["chat_id"] = chat_id

    return response

@app.post("/strategy/bars")
async def strategy_bars(req: BarsRequest):
    df = fetch_bar(req.symbol, req.interval, req.lookback, prepost=req.prepost)
    if df is None or df.empty:
        return {"symbol": req.symbol, "bars": []}
    # Normalize -> [{date, open, high, low, close, volume}]
    out: List[dict] = []
    dfr = df.reset_index()
    for _, row in dfr.iterrows():
        # ISO date string; for intraday keep full timestamp
        ts = row[dfr.columns[0]]
        if hasattr(ts, "isoformat"):
            date = ts.isoformat()
        else:
            date = str(ts)
        out.append({
            "date": date,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close":float(row["close"]),
            "volume": int(row["volume"]) if "volume" in dfr.columns else 0
        })
    return {"symbol": req.symbol, "bars": out}

# ---------- New: technical analysis + server-rendered charts ----------
@app.post("/strategy/analyze")
async def strategy_analyze(req: AnalyzeRequest):
    # ensure charts land in our mounted folder
    os.environ["MCP_TMP_DIR"] = CHART_DIR

    res = analyze_technicals_with_charts(
        symbol=req.symbol,
        interval=req.interval,
        lookback=req.lookback,
        horizon=req.horizon,
        prepost=req.prepost,
    )

    # Convert file:// URIs to server paths /charts/<file>.png for browser
    charts = []
    for c in res.get("charts", []):
        fname = os.path.basename(c.get("path", "")) or os.path.basename(c.get("uri",""))
        charts.append({
            "name": c.get("name"),
            "url": f"/charts/{fname}",
            "format": c.get("format", "png"),
        })

    res["charts"] = charts
    return res