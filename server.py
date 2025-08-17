import os
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from helpers import get_answer_from_question

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

