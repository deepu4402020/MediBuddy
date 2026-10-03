"""FastAPI server: POST /chat, static public/."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.graph import build_graph, run_turn

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SOPS_DIR = os.getenv("SOPS_DIR", str(ROOT / "sops"))
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="SOP Weather Safety Chatbot")

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
        llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0, api_key=api_key)
        _graph = build_graph(llm, SOPS_DIR)
    return _graph


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    trace: dict


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if len(body.message) > 500:
        raise HTTPException(status_code=400, detail="message exceeds 500 characters")

    result = await run_turn(get_graph(), body.sessionId, body.message.strip())
    reply = result.get("reply") or ""
    trace = result.get("trace") or {}
    return ChatResponse(reply=reply, trace=trace)


public_dir = ROOT / "public"
if public_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="static")


def main():
    import uvicorn

    uvicorn.run("src.server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
