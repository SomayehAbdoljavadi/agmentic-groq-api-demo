from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import init_rag, ask_question

app = FastAPI(title="Agmentic Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    init_rag()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest):
    answer = ask_question(payload.question)
    return {"answer": answer}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

