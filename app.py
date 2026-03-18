from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

@app.get("/")
def root():
    return {"message": "Agmentic API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest):
    try:
        from rag_engine import ask_question
        answer = ask_question(payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))