import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SYSTEM_PROMPT = """
You are event Network Intelligence assistant.

Act as a fast, executive decision-support system — not a chatbot.

Your job is to help leadership quickly identify:
• who to invite
• who to connect
• where revenue opportunities exist
• which relationships matter most

SPEED IS CRITICAL.
Always provide the answer immediately.

RESPONSE STYLE:
1. Start with a direct answer
2. Then give short supporting reasoning
3. Then list the most relevant names / entities / opportunities
4. If useful, mention which type of file the answer came from
5. If the answer is not in the provided DATA, say:
Not found in provided data.

IMPORTANT RULES:
- Use ONLY the provided DATA
- Do NOT use outside knowledge
- Do NOT invent facts
- If something is missing, clearly say it is missing
"""

MODEL_NAME = "llama-3.1-8b-instant"

_vectordb = None
_client = None


def get_groq_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        _client = Groq(api_key=api_key)
    return _client


def load_markdown_file(file_path: Path) -> Document:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "file_type": "markdown",
            "file_name": file_path.name,
            "folder": file_path.parent.name,
        },
    )


def load_json_file(file_path: Path) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            docs.append(
                Document(
                    page_content=json.dumps(item, ensure_ascii=False, indent=2),
                    metadata={
                        "source": str(file_path),
                        "file_type": "json",
                        "file_name": file_path.name,
                        "folder": file_path.parent.name,
                        "row": i,
                    },
                )
            )
    elif isinstance(data, dict):
        docs.append(
            Document(
                page_content=json.dumps(data, ensure_ascii=False, indent=2),
                metadata={
                    "source": str(file_path),
                    "file_type": "json",
                    "file_name": file_path.name,
                    "folder": file_path.parent.name,
                },
            )
        )

    return docs


def load_csv_file(file_path: Path) -> list[Document]:
    df = pd.read_csv(file_path)
    docs = []

    for i, row in df.iterrows():
        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(
            Document(
                page_content=row_text,
                metadata={
                    "source": str(file_path),
                    "file_type": "csv",
                    "file_name": file_path.name,
                    "folder": file_path.parent.name,
                    "row": int(i),
                },
            )
        )

    return docs


def load_documents(data_dir: Path) -> list[Document]:
    docs = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".md":
                docs.append(load_markdown_file(file_path))
            elif suffix == ".csv":
                docs.extend(load_csv_file(file_path))
            elif suffix == ".json":
                docs.extend(load_json_file(file_path))
        except Exception as e:
            print(f"Skipping {file_path} because of error: {e}")

    return docs


def build_vector_db(docs: list[Document]):
    # import های سنگین فقط اینجا
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(split_docs, embeddings)
    return vectordb


def get_vectordb():
    global _vectordb

    if _vectordb is None:
        print("Initializing vector DB...")
        docs = load_documents(DATA_DIR)
        print(f"Loaded {len(docs)} raw documents/rows from {DATA_DIR}")
        _vectordb = build_vector_db(docs)
        print("Vector DB initialized.")

    return _vectordb


def ask_question(question: str, k: int = 6) -> str:
    vectordb = get_vectordb()
    client = get_groq_client()

    docs = vectordb.similarity_search(question, k=k)

    context_parts = []
    for d in docs:
        source_info = (
            f"[SOURCE: {d.metadata.get('file_name', 'unknown')} | "
            f"FOLDER: {d.metadata.get('folder', 'unknown')} | "
            f"TYPE: {d.metadata.get('file_type', 'unknown')}]"
        )
        context_parts.append(f"{source_info}\n{d.page_content}")

    context = "\n\n".join(context_parts)

    final_prompt = f"""
{SYSTEM_PROMPT}

DATA:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content