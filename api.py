import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import (
    get_or_create_session,
    get_last_session,
    process_question,
    save_exchange,
    get_session_history,
    close_resources,
)
from memory import save_feedback, is_neo4j_available

LOG_FORMAT = "%(asctime)s [API] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API AutoDrive démarrée sur le port 8000")
    yield
    close_resources()
    logger.info("API AutoDrive arrêtée")


app = FastAPI(
    title="AutoDrive RAG API",
    description="API pour le système RAG multi-agents AutoDrive",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question de l'utilisateur", min_length=1)
    session_id: Optional[str] = Field(None, description="ID de session optionnel")


class ChatResponse(BaseModel):
    reponse: str
    sources: List[str]
    session_id: str
    agent_utilise: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    reponse: str
    utile: bool


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]


class HealthResponse(BaseModel):
    status: str
    neo4j: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    neo4j_status = "connected" if is_neo4j_available() else "disconnected"
    return HealthResponse(status="ok", neo4j=neo4j_status)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Requête chat reçue: question='{request.question[:50]}...', session_id={request.session_id}")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    if request.session_id:
        session_id, history = get_or_create_session(request.session_id)
    else:
        session_id, history = get_last_session()
        if not session_id:
            session_id, history = get_or_create_session()

    result = process_question(request.question, session_id, history)

    if session_id != "no-memory":
        save_exchange(session_id, request.question, result["reponse"])

    logger.info(f"Réponse générée: agent={result['agent_utilise']}, sources={len(result['sources'])}")

    return ChatResponse(
        reponse=result["reponse"],
        sources=result["sources"],
        session_id=result["session_id"],
        agent_utilise=result["agent_utilise"],
    )


@app.get("/sessions/{session_id}/historique", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info(f"Historique demandé pour session: {session_id}")

    if session_id == "no-memory":
        raise HTTPException(status_code=404, detail="Session sans mémoire non trouvée")

    messages = get_session_history(session_id)

    return HistoryResponse(session_id=session_id, messages=messages)


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    logger.info(f"Feedback reçu: session={request.session_id}, utile={request.utile}")

    if not is_neo4j_available():
        raise HTTPException(status_code=503, detail="Neo4j non disponible")

    success = save_feedback(
        session_id=request.session_id,
        question=request.question,
        reponse=request.reponse,
        utile=request.utile,
    )

    if success:
        return FeedbackResponse(success=True, message="Feedback enregistré")
    else:
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)