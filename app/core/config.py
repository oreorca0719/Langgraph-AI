"""
Kaiper AI — 중앙 설정 모듈

운영(App Runner)에서는 환경변수로 값을 주입합니다.
여기서 정의된 기본값은 로컬 개발 환경용 fallback입니다.
"""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────

def has_gemini_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _ensure_google_api_key_env() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))


def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    _ensure_google_api_key_env()
    model = model_name or os.getenv("LLM_MODEL", "gemini-3-flash-preview")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

class GeminiRAGEmbeddings(GoogleGenerativeAIEmbeddings):
    """LangChain Google GenAI 임베딩 래퍼 (RAG용 task_type 분기)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.task_type = "retrieval_document"
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.task_type = "retrieval_query"
        return super().embed_query(text)


def get_embeddings() -> GeminiRAGEmbeddings:
    _ensure_google_api_key_env()
    return GeminiRAGEmbeddings(model="gemini-embedding-001")


# ──────────────────────────────────────────────
# Chroma / RAG
# ──────────────────────────────────────────────

CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH",    "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_knowledge")

RETRIEVAL_MIN_RELEVANCE = float(os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.3"))
RETRIEVAL_MAX_DISTANCE  = float(os.getenv("RETRIEVAL_MAX_DISTANCE",  "0.75"))
RETRIEVAL_TOP_K         = int(os.getenv("RETRIEVAL_TOP_K", "5"))


# ──────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────

KNOWLEDGE_DIR        = os.getenv("KNOWLEDGE_DIR",        "./knowledge_data")
AUTO_INGEST          = os.getenv("AUTO_INGEST",          "1")
S3_KNOWLEDGE_BUCKET  = os.getenv("S3_KNOWLEDGE_BUCKET",  "")
S3_KNOWLEDGE_PREFIX  = os.getenv("S3_KNOWLEDGE_PREFIX",  "knowledge_data/")
INGEST_CHUNK_MAX_CHARS = int(os.getenv("INGEST_CHUNK_MAX_CHARS", "1200"))
INGEST_CHUNK_OVERLAP   = int(os.getenv("INGEST_CHUNK_OVERLAP",   "200"))


# ──────────────────────────────────────────────
# Semantic Router
# ──────────────────────────────────────────────

ROUTER_TOP1_MIN  = float(os.getenv("ROUTER_TOP1_MIN",  "0.62"))
ROUTER_MARGIN_MIN = float(os.getenv("ROUTER_MARGIN_MIN", "0.08"))


# ──────────────────────────────────────────────
# LLM Intent Fallback
# ──────────────────────────────────────────────

LLM_FALLBACK_ENABLED     = os.getenv("LLM_FALLBACK_ENABLED", "1")
LLM_FALLBACK_TIMEOUT_SEC = float(os.getenv("LLM_FALLBACK_TIMEOUT_SEC", "8"))
LLM_FALLBACK_CONFIDENCE_MIN = float(os.getenv("LLM_FALLBACK_CONFIDENCE_MIN", "0.5"))
