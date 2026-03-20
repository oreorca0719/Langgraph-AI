"""
Gemini/Google GenAI 설정 유틸.

운영(App Runner)에서는 환경변수로 키를 주입해야 합니다.
- GOOGLE_API_KEY (langchain_google_genai 기본)
- GEMINI_API_KEY (편의: 있으면 GOOGLE_API_KEY로 매핑)

주의: 키를 코드에 하드코딩하지 마세요.
"""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def has_gemini_api_key() -> bool:
    """LLM 호출 가능 여부(키 존재)를 반환합니다."""
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _ensure_google_api_key_env() -> None:
    """GEMINI_API_KEY만 주어진 경우, 라이브러리가 읽는 GOOGLE_API_KEY로 매핑합니다."""
    if os.getenv("GOOGLE_API_KEY"):
        return
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    _ensure_google_api_key_env()
    model = model_name or os.getenv("LLM_MODEL", "gemini-3-flash-preview")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


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
