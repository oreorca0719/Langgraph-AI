"""
Chroma + BM25 Hybrid Retriever — Retriever Protocol 구현체.

기존 `app.graph.nodes.knowledge_search`의 `_search_hybrid`를 Protocol에 wrap.
v1과 동일한 검색 로직 + score 정규화 + Document 변환.
"""
from __future__ import annotations

import asyncio
import re
import threading
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from app.core.config import (
    get_embeddings,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RETRIEVAL_TOP_K,
)
from app.graph_v2.retrievers.base import Document, Retriever


_HYBRID_FETCH_MULTIPLIER = 4
_RRF_K = 60


class ChromaHybridRetriever:
    """Chroma 시맨틱 + BM25 RRF hybrid retriever.

    - name: "chroma_hybrid"
    - description: 사내 문서 검색 (사내 PPT, PDF, DOCX, TXT)
    """

    name: str = "chroma_hybrid"
    description: str = (
        "사내 지식 베이스 검색. PPT 슬라이드, PDF 페이지, DOCX/TXT 문서를 "
        "시맨틱 + 키워드 hybrid (RRF)로 검색. 사내 정책·매뉴얼·회의록·기획 산출물·강사 자료 등."
    )

    def __init__(self) -> None:
        self._chroma: Optional[Chroma] = None
        self._chroma_lock = threading.Lock()
        self._bm25 = None
        self._bm25_docs: list[LCDocument] = []
        self._bm25_lock = threading.Lock()

    # ─── public Retriever Protocol ───────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
        fetch_k = top_k * _HYBRID_FETCH_MULTIPLIER
        sem = self._semantic_search(query, fetch_k)
        bm = self._bm25_search(query, fetch_k)
        merged = self._rrf(sem, bm, top_k)
        return [self._to_document(d, score=s) for d, s in merged]

    async def aretrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
        # Chroma·BM25 모두 sync 라이브러리 — thread pool로 비동기화
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retrieve, query, top_k)

    # ─── 내부 ──────────────────────────────────────────

    def _get_chroma(self) -> Chroma:
        if self._chroma is None:
            with self._chroma_lock:
                if self._chroma is None:
                    self._chroma = Chroma(
                        persist_directory=CHROMA_DB_PATH,
                        embedding_function=get_embeddings(),
                        collection_name=CHROMA_COLLECTION,
                    )
        return self._chroma

    def _get_bm25(self):
        if self._bm25 is not None:
            return self._bm25, self._bm25_docs
        with self._bm25_lock:
            if self._bm25 is not None:
                return self._bm25, self._bm25_docs
            from rank_bm25 import BM25Okapi  # type: ignore
            collection = self._get_chroma()._collection
            res = collection.get(include=["documents", "metadatas"])
            raw_docs = res.get("documents") or []
            raw_metas = res.get("metadatas") or []
            docs = [
                LCDocument(page_content=t, metadata=m)
                for t, m in zip(raw_docs, raw_metas)
                if t
            ]
            if not docs:
                return None, []
            tokenized = [self._tokenize(d.page_content) for d in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_docs = docs
        return self._bm25, self._bm25_docs

    def invalidate_bm25_cache(self) -> None:
        """문서 재인제스트 후 BM25 인덱스 초기화."""
        with self._bm25_lock:
            self._bm25 = None
            self._bm25_docs = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())

    def _semantic_search(self, query: str, k: int) -> list[tuple[LCDocument, float]]:
        # Chroma의 similarity_search_with_score는 distance 반환 (작을수록 유사)
        # → 1 / (1 + distance) 로 0~1 정규화
        try:
            results = self._get_chroma().similarity_search_with_score(query, k=k)
            return [(d, 1.0 / (1.0 + dist)) for d, dist in results]
        except Exception:
            # Fallback — score 정보 없으면 균일 점수
            docs = self._get_chroma().similarity_search(query, k=k)
            return [(d, 0.5) for d in docs]

    def _bm25_search(self, query: str, k: int) -> list[tuple[LCDocument, float]]:
        bm25, docs = self._get_bm25()
        if bm25 is None:
            return []
        scores = bm25.get_scores(self._tokenize(query))
        if not len(scores):
            return []
        # BM25 score 정규화 (max 기준)
        max_s = max(scores) if max(scores) > 0 else 1.0
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(docs[i], scores[i] / max_s) for i in ranked if scores[i] > 0]

    def _rrf(
        self,
        sem: list[tuple[LCDocument, float]],
        bm: list[tuple[LCDocument, float]],
        k: int,
    ) -> list[tuple[LCDocument, float]]:
        """Reciprocal Rank Fusion. 두 랭킹의 rank로 결합. 최종 score는 정규화된 normalized score 평균."""
        scores: dict[str, float] = {}
        norm_scores: dict[str, list[float]] = {}
        doc_map: dict[str, LCDocument] = {}

        for rank, (doc, ns) in enumerate(sem):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            norm_scores.setdefault(key, []).append(ns)
            doc_map[key] = doc
        for rank, (doc, ns) in enumerate(bm):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            norm_scores.setdefault(key, []).append(ns)
            doc_map.setdefault(key, doc)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [
            (doc_map[k_], sum(norm_scores[k_]) / len(norm_scores[k_]))
            for k_ in sorted_keys
        ]

    def _to_document(self, lc: LCDocument, score: float) -> Document:
        md = dict(lc.metadata or {})
        return Document(
            content=lc.page_content or "",
            source=md.get("display_source") or md.get("path") or md.get("title") or "",
            score=max(0.0, min(1.0, score)),  # clamp 0~1
            metadata={
                "title": md.get("title", ""),
                "doc_id": md.get("display_source", md.get("title", "")),
                "location": (
                    f"Page {md['page_number']}" if md.get("page_number")
                    else f"Chunk {md.get('chunk_index', 0)}"
                ),
                "chunk_index": md.get("chunk_index", 0),
                "total_chunks": md.get("total_chunks", 0),
                **{k: v for k, v in md.items()
                   if k not in {"title", "display_source", "page_number", "chunk_index", "total_chunks", "path"}},
            },
        )
