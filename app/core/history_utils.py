from __future__ import annotations

import json
import math
import os
from typing import Any, List

from langchain_core.messages import HumanMessage

HISTORY_MAX_MESSAGES        = int(os.getenv("HISTORY_MAX_MESSAGES", "40"))
HISTORY_RELEVANCE_THRESHOLD = float(os.getenv("HISTORY_RELEVANCE_THRESHOLD", "0.40"))
HISTORY_ALWAYS_KEEP_LAST_N  = int(os.getenv("HISTORY_ALWAYS_KEEP_LAST_N", "0"))


def extract_text_content(content: Any) -> str:
    """LLM 응답 content (str | list | dict | None)에서 텍스트를 추출합니다.

    Gemini는 str, Claude API는 list[{'type':'text','text':...}] 형태로 반환하므로
    모든 노드에서 이 함수를 통해 텍스트를 추출해야 합니다.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def filter_history_by_relevance(
    history: List,
    user_input: str,
    input_embedding: List[float] | None = None  # ← 추가: 캐시된 임베딩
) -> List:
    if not history or not user_input.strip():
        return history

    pairs: List[tuple] = []
    i = 0
    while i < len(history) - 1:
        if isinstance(history[i], HumanMessage):
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return history

    always_n    = HISTORY_ALWAYS_KEEP_LAST_N
    keep_always = pairs[-always_n:] if always_n > 0 else []
    candidates  = pairs[:-always_n]  if always_n > 0 else pairs

    if not candidates:
        result: List = []
        for h, a in keep_always:
            result.extend([h, a])
        return result

    try:
        from app.core.config import get_embeddings

        # ✅ 캐시된 임베딩 사용, 없으면 계산
        if input_embedding:
            query_vec = input_embedding
        else:
            embeddings = get_embeddings()
            query_vec = embeddings.embed_query(user_input)

        # 히스토리 임베딩은 배치 (필수)
        embeddings = get_embeddings()
        human_texts = [(p[0].content or "") for p in candidates]
        msg_vecs = embeddings.embed_documents(human_texts)

        kept = [
            pair for pair, vec in zip(candidates, msg_vecs)
            if cosine(query_vec, vec) >= HISTORY_RELEVANCE_THRESHOLD
        ]
    except Exception as e:
        print(f"DEBUG: [HistoryFilter] 임베딩 실패, 전체 히스토리 사용: {e}")
        kept = candidates

    result = []
    for h, a in (kept + keep_always):
        result.extend([h, a])
    return result
