"""
chunk_question_types Classifier — chunk별 답할 수 있는 질문 유형 분류.

호출 빈도: chunk당 1회 (인제스트 시점)
용도: retriever가 query의 question_type과 chunk가 답할 수 있는 유형을 매칭하여 score boost

라벨 enum (router의 question_type과 동일):
  exact_phrase / numerical / list_n / fill_blank / reasoning / comparison

Multi-label: 한 chunk가 여러 유형에 부합할 수 있음.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from app.core.config import get_llm


VALID_QTYPES = {"exact_phrase", "numerical", "list_n", "fill_blank", "reasoning", "comparison"}


_QTYPE_PROMPT = """다음 chunk를 보고 답할 수 있는 질문 유형을 모두 골라 JSON 배열로 반환.

라벨:
- exact_phrase: 정확한 문구·명사·인용 회수 가능 (예: 슬로건, 명칭, 약어 풀네임)
- numerical: 수치·시간·일자 회수 가능 (예: 회원 수, 매출액, 9시~12시)
- list_n: N개 항목 나열 답변 가능 (예: 4단계, 3가지 요소, 5개 항목)
- fill_blank: 빈칸 채우기에 쓸 핵심 단어가 있음
- reasoning: 추론·이유·의미·설명 답변 가능
- comparison: 두 대상의 차이·비교 답변 가능

판단 원칙:
- 부합하는 것만 (보수적). 애매하면 제외.
- 표 markdown chunk는 보통 numerical + list_n + comparison 가능.
- 단순 본문은 reasoning + exact_phrase 중심.

【출력】 JSON 배열만. 다른 텍스트 금지.
예: ["numerical", "list_n", "comparison"]

[chunk 내용]
{text}
"""


def classify_chunk_qtypes(chunk_text: str, model_name: str | None = None) -> list[str]:
    """chunk 텍스트 → 부합 question_types list (multi-label).

    실패 시 빈 list 반환 (graceful — boost 없음).
    """
    if not chunk_text or not chunk_text.strip():
        return []
    try:
        resp = get_llm(model_name).invoke([
            HumanMessage(content=_QTYPE_PROMPT.format(text=chunk_text[:2000]))
        ])
        raw = resp.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        raw = str(raw).strip()
        # Fenced code 정리
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        # JSON 추출 시도 (배열만 잘라내기)
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        # 검증: VALID_QTYPES에 있는 것만
        result = [t for t in parsed if isinstance(t, str) and t in VALID_QTYPES]
        return result
    except Exception as e:
        print(f"[QTYPE] classify failed (non-fatal): {e}")
        return []
