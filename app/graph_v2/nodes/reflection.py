"""
Reflection node — 답변 자기검증 (FR-403, FR-404, FR-405).

검증 항목:
  - groundedness: 답변이 검색 근거에서 도출됐는가
  - relevance: 답변이 원 질문에 답하는가
  - hallucination_risk: 환각 의심 여부
  - 추가 (Phase D 학습 — 명확한 trigger만):
    * 수치/entity in chunks 검증 (regex 기반, 빠름)
    * no_info_misclaim 검출

Reflection 실패 + replan_iterations < 1 → router로 회귀.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_llm
from app.graph_v2.states.state import GraphState, VerificationResult
from app.graph_v2.retrievers.base import Document


# ────────────────────────────────────────────────────────────
# Regex 기반 가벼운 검증 (LLM 호출 없음)
# ────────────────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}|\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:명|억|만원|만\s*원|억\s*원|시간|분|일|차|곳|개|단계|자|회|건|개월|년|％|%|원))?)"
)


def _extract_numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(1)) for m in _NUMBER_PATTERN.finditer(text or "")}


def _check_facts_in_chunks(answer: str, docs: list[Document]) -> list[str]:
    """답변의 수치가 chunks에 존재하는지 정규식 검증. 부재 시 issue."""
    if not docs or not answer:
        return []
    chunks_norm = re.sub(r"\s+", "", "\n".join(d.content or "" for d in docs))
    issues = []
    for num in _extract_numbers(answer):
        if num not in chunks_norm:
            issues.append(f"수치 '{num}' 검색 결과에 없음")
            if len(issues) >= 5:
                break
    return issues


_NO_INFO_PHRASES = [
    "찾을 수 없습니다", "확인할 수 없습니다", "확인되지 않", "포함되어 있지 않",
    "정보가 없", "찾을 수 없다",
]


def _check_no_info_misclaim(answer: str, docs: list[Document]) -> bool:
    """답변이 '정보 없음'인데 chunks가 충분히 있으면 의심."""
    if len(docs) < 3:
        return False
    short = len(answer) < 200
    has_no_info = any(p in answer for p in _NO_INFO_PHRASES)
    return short and has_no_info


# ────────────────────────────────────────────────────────────
# LLM-as-judge 검증 (groundedness, relevance, hallucination)
# ────────────────────────────────────────────────────────────

_REFLECTION_PROMPT = """당신은 사내 RAG 시스템의 답변 자기검증 평가자입니다.

【평가 항목】 (각 0.0~1.0)
1. groundedness: 답변이 검색 결과에서 도출됐는가 (0=완전히 추측, 1=모두 검색 결과 기반)
2. relevance: 답변이 원 질문에 답하는가 (0=무관, 1=정확히 답)
3. hallucination_risk: 검색 결과에 없는 사실이 답변에 있는가 (0=없음, 1=명백 환각)

【출력】 JSON만:
{
  "groundedness": 0.0~1.0,
  "relevance": 0.0~1.0,
  "hallucination_risk": 0.0~1.0,
  "passed": true/false,
  "reason": "한 줄 요약"
}

passed 기준: groundedness >= 0.7 AND relevance >= 0.7 AND hallucination_risk <= 0.3
"""


def _call_reflection_llm(question: str, answer: str, docs: list[Document]) -> dict:
    if not answer or not docs:
        return {}
    chunks_text = "\n".join(f"[{i}] {(d.content or '')[:400]}" for i, d in enumerate(docs[:5]))
    user_content = (
        f"[질문]\n{question}\n\n"
        f"[검색 결과 chunks]\n{chunks_text}\n\n"
        f"[시스템 답변]\n{answer[:1500]}"
    )
    try:
        resp = get_llm().invoke([
            SystemMessage(content=_REFLECTION_PROMPT),
            HumanMessage(content=user_content),
        ])
        raw = resp.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        raw = str(raw).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[REFLECTION] LLM call failed (non-fatal): {e}")
        return {}


def reflection_node(state: GraphState) -> dict:
    """답변 검증."""
    answer = state.answer or ""
    docs: list[Document] = state.retrieved_docs or []

    # Regex-based 빠른 검증
    fact_issues = _check_facts_in_chunks(answer, docs)
    no_info_misclaim = _check_no_info_misclaim(answer, docs)

    # LLM-as-judge (chunks가 있을 때만)
    if docs and answer and not answer.startswith("관련 사내 문서를 찾을 수 없"):
        judge = _call_reflection_llm(state.input_data, answer, docs)
        groundedness = float(judge.get("groundedness", 1.0))
        relevance = float(judge.get("relevance", 1.0))
        hallucination_risk = float(judge.get("hallucination_risk", 0.0))
        llm_passed = bool(judge.get("passed", True))
        llm_calls_added = 1
    else:
        # docs 없거나 명시적 "정보 없음" 응답이면 LLM 호출 생략
        groundedness, relevance, hallucination_risk, llm_passed = 1.0, 1.0, 0.0, True
        llm_calls_added = 0

    # 종합 passed
    passed = (
        llm_passed
        and not no_info_misclaim
        and len(fact_issues) <= 2  # 수치 issue 최대 2개까지 관용
    )

    verif = VerificationResult(
        groundedness=groundedness,
        relevance=relevance,
        hallucination_risk=hallucination_risk,
        fact_issues=fact_issues,
        no_info_misclaim=no_info_misclaim,
        passed=passed,
    )

    label = (
        f"reflect:pass(g={groundedness:.2f},r={relevance:.2f},h={hallucination_risk:.2f})"
        if passed else
        f"reflect:fail(facts={len(fact_issues)},miscalim={no_info_misclaim},g={groundedness:.2f})"
    )

    return {
        "verification": verif,
        "llm_call_count": state.llm_call_count + llm_calls_added,
        "decision_path": [label],
    }


def route_after_reflection(state: GraphState) -> str:
    if not state.verification:
        return "end"
    if state.verification.needs_replan() and state.replan_iterations < 1:
        return "replan"
    return "end"
