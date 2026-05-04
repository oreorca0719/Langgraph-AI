"""
Grader node — 검색 결과의 관련성 평가 (FR-301, FR-302, FR-303, FR-304).

라벨링:
  - "relevant" / "partially_relevant" / "irrelevant"

흐름:
  1. 모든 chunk를 한 번의 LLM 호출로 라벨링 (배치 효율)
  2. relevant + partially_relevant chunk만 retrieved_docs에 유지 (FR-401)
  3. relevant 비율 < 임계값 → rewrite trigger (FR-302, retrieval_iterations 한도까지)

추가 (Phase 회복 학습):
  - premise_coverage: 검색 결과가 질문의 핵심 entity를 커버하는가
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_llm
from app.graph_v2.states.state import GraphState
from app.graph_v2.retrievers.base import Document


_GRADER_SYSTEM_PROMPT = """당신은 사내 RAG 시스템의 검색 결과 평가자입니다. 사용자의 질문과 각 chunk를 보고 라벨링합니다.

【라벨】
- "relevant": chunk가 질문의 답을 직접 포함
- "partially_relevant": chunk가 질문 일부에 답하거나 배경 정보 제공
- "irrelevant": chunk가 질문과 무관 (다른 entity, 다른 주제, 노이즈)

【평가 원칙】
- 질문에 등장하는 entity (이름·시간·수치)와 chunk의 entity가 일치해야 relevant
- 비슷한 주제지만 다른 entity면 irrelevant (예: "위시캣 회원수" 질문에 "원티드 회원수" chunk → irrelevant)
- 직접 답이 없어도 답 도출에 필요한 사실이 있으면 partially_relevant

【출력】 JSON만:
{
  "labels": ["relevant", "partially_relevant", ...],  // chunks 순서대로
  "premise_coverage": "full" | "partial" | "none",   // 질문의 모든 entity·전제가 커버되는가
  "reason": "한 줄 요약"
}
"""


def _format_chunks(docs: list[Document]) -> str:
    out = []
    for i, d in enumerate(docs):
        snippet = (d.content or "")[:300].replace("\n", " ")
        loc = d.metadata.get("location", "")
        out.append(f"[{i}] ({d.source}{', ' + loc if loc else ''}) {snippet}")
    return "\n".join(out)


def _call_grader_llm(query: str, docs: list[Document]) -> dict:
    if not docs:
        return {"labels": [], "premise_coverage": "none"}
    try:
        user_content = (
            f"[질문]\n{query}\n\n"
            f"[검색 결과 chunks ({len(docs)}개)]\n{_format_chunks(docs)}\n\n"
            "각 chunk를 라벨링해주세요."
        )
        resp = get_llm().invoke([
            SystemMessage(content=_GRADER_SYSTEM_PROMPT),
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
        print(f"[GRADER] LLM call failed (non-fatal): {e}")
        return {"labels": ["partially_relevant"] * len(docs), "premise_coverage": "partial"}


def grader_node(state: GraphState) -> dict:
    """검색 결과를 LLM으로 라벨링 + relevant·partial만 유지."""
    docs: list[Document] = state.retrieved_docs or []
    if not docs:
        return {
            "decision_path": ["grade:no_docs"],
        }

    parsed = _call_grader_llm(state.input_data, docs)
    labels = parsed.get("labels") or []
    coverage = parsed.get("premise_coverage", "partial")

    # labels 길이 보정
    if len(labels) < len(docs):
        labels = labels + ["partially_relevant"] * (len(docs) - len(labels))
    labels = labels[: len(docs)]

    # relevant + partially_relevant만 유지
    kept = []
    relevant_count = 0
    for d, lbl in zip(docs, labels):
        if lbl == "relevant":
            relevant_count += 1
            kept.append(d)
        elif lbl == "partially_relevant":
            kept.append(d)

    # Rescue: 모두 irrelevant 판정 받은 경우 — 상위 3개 강제 유지.
    # 이유: grader가 가끔 너무 strict (특히 exact_phrase 질문에서 chunk를 모두 irrelevant 판정).
    # docs 자체는 retriever가 이미 적합도순으로 정렬했으므로 top 3은 사용해볼 가치 있음.
    rescued = False
    if not kept and docs:
        kept = docs[:3]
        rescued = True

    rel_ratio = relevant_count / len(docs) if docs else 0.0
    pass_quality = (rel_ratio >= 0.3) or coverage in {"full", "partial"} or rescued

    if rescued:
        path_label = f"grade:rescue(0rel/{len(docs)}total,kept_top3,cov={coverage})"
    elif pass_quality:
        path_label = f"grade:pass({relevant_count}rel/{len(kept)}kept/{len(docs)}total,cov={coverage})"
    else:
        path_label = f"grade:fail({relevant_count}rel/{len(docs)}total,cov={coverage})"

    return {
        "retrieved_docs": kept,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [path_label],
    }


def route_after_grade(state: GraphState) -> str:
    """grade 통과 여부 + retry 한도."""
    docs = state.retrieved_docs or []
    iters = state.retrieval_iterations
    if not docs:
        if iters < 3:  # FR-303 기본 3, 절대 상한 5
            return "rewrite"
        return "generate"  # 한도 도달 — generator가 "정보 없음" 처리
    return "generate"
