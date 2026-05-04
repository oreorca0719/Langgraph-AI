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

【chunk 입력 필드 안내】
각 chunk는 본문 전체와 함께 다음 metadata를 포함합니다 (종합 판단에 활용):
- source / location / section: 출처와 위치
- doc_topic: 인제스트 시 분류된 문서 주제 (예: branding, history, vendor_report)
- retriever_score (0~1): retriever의 매칭 신뢰도 (높을수록 강함)
- rank: 검색 결과 내 순위 (1=최상위)
- entities: 정규식·키워드로 추출된 chunk의 핵심 entity 목록

질문 유형 (`[질문 유형: ...]`)도 참고: exact_phrase는 정확 문구, list_n은 N개 항목, numerical은 수치 회수 등.

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
        md = d.metadata or {}
        loc = md.get("location", "")
        section = md.get("section_path", "") or md.get("page_unit_title", "")
        doc_topic = md.get("doc_topic", "")
        entities = md.get("entities", "")
        score = d.score
        rank = i + 1
        header_parts = [f"source={d.source}"]
        if loc:
            header_parts.append(f"location={loc}")
        if section:
            header_parts.append(f"section={section}")
        if doc_topic:
            header_parts.append(f"doc_topic={doc_topic}")
        header_parts.append(f"retriever_score={score:.2f}")
        header_parts.append(f"rank={rank}")
        header = ", ".join(header_parts)

        body = d.content or ""
        ent_line = f"    [entities: {entities}]\n" if entities else ""
        out.append(
            f"[{i}] ({header})\n"
            f"{ent_line}"
            f"    [본문]\n"
            f"    {body}"
        )
    return "\n\n".join(out)


def _call_grader_llm(query: str, docs: list[Document], question_type: str) -> dict:
    if not docs:
        return {"labels": [], "premise_coverage": "none"}
    try:
        user_content = (
            f"[질문 유형: {question_type or 'reasoning'}]\n"
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
    """검색 결과를 LLM으로 라벨링 + relevant·partial만 유지.

    Rescue 로직은 제거됨 — 모두 irrelevant이면 빈 kept를 그대로 반환하여
    route_after_grade가 rewrite를 트리거하게 함 (3회 한도까지).
    """
    docs: list[Document] = state.retrieved_docs or []
    if not docs:
        return {
            "decision_path": ["grade:no_docs"],
        }

    parsed = _call_grader_llm(state.input_data, docs, state.question_type or "reasoning")
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

    if not kept:
        if state.retrieval_iterations >= 3:
            # 3회 rewrite 다 소진 후에도 못 찾음 — 시스템 회복 실패 (로그 분리용 라벨)
            path_label = (
                f"grade:exhausted_after_3rewrites(0rel/{len(docs)}total,cov={coverage})"
            )
        else:
            path_label = f"grade:all_irrelevant(0rel/{len(docs)}total,cov={coverage})"
    else:
        path_label = (
            f"grade:pass({relevant_count}rel/{len(kept)}kept/{len(docs)}total,cov={coverage})"
        )

    return {
        "retrieved_docs": kept,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [path_label],
    }


def route_after_grade(state: GraphState) -> str:
    """grade 통과 여부 + retry 한도.

    - kept = []  && iters < 3  → rewrite
    - kept = []  && iters >= 3 → generate (no_docs path, "찾을 수 없습니다")
    - kept = [.] → generate
    """
    docs = state.retrieved_docs or []
    iters = state.retrieval_iterations
    if not docs:
        if iters < 3:
            return "rewrite"
        return "generate"  # 한도 도달
    return "generate"
