"""
Generator node — 검색 결과 기반 답변 생성 + Citation 부착 (FR-401, FR-402).

핵심 설계 (Phase B 학습 회복):
  - question_type별 프롬프트 분기 — list_n 누락 방지, verbatim 강제 분기 적용
  - 모든 사실 진술에 [N] citation 강제 (FR-402)
  - relevant + partial 문서만 컨텍스트로 사용 (FR-401)
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_llm
from app.graph_v2.states.state import Citation, GraphState
from app.graph_v2.retrievers.base import Document


# ────────────────────────────────────────────────────────────
# question_type별 프롬프트
# ────────────────────────────────────────────────────────────

_COMMON_PRINCIPLES = """【핵심 원칙】
1. 수치·고유명사·인용 문구는 검색 결과 그대로 (verbatim, paraphrase 금지).
2. 질문이 묻는 것만 답하세요. 관련 배경·추가 설명은 절대 추가 금지.
3. 모든 사실 진술 뒤에 [N] citation 부착. 부착할 수 없는 진술은 작성 금지.
4. 답변 작성 전 내부 확인:
   - 답에 포함될 모든 단어가 검색 결과의 어느 chunk에 있는가?
   - 질문의 entity와 chunk entity가 일치하는가?
5. 검색 결과에 답이 없으면 "관련 사내 문서를 찾을 수 없습니다."
6. 답변 시작에 '~는 다음과 같습니다:' 같은 서두 금지 — 바로 답.
"""

_PROMPT_BY_TYPE = {
    "exact_phrase": _COMMON_PRINCIPLES + """
【exact_phrase 응답 규칙】
- 검색 결과의 정확한 문구만. 한 줄로.
- 예: "대한민국 최초 금융 IT 매칭 플랫폼 [1]"
""",

    "numerical": _COMMON_PRINCIPLES + """
【numerical 응답 규칙】
- 수치·시간만 답. 단위 포함 (예: 41만+, 09:00~12:30).
- 한 줄로. 부수 설명 금지.
- 예: "09:00~12:30 [1]"
""",

    "list_n": _COMMON_PRINCIPLES + """
【list_n 응답 규칙】 (★ 가장 주의 ★)
- 질문이 'N가지/N개/N단계'로 명시한 경우 정확히 N개 항목.
- 각 항목을 새 줄로 분리, 검색 결과 표현 그대로.
- 누락 절대 금지 — 답하기 전 모든 항목을 검색 결과에서 찾았는지 확인.
- **다른 슬라이드/페이지의 다른 주제 항목 절대 추가 금지.** 같은 chunk 안에 있는 동일 주제 항목만 사용.
- 예: "• 차별성 [1]\\n• 베네핏 [1]\\n• 행동 유발 [1]"
- N이 명시되지 않은 경우 검색 결과에 있는 모든 항목을 나열.

【list_n 자기검증 4단계】 (답변 작성 직전 반드시 수행)
1. 질문에서 N 추출 (예: "3가지", "4개", "각 단계" → 모든 항목)
2. 검색 결과에서 답이 될 N 개 항목 후보를 verbatim 으로 나열
3. 각 항목이 검색 결과에 그대로 등장하는가? (paraphrase·번역 금지)
4. 다른 슬라이드의 무관한 항목 (배경 설명, 다른 주제) 이 섞이지 않았는가?
   ↳ 의심되면 제거. 정답 N개 보다 적게 답하는 것이 잘못된 항목 추가보다 낫다.
""",

    "fill_blank": _COMMON_PRINCIPLES + """
【fill_blank 응답 규칙】
- 빈칸에 들어갈 단어/구만 답. 전체 문장 절대 금지.
- 예: "충격 [1]" (질문이 'Day 1 = 첫인상 + ___ 충격'일 때 → 답: "이게 되네!")
""",

    "comparison": _COMMON_PRINCIPLES + """
【comparison 응답 규칙】
- 비교 대상 entity별로 명시.
- 검색 결과에 있는 차이만. 추측 금지.
- 예: "• 일반 매칭: 단발성 연결, 자원 제공 [1]\\n• 그레이트프로: 지속 가능한 동반, 가치 존중 [1]"
""",

    "reasoning": _COMMON_PRINCIPLES + """
【reasoning 응답 규칙】
- 검색 결과에 있는 사실로만 추론.
- 짧게 답하되 핵심 근거에 [N] 부착.
- 결론 → 근거 순서.
""",
}


def _format_docs_for_context(docs: list[Document]) -> str:
    if not docs:
        return ""
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", d.source)
        loc = d.metadata.get("location", "")
        blocks.append(f"[{i}] {title}{(' ' + loc) if loc else ''}\n{(d.content or '')[:1500]}")
    return "\n\n".join(blocks)


def _extract_cited_ids(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\[(\d{1,3})\]", text or "")}


def _build_citations(docs: list[Document], cited_ids: set[int]) -> list[Citation]:
    out = []
    for i, d in enumerate(docs, start=1):
        if i not in cited_ids:
            continue
        out.append(Citation(
            id=i,
            doc_id=d.metadata.get("doc_id", d.source),
            snippet=(d.content or "")[:200],
            score=d.score,
            location=d.metadata.get("location", ""),
        ))
    return out


def generator_node(state: GraphState) -> dict:
    """답변 생성 + citations 부착."""
    docs: list[Document] = state.retrieved_docs or []
    user_input = (state.input_data or "").strip()

    # docs 없으면 명시적 거절. 시스템 로그는 두 케이스 분리:
    #  (a) 처음부터 retrieve가 결과를 못 가져옴 — 사내 문서에 정말 없음 가능성
    #  (b) rewrite 3회 후에도 못 찾음 — 시스템 회복 실패
    # 사용자 응답은 동일.
    if not docs:
        msg = "관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
        path = state.decision_path or []
        exhausted = any("exhausted_after_3rewrites" in p for p in path)
        if exhausted:
            print("[GENERATOR] no_docs (retrieve exhausted after 3 rewrites)")
            log_label = "generate:no_docs(exhausted)"
        else:
            print("[GENERATOR] no_docs (initial empty retrieve)")
            log_label = "generate:no_docs(initial)"
        return {
            "answer": msg,
            "citations": [],
            "messages": [HumanMessage(content=user_input), AIMessage(content=msg)],
            "decision_path": [log_label],
        }

    qtype = state.question_type or "reasoning"
    sys_prompt = _PROMPT_BY_TYPE.get(qtype, _PROMPT_BY_TYPE["reasoning"])
    sys_content = f"{sys_prompt}\n\n【검색 결과】\n{_format_docs_for_context(docs)}"

    try:
        resp = get_llm().invoke([
            SystemMessage(content=sys_content),
            HumanMessage(content=user_input),
        ])
        raw = resp.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        answer = str(raw).strip()
    except Exception as e:
        print(f"[GENERATOR] LLM call failed: {e}")
        answer = "응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    if not answer:
        answer = "관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."

    cited_ids = _extract_cited_ids(answer)
    citations = _build_citations(docs, cited_ids)

    return {
        "answer": answer,
        "citations": citations,
        "messages": [HumanMessage(content=user_input), AIMessage(content=answer)],
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"generate:{qtype}({len(citations)}cit)"],
    }
