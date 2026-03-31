from __future__ import annotations

import json
from typing import Any, Dict

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_llm
from app.core import trace_buffer
from app.core.history_utils import extract_text_content as _content_to_text
from app.graph.states.state import GraphState
from app.graph.nodes.knowledge_search import _search_hybrid, _format_docs
from app.security.content_sanitizer import sanitize_docs


_AMBIGUOUS_REFS = re.compile(
    r"^(이걸|이거|이것|위\s*내용|앞\s*내용|저거|그거|그것|방금|아까|해당\s*내용)",
    re.I,
)


def _resolve_search_query(user_input: str, messages: list) -> str:
    """
    user_input이 모호한 지시어로 시작하면 직전 AIMessage에서 주제를 추출해
    검색에 적합한 쿼리로 재구성. 그렇지 않으면 user_input 그대로 반환.
    """
    if not _AMBIGUOUS_REFS.search(user_input):
        return user_input

    prev_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            prev_ai = str(msg.content)[:1000]
            break

    if not prev_ai:
        return user_input

    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "이전 AI 답변을 읽고, RFP 문서 검색에 사용할 핵심 주제 키워드를 한 문장으로 추출하세요.\n"
                "검색 쿼리만 출력하고 다른 텍스트는 출력하지 마세요."
            )),
            HumanMessage(content=f"[이전 AI 답변]\n{prev_ai}\n\n[현재 요청]\n{user_input}"),
        ])
        query = _content_to_text(response.content)
        return query if query else user_input
    except Exception:
        return user_input


# ── 에이전트 1: rfp_research_node ────────────────────────────

def rfp_research_node(state: GraphState) -> Dict[str, Any]:
    """
    RFP 작성에 필요한 사내 문서·컨텍스트를 수집합니다.
    - 사용자 요청에서 핵심 키워드 추출
    - ChromaDB에서 관련 문서 검색
    - 검색 결과를 rfp_research 필드에 저장
    """
    print("--- [NODE] RFP Research ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    extracted = (state.get("extracted_text") or "").strip()
    messages = list(state.get("messages") or [])

    planner_context = (state.get("planner_context") or "").strip()
    search_query = _resolve_search_query(user_input, messages)

    trace_buffer.push(trace_id, node="rfp_research", event="enter", label="execute",
                      data={"input": user_input[:200], "search_query": search_query[:200]})

    # 사내 문서에서 관련 자료 검색
    docs = _search_hybrid(search_query, k=5)
    docs = sanitize_docs(docs, source="rfp_research")
    search_result = _format_docs(docs)

    # LLM으로 요구사항·컨텍스트 정리
    research_prompt = (
        "당신은 RFP 작성을 위한 사전 조사 에이전트입니다.\n"
        "아래 정보를 바탕으로 RFP 작성에 필요한 핵심 컨텍스트를 정리하세요.\n\n"
        "정리 항목:\n"
        "1. 프로젝트 배경 및 목적 (추론 가능한 범위)\n"
        "2. 핵심 요구사항 키워드\n"
        "3. 관련 사내 문서에서 참고할 내용 (있는 경우)\n"
        "4. 작성 시 주의사항\n\n"
        "없는 정보는 '[정보 없음]'으로 표시하세요."
    )

    human_content = f"사용자 요청:\n{user_input}"
    if planner_context:
        human_content += f"\n\n[Planner 사전 검색 결과]\n{planner_context}"
    if search_result:
        human_content += f"\n\n사내 문서 검색 결과:\n{search_result}"
    if extracted:
        human_content += f"\n\n첨부 문서 내용:\n{extracted[:5000]}"

    response = get_llm().invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=human_content),
    ])
    research_result = _content_to_text(response.content)

    trace_buffer.push(trace_id, node="rfp_research", event="exit", label="execute",
                      data={"research_len": len(research_result)})

    return {"rfp_research": research_result}


# ── 에이전트 2: rfp_draft_node ───────────────────────────────

def rfp_draft_node(state: GraphState) -> Dict[str, Any]:
    """
    rfp_research 결과를 받아 RFP 초안을 작성합니다.
    수정 요청(human_review revise)인 경우 review_notes를 반영합니다.
    """
    print("--- [NODE] RFP Draft ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    chat_history = state.get("messages") or []
    args = state.get("task_args") or {}
    research = (state.get("rfp_research") or "").strip()
    review_notes = (state.get("rfp_review_notes") or "").strip()
    existing_draft = (state.get("draft_rfp") or "").strip()

    trace_buffer.push(trace_id, node="rfp_draft", event="enter", label="execute",
                      data={"has_research": bool(research), "has_review_notes": bool(review_notes)})

    llm = get_llm()

    system_content = (
        "당신은 RFP(제안요청서/요구사항 정의서) 작성 에이전트입니다.\n"
        "사전 조사 결과와 사용자 요청을 바탕으로 실무형 문서를 작성하세요.\n"
        "한국어로, 아래 목차를 반드시 포함하세요:\n"
        "1. 배경/목적\n"
        "2. 범위(Scope)\n"
        "3. 요구사항(기능/비기능)\n"
        "4. 데이터/연동/보안\n"
        "5. 일정/마일스톤\n"
        "6. 산출물\n"
        "7. 평가 기준\n"
        "8. 가정/제약\n"
    )

    human_parts = [f"요청:\n{user_input}"]
    if research:
        human_parts.append(f"사전 조사 결과:\n{research}")
    if review_notes:
        human_parts.append(f"검토 의견 (반드시 반영):\n{review_notes}")
    if existing_draft:
        human_parts.append(f"기존 초안 (수정 기준):\n{existing_draft[:10000]}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "\n\n".join(human_parts)),
    ])

    resp = (prompt | llm).invoke({
        "chat_history": chat_history,
    })

    draft_text = _content_to_text(resp.content if hasattr(resp, "content") else resp)

    trace_buffer.push(trace_id, node="rfp_draft", event="exit", label="execute",
                      data={"draft_len": len(draft_text)})

    return {
        "draft_rfp": draft_text,
        "draft_email": None,
        "rfp_review_notes": "",
        "current_task": "rfp_draft",
    }


# ── 에이전트 3: rfp_review_node ──────────────────────────────

def rfp_review_node(state: GraphState) -> Dict[str, Any]:
    """
    작성된 RFP 초안을 검토합니다.
    - 8개 필수 섹션 존재 여부 확인
    - 요구사항 누락 여부 확인
    - 미흡 시 rfp_review_notes에 의견을 저장하고 rfp_draft_node로 루프백
    - 충분 시 human_review_node로 진행
    """
    print("--- [NODE] RFP Review ---")

    trace_id = (state.get("trace_id") or "")
    draft = (state.get("draft_rfp") or "").strip()
    user_input = (state.get("input_data") or "").strip()
    retry_count = state.get("retry_count") or 0

    trace_buffer.push(trace_id, node="rfp_review", event="enter", label="execute",
                      data={"draft_len": len(draft), "retry_count": retry_count})

    # 최대 1회 재작성 후 강제 통과
    if retry_count >= 1 or not draft:
        trace_buffer.push(trace_id, node="rfp_review", event="exit", label="execute",
                          data={"result": "pass_forced"})
        return {"rfp_review_notes": "", "retry_count": 0}

    review_prompt = (
        "당신은 RFP 검토 에이전트입니다. 아래 RFP 초안을 검토하고 JSON으로만 응답하세요.\n\n"
        "검토 기준:\n"
        "1. 8개 필수 섹션(배경/목적, 범위, 요구사항, 데이터/연동/보안, 일정/마일스톤, 산출물, 평가 기준, 가정/제약) 모두 포함\n"
        "2. 요구사항 섹션에 기능/비기능 요구사항이 구체적으로 기술\n"
        "3. 사용자 요청의 핵심 내용이 반영\n\n"
        "응답 형식:\n"
        "{\"pass\": true/false, \"notes\": \"미흡한 점 한 줄 요약 (pass=true면 빈 문자열)\"}"
    )

    try:
        from langchain_core.output_parsers import JsonOutputParser
        chain = get_llm() | JsonOutputParser()
        result = chain.invoke([
            SystemMessage(content=review_prompt),
            HumanMessage(content=f"사용자 요청:\n{user_input}\n\nRFP 초안:\n{draft[:8000]}"),
        ])
        is_pass = bool(result.get("pass", True))
        notes = str(result.get("notes", "")).strip()
    except Exception:
        is_pass = True
        notes = ""

    trace_buffer.push(trace_id, node="rfp_review", event="exit", label="execute",
                      data={"result": "pass" if is_pass else "needs_revision", "notes": notes[:100]})

    if is_pass:
        return {"rfp_review_notes": "", "retry_count": 0}

    return {"rfp_review_notes": notes, "retry_count": (retry_count + 1)}


def route_after_rfp_review(state: GraphState) -> str:
    """rfp_review_node 이후 분기."""
    notes = (state.get("rfp_review_notes") or "").strip()
    if notes:
        return "rfp_draft"     # 재작성
    return "human_review"      # 검토 통과 → 사용자 확인
