from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.config import get_llm
from app.core import trace_buffer
from app.graph.states.state import GraphState


# ── 라우팅 결과 스키마 ────────────────────────────────────────────

class RouteDecision(BaseModel):
    task: Literal[
        "email_draft", "rfp_draft", "knowledge_search",
        "file_chat", "file_extract", "ai_guide",
        "detail_search", "planner", "rejection",
    ]
    missing_slots: List[Literal["to", "subject", "project_scope", "rfp_content", "file_path", "file_context"]] = Field(
        default_factory=list,
        description="태스크 실행에 필요하지만 사용자 입력에 없는 슬롯",
    )


# ── 라우터 시스템 프롬프트 ────────────────────────────────────────

_ROUTER_SYSTEM = """\
당신은 사내 AI 어시스턴트의 라우터입니다.
사용자 요청을 아래 태스크 중 정확히 하나로 분류하고,
태스크 실행에 필요한 정보가 없으면 missing_slots에 포함하세요.

[태스크 목록]
- email_draft      : 이메일·메일 초안 작성 요청
- rfp_draft        : RFP·제안요청서·요구사항 정의서 작성 요청
- knowledge_search : 사내 문서·정보 검색·질의 응답
- file_chat        : 업로드된 파일 내용 질문·분석 (파일이 있을 때만)
- file_extract     : 파일 텍스트 추출 요청
- ai_guide         : 인사·자기소개·기능 안내 요청
- detail_search    : 직전 검색 결과에 대한 심화·추가 질의
- planner          : 검색 후 문서 작성 등 2단계 복합 요청
- rejection        : 사내 업무와 전혀 무관한 요청 (날씨, 게임, 개인 질문 등)

[missing_slots 판단 기준]
- "to"            : email_draft인데 수신자(사람·부서·회사)를 전혀 특정할 수 없을 때
- "subject"       : email_draft인데 이메일의 목적·주제·내용을 전혀 알 수 없을 때
- "project_scope" : rfp_draft인데 대상 프로젝트·범위를 전혀 알 수 없을 때
- "rfp_content"   : rfp_draft인데 프로젝트 범위는 있으나 요구사항·목적·배경을 전혀 알 수 없을 때
- "file_path"     : file_extract인데 파일 경로·파일명이 없을 때
- "file_context"  : file_chat인데 업로드된 파일이 없다고 명시된 경우

[판단 원칙]
- 수신자가 회사명·부서명·직함으로라도 특정된다면 "to" 슬롯은 누락이 아닙니다.
- 이메일 목적·주제가 조금이라도 언급되면 "subject" 슬롯은 누락이 아닙니다.
- detail_search는 직전 대화에서 knowledge_search·detail_search가 있었을 때만 선택하세요.
- planner는 "~검색해서 ~작성해줘" 처럼 선행 조사 + 후행 작성이 명확히 결합된 경우입니다.
- 애매하면 knowledge_search를 기본으로 선택하세요.
"""


# ── 노드 ─────────────────────────────────────────────────────────

def task_router_node(state: GraphState) -> Dict[str, Any]:
    print("--- [NODE] Task Router ---")

    trace_id   = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    task_args  = state.get("task_args") or {}

    trace_buffer.push(trace_id, node="task_router", event="enter", label="execute",
                      data={"input": user_input[:200]})

    if not user_input:
        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={"task_type": "knowledge_search", "mode": "empty_input"})
        return {"task_type": "knowledge_search", "task_args": task_args}

    # ── 세션 컨텍스트 구성 ────────────────────────────────────────
    file_context_present = bool((state.get("file_context") or "").strip())
    prev_task            = (state.get("task_type") or "").strip()
    prev_messages        = state.get("messages") or []

    context_lines: List[str] = []
    if file_context_present:
        file_name = (state.get("file_context_name") or "파일")
        context_lines.append(f"- 현재 세션에 '{file_name}' 파일이 업로드되어 있습니다.")
    else:
        context_lines.append("- 현재 세션에 업로드된 파일이 없습니다.")

    if prev_task in ("knowledge_search", "detail_search") and prev_messages:
        context_lines.append(f"- 직전 작업: {prev_task} (심화 질의 가능)")

    context_section = "\n".join(context_lines)

    try:
        llm_router = get_llm().with_structured_output(RouteDecision)
        decision: RouteDecision = llm_router.invoke([
            SystemMessage(content=_ROUTER_SYSTEM + f"\n\n[현재 세션 정보]\n{context_section}"),
            HumanMessage(content=user_input),
        ])

        routed        = decision.task
        missing_slots = list(decision.missing_slots)

        debug: Dict[str, Any] = {
            "mode":          "llm",
            "decision":      routed,
            "missing_slots": missing_slots,
            "final_source":  "llm_router",
        }

        if missing_slots:
            debug["original_task"] = routed          # 원래 task 보존 (clarification 확인 메시지용)
            routed                = "clarification"
            debug["decision"]     = "clarification"
            debug["final_source"] = "slot_missing"

        merged_args = {**task_args, "routing_debug": debug}
        if missing_slots:
            merged_args["missing_slots"] = missing_slots

        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={
                              "task_type":    routed,
                              "mode":         "llm",
                              "final_source": debug.get("final_source", ""),
                          })
        return {"task_type": routed, "task_args": merged_args}

    except Exception as e:
        # LLM 실패 시 knowledge_search로 안전 폴백
        fallback_debug: Dict[str, Any] = {
            "mode":         "error_fallback",
            "decision":     "knowledge_search",
            "final_source": "error_fallback",
            "reason":       str(e),
        }
        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={"task_type": "knowledge_search", "mode": "error_fallback", "error": str(e)})
        return {"task_type": "knowledge_search", "task_args": {**task_args, "routing_debug": fallback_debug}}


def rejection_node(state: GraphState) -> Dict[str, Any]:
    print("--- [NODE] Rejection ---")
    user_input = (state.get("input_data") or "").strip()
    trace_id   = (state.get("trace_id") or "")
    trace_buffer.push(trace_id, node="rejection", event="exit", label="execute",
                      data={"input": user_input[:200]})
    return {
        **state,
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content="해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요."),
        ],
        "citations_used": [],
    }


_ALLOWED = {
    "knowledge_search", "ai_guide", "file_chat", "file_extract",
    "email_draft", "rfp_draft", "detail_search", "planner",
    "clarification", "rejection",
}


def route_by_task(state: GraphState) -> str:
    task = (state.get("task_type") or "knowledge_search").strip()
    if task in ("chat", "unknown", ""):
        return "knowledge_search"
    if task == "injection":
        return "rejection"
    if task in _ALLOWED:
        return task
    return "knowledge_search"


def route_after_input_guard(state: GraphState) -> str:
    task = (state.get("task_type") or "").strip()
    if task == "injection":
        return "rejection"
    return "task_router"
