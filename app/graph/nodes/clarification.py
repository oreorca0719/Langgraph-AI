from __future__ import annotations

import re

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.core.config import get_llm
from app.core import trace_buffer
from app.core.history_utils import extract_text_content
from app.graph.states.state import GraphState


_FILE_REFERENCE_RE = re.compile(r"(?:^|[\/\s])[^\/\s]+\.(pdf|docx|pptx|xlsx|xlsm|txt|md|csv|log)\b", re.I)
_COMMAND_RE = re.compile(
    r"(해줘|해주세요|해줄|작성|생성|만들어|써줘|써주|고쳐|수정|바꿔|찾아|알려|보내줘|보내주|초안|이메일|rfp|메일)",
    re.I,
)


def _looks_like_file_reference(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if _FILE_REFERENCE_RE.search(text):
        return True
    if any(token in text for token in ("./", ".\\", "/", "\\")):
        return True
    return False


# ── Priority 3: 슬롯별 고정 질문 ───────────────────────────────
_SLOT_QUESTIONS: dict[str, str] = {
    "file_path":     "분석할 파일 경로 또는 파일명을 알려주세요.",
    "file_context":  "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요.",
    "to":            "이메일 수신자 또는 받는 부서를 알려주세요. (예: hr@example.com 또는 인사팀)",
    "subject":       "이메일의 목적이나 주요 내용을 알려주세요. (예: AI 교육 제안 관련 미팅 요청)",
    "project_scope": "RFP 대상 프로젝트명이나 제안 범위를 알려주세요.",
    "rfp_content":   "RFP에 포함할 주요 요구사항이나 목적을 알려주세요. (예: 기존 시스템 연동 필요, 6개월 내 납품 목표)",
}


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    라우팅 신뢰도가 낮을 때 사용자 의도를 확인하는 노드.

    흐름:
    1. clarification_count >= 2 → 무한 루프 방어, knowledge_search 강제 fallback
    2. LLM이 사내 기능 연관 여부 판단
       - 범위 외 → rejection_node로 직행
       - 연관 가능 → 의도 확인 질문 생성
    3. interrupt()로 질문을 사용자에게 전달하고 응답 대기
    4. 재개 후 original + 응답을 결합 → task_router_node 루프백
    """
    print("--- [NODE] Clarification ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    clarification_count = (state.get("clarification_count") or 0)
    task_type = (state.get("task_type") or "").strip()
    task_args = state.get("task_args") or {}
    missing_slots: list[str] = task_args.get("missing_slots") or []

    trace_buffer.push(trace_id, node="clarification", event="enter", label="execute",
                      data={"input": user_input[:200], "count": clarification_count, "missing_slots": missing_slots})

    # 루프 방어: 3회 이상 발동 시 강제 knowledge_search fallback (슬롯 2단계 수집 허용)
    if clarification_count >= 3:
        trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                          data={"result": "loop_fallback"})
        old_args = state.get("task_args") or {}
        clean_args = {k: v for k, v in old_args.items() if k != "missing_slots"}
        clean_args["routing_debug"] = {"final_source": "clarification_loop_fallback"}
        return {
            "task_type": "knowledge_search",
            "clarification_count": 0,
            "task_args": clean_args,
        }

    # ── Priority 3: 슬롯 기반 clarification (LLM 우회) ───────────────
    if missing_slots:
        slot = missing_slots[0]
        question_text = _SLOT_QUESTIONS.get(
            slot,
            f"'{slot}' 정보를 추가로 알려주세요."
        )

        trace_buffer.push(trace_id, node="clarification", event="call", label="execute",
                          data={"mode": "slot_question", "slot": slot})

        user_response = interrupt({
            "type":    "clarification",
            "message": question_text,
        })

        normalized_response = user_response.strip()

        # 슬롯 응답 유효성 검증
        slot_invalid = False
        if slot in {"file_path", "file_context"} and not _looks_like_file_reference(normalized_response):
            slot_invalid = True
        elif slot in {"to", "subject"} and _COMMAND_RE.search(normalized_response):
            # 명령형 문장은 슬롯 값이 아님 (예: "이메일 초안 하나 생성해줘")
            slot_invalid = True

        if slot_invalid:
            retry_args = {**task_args, "missing_slots": [slot]}
            trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                              data={"result": "slot_retry_invalid", "slot": slot,
                                    "slot_value_len": len(normalized_response)})
            return {
                "input_data": user_input,
                "task_type": "",
                "task_args": retry_args,
                "messages": [HumanMessage(content=user_input), AIMessage(content=question_text)],
                "interrupt_type": "",
                "clarification_count": clarification_count + 1,
            }

        # 슬롯 확인 후 요청 내용 정리해서 재확인 질문
        _TASK_LABELS = {
            "email_draft":  "이메일 초안 작성",
            "rfp_draft":    "RFP 초안 작성",
            "file_extract": "파일 분석",
            "file_chat":    "파일 Q&A",
        }
        _SLOT_LABELS = {
            "to":            f"수신자: {normalized_response}",
            "subject":       f"내용: {normalized_response}",
            "project_scope": f"범위: {normalized_response}",
            "rfp_content":   f"요구사항: {normalized_response}",
            "file_path":     f"파일: {normalized_response}",
            "file_context":  f"파일: {normalized_response}",
        }
        # task_args의 routing_debug에서 원래 의도된 task 이름 가져오기
        routing_debug = task_args.get("routing_debug") or {}
        intended_task = routing_debug.get("original_task") or task_type
        task_label = _TASK_LABELS.get(intended_task, "요청")
        slot_label = _SLOT_LABELS.get(slot, normalized_response)
        confirm_msg = (
            f"요청하신 내용을 확인했습니다.\n\n"
            f"**{task_label}**\n"
            f"- {slot_label}\n\n"
            f"진행할까요?"
        )
        interrupt({"type": "clarification", "message": confirm_msg})

        combined = f"{user_input} {normalized_response}".strip()
        new_task_args = {k: v for k, v in task_args.items() if k != "missing_slots"}

        if slot == "to":
            new_task_args["to"] = normalized_response
        elif slot == "subject":
            new_task_args["subject"] = normalized_response
        elif slot == "rfp_content":
            new_task_args["rfp_content"] = normalized_response
        elif slot in {"file_path", "file_context"}:
            new_task_args["file_path"] = normalized_response
        elif slot == "project_scope":
            new_task_args["project_scope"] = normalized_response

        trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                          data={"result": "slot_confirmed", "slot": slot,
                                "combined_len": len(combined), "slot_value_len": len(normalized_response)})
        return {
            "input_data":          combined,
            "task_type":           "",
            "task_args":           new_task_args,
            "messages":            [HumanMessage(content=user_input),
                                    AIMessage(content=question_text)],
            "interrupt_type":      "",
            "clarification_count": clarification_count + 1,
        }

    # ── 기존 LLM 기반 clarification (missing_slots 없음) ─────────────
    system_content = (
        "당신은 사내 AI 어시스턴트입니다. 사용자의 요청 의도가 불분명합니다.\n\n"
        "이 시스템이 제공하는 기능:\n"
        "① 사내 문서·정보 검색\n"
        "② 이메일 초안 작성\n"
        "③ RFP 초안 작성\n"
        "④ 파일 분석\n"
        "⑤ AI 기능 안내\n\n"
        "아래 두 가지 중 하나로만 응답하세요.\n\n"
        "A) 요청이 위 기능 중 하나와 연관될 가능성이 있으면:\n"
        "   어떤 작업을 원하시는지 1문장으로 질문하세요.\n\n"
        "B) 요청이 위 기능과 전혀 무관하면:\n"
        "   정확히 이 문장만 출력: "
        "'해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다.'\n\n"
        "A 또는 B 중 하나만 출력하고 다른 내용은 추가하지 마세요."
    )

    response = get_llm().invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ])
    response_text = extract_text_content(response.content)
    is_rejection = "지원 범위에 포함되지 않아" in response_text

    trace_buffer.push(trace_id, node="clarification", event="call", label="execute",
                      data={"is_rejection": is_rejection})

    if is_rejection:
        trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                          data={"result": "rejection"})
        return {
            "task_type": "rejection",
            "messages": [HumanMessage(content=user_input), AIMessage(content=response_text)],
        }

    # interrupt: 질문을 프론트엔드에 전달하고 사용자 응답 대기
    user_response = interrupt({
        "type": "clarification",
        "message": response_text,
    })

    # 재개: original + 응답 결합 후 task_router로 루프백
    combined = f"{user_input} {user_response}".strip()
    trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                      data={"result": "resumed", "combined_len": len(combined)})

    return {
        "input_data": combined,
        "task_type": "",          # task_router가 재분류
        "task_args": task_args,   # 기존 task_args 보존 (라우팅 컨텍스트 유지)
        "messages": [HumanMessage(content=user_input), AIMessage(content=response_text)],
        "interrupt_type": "",
        "clarification_count": clarification_count + 1,
    }


def route_after_clarification(state: GraphState) -> str:
    """clarification_node 이후 분기."""
    task = (state.get("task_type") or "").strip()
    if task == "rejection":
        return "rejection"
    if task == "knowledge_search":
        return "knowledge_search"  # 루프 방어 fallback 경로
    return "task_router"
