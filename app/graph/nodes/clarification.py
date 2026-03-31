from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.core.config import get_llm
from app.core import trace_buffer
from app.core.history_utils import extract_text_content
from app.graph.states.state import GraphState


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    라우팅 신뢰도가 낮을 때 사용자 의도를 확인하는 노드.

    흐름:
    1. LLM이 사내 기능 연관 여부 판단
       - 범위 외 → rejection_node로 직행
       - 연관 가능 → 의도 확인 질문 생성
    2. interrupt()로 질문을 사용자에게 전달하고 응답 대기
    3. 재개 후 original + 응답을 결합 → task_router_node 루프백
    """
    print("--- [NODE] Clarification ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()

    trace_buffer.push(trace_id, node="clarification", event="enter", label="execute",
                      data={"input": user_input[:200]})

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
        "messages": [HumanMessage(content=user_input), AIMessage(content=response_text)],
        "interrupt_type": "",
    }


def route_after_clarification(state: GraphState) -> str:
    """clarification_node 이후 분기."""
    task = (state.get("task_type") or "").strip()
    if task == "rejection":
        return "rejection"
    return "task_router"
