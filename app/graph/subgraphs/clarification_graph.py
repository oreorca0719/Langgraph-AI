from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import get_llm
from app.core import trace_buffer
from app.graph.states.state import GraphState


def build_clarification_subgraph():
    """
    라우팅 신뢰도가 낮을 때 사용자 의도를 확인하는 서브그래프.
    - 사내 기능과 연관 가능한 요청 → 의도 확인 질문 반환 + clarification_needed=True
    - 명백히 범위 외 요청         → 거절 메시지 반환
    """

    def clarification_node(state: GraphState) -> Dict[str, Any]:
        print("--- [NODE] Clarification ---")

        user_input = (state.get("input_data") or "").strip()
        trace_id = (state.get("trace_id") or "")

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
            "   어떤 작업을 원하시는지 1문장으로 질문하세요.\n"
            "   예: '혹시 기존 문서를 검색하시나요, 아니면 새로 작성을 원하시나요?'\n\n"
            "B) 요청이 위 기능과 전혀 무관하면:\n"
            "   '해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. "
            "사내 업무 관련 질문을 입력해 주세요.'라고만 답하세요.\n\n"
            "규칙: A 또는 B 중 하나만 출력하고, 다른 내용은 추가하지 마세요."
        )

        response = get_llm().invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ])

        response_text = str(response.content).strip()
        is_rejection = "지원 범위에 포함되지 않아" in response_text

        trace_buffer.push(trace_id, node="clarification", event="exit", label="execute",
                          data={"is_rejection": is_rejection})

        if is_rejection:
            return {
                **state,
                "messages": [HumanMessage(content=user_input), AIMessage(content=response_text)],
                "clarification_needed": False,
                "clarification_original_input": "",
                "citations_used": [],
            }

        return {
            **state,
            "messages": [HumanMessage(content=user_input), AIMessage(content=response_text)],
            "clarification_needed": True,
            "clarification_original_input": user_input,
            "citations_used": [],
        }

    g = StateGraph(GraphState)
    g.add_node("clarification", clarification_node)
    g.set_entry_point("clarification")
    g.add_edge("clarification", END)
    return g.compile()
