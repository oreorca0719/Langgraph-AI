from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.core.config import get_llm
from app.core import trace_buffer
from app.core.history_utils import HISTORY_MAX_MESSAGES, filter_history_by_relevance as _filter_history_by_relevance
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize


def build_chat_subgraph():
    """
    첨부 파일 Q&A 전용 서브그래프 (file_chat).
    get_attached_file 단일 도구만 사용. 사내 RAG 검색 없음.
    """

    def file_chat_node(state: GraphState) -> Dict[str, Any]:
        print("--- [NODE] File Chat ---")

        trace_id          = (state.get("trace_id") or "")
        user_input        = (state.get("input_data") or "").strip()
        raw_history       = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
        chat_history      = _filter_history_by_relevance(raw_history, user_input)
        file_context      = (state.get("file_context") or "").strip()
        file_context_name = (state.get("file_context_name") or "첨부 파일").strip()

        trace_buffer.push(trace_id, node="file_chat", event="enter", label="execute",
                          data={"input": user_input[:200], "has_file": bool(file_context)})

        @tool
        def get_attached_file() -> str:
            """사용자가 현재 세션에서 첨부한 파일의 전체 내용을 반환합니다."""
            if not file_context:
                return "현재 첨부된 파일이 없습니다."
            print(f"DEBUG: [Tool] get_attached_file name={file_context_name}")
            trace_buffer.push(trace_id, node="tool:get_attached_file", event="call",
                              label="execute", data={"file_name": file_context_name})
            safe_context = sanitize(file_context, source=f"file:{file_context_name}")
            return f"[파일명: {file_context_name}]\n\n{safe_context}"

        system_content = (
            "당신은 Kaiper AI입니다. 사용자가 업로드한 파일을 분석하고 질문에 답하는 역할입니다.\n\n"
            f"현재 첨부된 파일: [{file_context_name}]\n"
            "get_attached_file 도구로 파일 내용을 읽고 사용자 질문에 답하세요.\n\n"
            "• 파일 내용 중심으로 답하고, 없는 내용은 생성하지 마세요.\n"
            "• 정중한 비즈니스 어투로 답하세요."
        )

        agent = create_react_agent(get_llm(), [get_attached_file])
        agent_input = {
            "messages": [SystemMessage(content=system_content)]
            + chat_history
            + [HumanMessage(content=user_input)]
        }

        result = agent.invoke(agent_input)
        final_message = result["messages"][-1]

        finish_reason = (
            (getattr(final_message, "response_metadata", None) or {}).get("finish_reason", "") or ""
        ).upper()
        if finish_reason == "MAX_TOKENS":
            final_message = AIMessage(
                content="요청하신 내용이 처리 가능한 텍스트 범위를 초과하였습니다. 질문을 더 구체적으로 나누어 입력해 주세요."
            )

        response_len = len(str(final_message.content))
        print(f"DEBUG: [File Chat] 응답 생성 완료 ({response_len}자)")
        trace_buffer.push(trace_id, node="file_chat", event="exit", label="execute",
                          data={"response_len": response_len})

        return {
            **state,
            "messages": [HumanMessage(content=user_input), final_message],
            "citations_used": [],
        }

    g = StateGraph(GraphState)
    g.add_node("file_chat", file_chat_node)
    g.set_entry_point("file_chat")
    g.add_edge("file_chat", END)
    return g.compile()
