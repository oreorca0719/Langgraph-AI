"""
Main graph builder — v2 그래프 조립.

토폴로지:
  ┌─────────┐
  │  START  │
  └─────────┘
       ↓
  [security_gate]
       ├─(blocked)→ END
       └─(pass)
       ↓
  [router]
       ├─(no_retrieval)→ [no_retrieval_answer] → END
       ├─(ai_guide)→ [ai_guide] → END
       ├─(file_chat)→ [file_chat] → END
       └─(single/multi)
       ↓
  [retrieve_subgraph]
       ↓
  [generate_subgraph]
       ↓
  (verification.passed?)
       ├─(pass)→ END
       └─(fail + replan < 1)→ [router] (재계획)
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import GraphState
from app.graph_v2.nodes.security import security_gate_node, route_after_security
from app.graph_v2.nodes.router import router_node, route_by_decision
from app.graph_v2.nodes.qa_lookup import qa_lookup_node, route_after_qa_lookup
from app.graph_v2.subgraphs.retrieve import build_retrieve_subgraph
from app.graph_v2.subgraphs.generate import build_generate_subgraph


# ─── Leaf nodes (검색 없는 분기) ──────────────────────────────

def _no_retrieval_answer_node(state: GraphState) -> dict:
    """검색 없이 일반 응답 (인사·메타 질의). 짧은 정중 답변."""
    user_input = state.input_data or ""
    sys = (
        "당신은 Kaiper AI 사내 어시스턴트입니다. 인사·간단한 메타 질의에 짧고 정중하게 답하세요. "
        "사내 문서 검색·파일 분석·기능 안내 등을 도울 수 있다고 안내하세요. "
        "구체적인 사실 회수가 필요한 질의는 검색 기능 사용을 권유하세요."
    )
    try:
        resp = get_llm().invoke([SystemMessage(content=sys), HumanMessage(content=user_input)])
        ans = extract_text_content(resp.content) or "안녕하세요. Kaiper AI 사내 어시스턴트입니다."
    except Exception:
        ans = "안녕하세요. Kaiper AI 사내 어시스턴트입니다."
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "decision_path": ["no_retrieval_answer"],
    }


def _ai_guide_node(state: GraphState) -> dict:
    """기능 안내 응답."""
    user_input = state.input_data or ""
    ans = (
        "Kaiper AI 사내 어시스턴트는 다음 기능을 제공합니다.\n"
        "• **사내 문서 검색**: 사내 정책·매뉴얼·기획 산출물·회의록 등을 자연어로 검색하고 출처와 함께 답변\n"
        "• **심화 검색**: 직전 검색 결과에 대한 후속 질의 처리\n"
        "• **파일 분석**: PDF·DOCX·XLSX·PPTX·TXT 첨부파일의 내용 요약·Q&A\n\n"
        "구체적인 질문(예: '체크리스트 9번 항목', '2일차 오후 시간')은 그대로 입력하시면 됩니다."
    )
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "decision_path": ["ai_guide"],
    }


def _file_chat_node(state: GraphState) -> dict:
    """file_context 있을 때만 사용 — 파일 내용 기반 답변."""
    user_input = state.input_data or ""
    file_text = (state.file_context or "").strip()
    if not file_text:
        ans = "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요."
        return {
            "answer": ans,
            "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
            "decision_path": ["file_chat:no_file"],
        }
    sys = (
        "사용자가 첨부한 파일 내용을 분석하는 어시스턴트입니다. 아래 [파일 내용]만을 근거로 답하세요.\n"
        "파일에 없는 내용은 '파일에서 확인할 수 없습니다'로 답하세요.\n\n"
        f"[파일 내용]\n{file_text[:8000]}"
    )
    try:
        resp = get_llm().invoke([SystemMessage(content=sys), HumanMessage(content=user_input)])
        ans = extract_text_content(resp.content) or "응답을 생성하지 못했습니다."
    except Exception as e:
        ans = f"파일 분석 중 오류가 발생했습니다: {e}"
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": ["file_chat"],
    }


def _rejected_node(state: GraphState) -> dict:
    """Security 또는 router에서 차단된 케이스. messages에 추가 안 함 (history 오염 방지)."""
    user_input = state.input_data or ""
    ans = "해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요."
    return {
        "answer": ans,
        "decision_path": ["rejected"],
    }


# ─── Replan reset (router 회귀 전 카운터·상태 초기화) ─────

def _replan_reset_node(state: GraphState) -> dict:
    """Reflection fail 후 router 회귀 시 검색 결과·답변·검증 reset, replan_iterations++."""
    return {
        "retrieved_docs": [],
        "citations": [],
        "answer": "",
        "verification": None,
        "retrieval_iterations": 0,
        "sub_questions": [],
        "replan_iterations": state.replan_iterations + 1,
        "decision_path": [f"replan:{state.replan_iterations + 1}"],
    }


# ─── Main graph builder ──────────────────────────────────

def build_main_graph(checkpointer=None):
    g = StateGraph(GraphState)

    # Compile subgraphs
    retrieve_sub = build_retrieve_subgraph()
    generate_sub = build_generate_subgraph()

    # Nodes
    g.add_node("security_gate", security_gate_node)
    g.add_node("router", router_node)
    g.add_node("qa_lookup", qa_lookup_node)
    g.add_node("retrieve", retrieve_sub)
    g.add_node("generate", generate_sub)
    g.add_node("replan_reset", _replan_reset_node)
    g.add_node("no_retrieval_answer", _no_retrieval_answer_node)
    g.add_node("ai_guide", _ai_guide_node)
    g.add_node("file_chat", _file_chat_node)
    g.add_node("rejected", _rejected_node)

    g.set_entry_point("security_gate")

    # Edges
    g.add_conditional_edges(
        "security_gate", route_after_security,
        {"rejected": "rejected", "router": "router"},
    )
    # router → retrieve 경로는 qa_lookup을 먼저 거침 (캐시 hit 시 retrieve 스킵)
    g.add_conditional_edges(
        "router", route_by_decision,
        {
            "rejected": "rejected",
            "no_retrieval_answer": "no_retrieval_answer",
            "ai_guide": "ai_guide",
            "file_chat": "file_chat",
            "retrieve": "qa_lookup",
        },
    )
    g.add_conditional_edges(
        "qa_lookup", route_after_qa_lookup,
        {"end": END, "retrieve": "retrieve"},
    )
    g.add_edge("retrieve", "generate")

    def _route_after_generate(state: GraphState) -> str:
        if (state.verification and state.verification.needs_replan()
                and state.replan_iterations < 1):
            return "replan"
        return "end"

    g.add_conditional_edges(
        "generate", _route_after_generate,
        {"replan": "replan_reset", "end": END},
    )
    g.add_edge("replan_reset", "router")
    g.add_edge("no_retrieval_answer", END)
    g.add_edge("ai_guide", END)
    g.add_edge("file_chat", END)
    g.add_edge("rejected", END)

    return g.compile(checkpointer=checkpointer)
