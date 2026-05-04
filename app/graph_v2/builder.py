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

from langgraph.graph import StateGraph, END

from app.graph_v2.states.state import GraphState
from app.graph_v2.nodes.security import security_gate_node, route_after_security
from app.graph_v2.nodes.router import router_node, route_by_decision
from app.graph_v2.subgraphs.retrieve import build_retrieve_subgraph
from app.graph_v2.subgraphs.generate import build_generate_subgraph


# ─── Placeholder leaf nodes ──────────────────────────────

def _no_retrieval_answer_node(state: GraphState) -> dict:
    # TODO: 검색 없이 일반 응답 (인사·메타 질의)
    return {"answer": "안녕하세요. Kaiper AI 사내 어시스턴트입니다.", "decision_path": ["no_retrieval_answer"]}


def _ai_guide_node(state: GraphState) -> dict:
    # TODO: 기능 안내 응답
    return {"answer": "사내 문서 검색·심화 검색·파일 분석 기능을 제공합니다.", "decision_path": ["ai_guide"]}


def _file_chat_node(state: GraphState) -> dict:
    # TODO: file_context 기반 답변
    return {"answer": "", "decision_path": ["file_chat"]}


def _rejected_node(state: GraphState) -> dict:
    return {
        "answer": "해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않습니다.",
        "decision_path": ["rejected"],
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
    g.add_node("retrieve", retrieve_sub)
    g.add_node("generate", generate_sub)
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
    g.add_conditional_edges(
        "router", route_by_decision,
        {
            "rejected": "rejected",
            "no_retrieval_answer": "no_retrieval_answer",
            "ai_guide": "ai_guide",
            "file_chat": "file_chat",
            "retrieve": "retrieve",
        },
    )
    g.add_edge("retrieve", "generate")

    def _route_after_generate(state: GraphState) -> str:
        if (state.verification and state.verification.needs_replan()
                and state.replan_iterations < 1):
            return "router"
        return "end"

    g.add_conditional_edges(
        "generate", _route_after_generate,
        {"router": "router", "end": END},
    )
    g.add_edge("no_retrieval_answer", END)
    g.add_edge("ai_guide", END)
    g.add_edge("file_chat", END)
    g.add_edge("rejected", END)

    return g.compile(checkpointer=checkpointer)
