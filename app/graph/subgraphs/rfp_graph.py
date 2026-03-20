from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.states.state import GraphState
from app.graph.nodes.rfp_agent import rfp_draft_node


def build_rfp_subgraph():
    """
    rfp_draft 전용 서브그래프:
      rfp_draft -> END
    """
    g = StateGraph(GraphState)

    g.add_node("rfp_draft", rfp_draft_node)

    g.set_entry_point("rfp_draft")
    g.add_edge("rfp_draft", END)

    return g.compile()