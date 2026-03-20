from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.states.state import GraphState
from app.graph.nodes.email_agent import email_draft_node


def build_email_subgraph():
    """
    email_draft 전용 서브그래프:
      email_draft -> END
    """
    g = StateGraph(GraphState)

    g.add_node("email_draft", email_draft_node)

    g.set_entry_point("email_draft")
    g.add_edge("email_draft", END)

    return g.compile()