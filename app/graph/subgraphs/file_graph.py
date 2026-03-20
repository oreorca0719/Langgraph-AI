from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.states.state import GraphState
from app.graph.nodes.file_extractor import file_extractor_node


def build_file_subgraph():
    """
    file_extract 전용 서브그래프:
      file_extract -> END
    """
    g = StateGraph(GraphState)

    g.add_node("file_extract", file_extractor_node)

    g.set_entry_point("file_extract")
    g.add_edge("file_extract", END)

    return g.compile()