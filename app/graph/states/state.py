
from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from operator import add

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    input_data: str
    task_type: str
    task_args: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], add]
    citations: List[Dict[str, Any]]
    citations_used: List[Dict[str, Any]]
    extracted_text: str
    extracted_meta: Dict[str, Any]
    draft_email: Dict[str, Any]
    draft_rfp: str
    file_context: Optional[str]
    file_context_name: Optional[str]
    trace_id: str