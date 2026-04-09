from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from operator import add

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    # ── 기본 입출력 ──────────────────────────────────────────
    input_data: str
    input_embedding: List[float]       # ← 임베딩 캐시: 사용자 입력 벡터화 결과
    task_type: str
    task_args: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], add]
    citations_used: List[Dict[str, Any]]

    # ── 파일 ────────────────────────────────────────────────
    extracted_text: str
    extracted_meta: Dict[str, Any]
    file_context: Optional[str]
    file_context_name: Optional[str]

    # ── 초안 ────────────────────────────────────────────────
    draft_email: Dict[str, Any]
    draft_rfp: str

    # ── RFP 다중 에이전트 ────────────────────────────────────
    rfp_research: str           # research_node 수집 결과
    rfp_review_notes: str       # review_node 검토 의견

    # ── 순환 / 상태 제어 ─────────────────────────────────────
    retry_count: int            # knowledge search 재시도 횟수
    clarification_count: int    # clarification 발동 횟수 (루프 방어용)
    review_action: str          # "approve" | "revise" | "switch"
    current_task: str           # human_review switch 감지 기준
    interrupt_type: str         # "clarification" | "human_review"

    # ── Planner ──────────────────────────────────────────
    task_sequence: List[str]     # planner가 결정한 실행 순서
    planner_context: str         # 선행 task 실행 결과 컨텍스트

    # ── 트레이스 ─────────────────────────────────────────────
    trace_id: str
