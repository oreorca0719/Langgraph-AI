from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import get_llm
from app.core import trace_buffer
from app.graph.states.state import GraphState


_ALLOWED_TASKS = {
    "knowledge_search", "rfp_draft", "email_draft",
    "detail_search", "ai_guide", "file_chat",
}

_PLAN_SYSTEM = (
    "당신은 복합 업무 요청을 분석하는 플래너입니다.\n"
    "사용자 요청에서 수행할 작업들을 실행 순서대로 추출하세요.\n\n"
    "사용 가능한 작업:\n"
    "- knowledge_search: 사내 문서 검색 및 답변\n"
    "- rfp_draft: RFP(제안요청서) 작성\n"
    "- email_draft: 이메일 초안 작성\n"
    "- ai_guide: AI 도구 사용 가이드\n\n"
    "규칙:\n"
    "1. 작업은 최대 2개까지만 추출\n"
    "2. 검색/조회(knowledge_search)는 반드시 작성(rfp_draft/email_draft)보다 먼저 배치\n"
    "3. JSON 형식으로만 응답: {\"tasks\": [\"task1\", \"task2\"]}\n"
    "4. 유효하지 않은 작업 이름은 포함하지 마세요"
)


def _plan_tasks(user_input: str, trace_id: str) -> List[str]:
    """LLM으로 복합 요청을 분해하고 task 실행 순서를 반환."""
    try:
        chain = get_llm() | JsonOutputParser()
        result = chain.invoke([
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(content=f"요청: {user_input}"),
        ])
        tasks = result.get("tasks", [])
        valid = [t for t in tasks if t in _ALLOWED_TASKS][:2]
        return valid if valid else ["knowledge_search"]
    except Exception as e:
        trace_buffer.push(trace_id, node="planner", event="error", label="plan",
                          data={"error": str(e)})
        return ["knowledge_search"]


def _execute_knowledge_search(user_input: str) -> str:
    """지식 검색을 직접 실행하여 포맷된 문서 텍스트 반환."""
    from app.graph.nodes.knowledge_search import _search_hybrid, _format_docs
    from app.security.content_sanitizer import sanitize_docs

    docs = _search_hybrid(user_input, k=5)
    docs = sanitize_docs(docs, source="planner_search")
    return _format_docs(docs)


def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    복합 요청을 분해하고 순차 실행을 오케스트레이션합니다.

    흐름:
    1. LLM으로 task 실행 순서 결정 (최대 2개)
    2. 선행 검색 task(knowledge_search)를 직접 실행 → planner_context 누적
    3. 최종 task(rfp_draft / email_draft 등)를 task_type에 설정 → graph 엣지로 라우팅
    """
    print("--- [NODE] Planner ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()

    trace_buffer.push(trace_id, node="planner", event="enter", label="execute",
                      data={"input": user_input[:200]})

    # 1. task 순서 결정
    task_sequence = _plan_tasks(user_input, trace_id)

    trace_buffer.push(trace_id, node="planner", event="call", label="plan",
                      data={"task_sequence": task_sequence})

    # 2. 선행 task(검색) 직접 실행 → context 누적
    context_parts: List[str] = []
    final_task = task_sequence[-1] if task_sequence else "knowledge_search"
    pre_tasks = task_sequence[:-1]

    for task in pre_tasks:
        if task == "knowledge_search":
            search_result = _execute_knowledge_search(user_input)
            if search_result:
                context_parts.append(f"[사내 문서 검색 결과]\n{search_result}")

    planner_context = "\n\n".join(context_parts)

    trace_buffer.push(trace_id, node="planner", event="exit", label="execute",
                      data={"final_task": final_task, "context_len": len(planner_context)})

    return {
        "task_sequence": task_sequence,
        "planner_context": planner_context,
        "task_type": final_task,
    }


def route_after_planner(state: GraphState) -> str:
    """planner_node 이후 최종 task 노드로 라우팅."""
    task = (state.get("task_type") or "knowledge_search").strip()
    _MAP = {
        "knowledge_search": "knowledge_search",
        "rfp_draft":        "rfp_draft",
        "email_draft":      "email_draft",
        "ai_guide":         "ai_guide",
        "detail_search":    "detail_search",
        "file_chat":        "file_chat",
    }
    return _MAP.get(task, "knowledge_search")
