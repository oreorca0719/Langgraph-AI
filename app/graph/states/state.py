
from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from operator import add

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # 사용자의 입력 (텍스트 쿼리나 로그 데이터)
    input_data: str

    # 분석 결과: 'chatbot', 'anomaly_detection' 등 (현재 RAG에서는 선택)
    task_type: str

    # 라우팅/에이전트용 추가 인자(선택)
    task_args: Dict[str, Any]

    # 메시지 히스토리 (메시지가 추가될 때마다 리스트에 누적)
    messages: Annotated[Sequence[BaseMessage], add]

    # (레거시) RAG용 컨텍스트 문자열 리스트 - 기존 코드 호환용 (선택)
    context: List[str]

    # 다음으로 이동할 노드 이름 (LangGraph edge로 흐르면 없어도 됨)
    next_node: str

    # ✅ RAG 근거 기반 답변용
    retrieved_docs: List[Document]              # retrieval 결과 원본
    citations: List[Dict[str, Any]]             # 출처 정규화(직렬화 가능)
    citations_used: List[Dict[str, Any]]        # generator에서 실제로 내려줄 출처(선택)

    # ✅ 파일 추출/문서작성 에이전트용(선택)
    extracted_text: str
    extracted_meta: Dict[str, Any]
    draft_email: Dict[str, Any]
    draft_rfp: str

    # ✅ 첨부 파일 컨텍스트 (세션 유지용)
    file_context: Optional[str]
    file_context_name: Optional[str]

    # ✅ 요청 추적용 (트레이스 대시보드)
    trace_id: str


    # analyzer 결과(선택): task_args["analysis"]에 저장됨
    # analysis: Dict[str, Any]  # (직접 키로 쓰고 싶을 때만 사용)