from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.core.config import get_embeddings, get_llm
from app.core import trace_buffer
from app.graph.states.state import GraphState


CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_knowledge")


def _search_chroma(query: str, k: int = 3) -> List[Document]:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )
    min_relevance = float(os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.65"))
    max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.75"))

    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        pairs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [doc for doc, score in pairs if score >= min_relevance]

    if hasattr(vectorstore, "similarity_search_with_score"):
        pairs = vectorstore.similarity_search_with_score(query, k=k)
        return [doc for doc, score in pairs if score <= max_distance]

    return vectorstore.similarity_search(query, k=k)


def _format_search_result(docs: List[Document]) -> str:
    if not docs:
        return "관련 문서를 찾을 수 없습니다."
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        md = getattr(d, "metadata", {}) or {}
        title = md.get("title") or md.get("file_name") or "문서"
        blocks.append(f"[{i}] {title}\n{text[:1200]}")
    return "\n\n".join(blocks)


def build_chat_subgraph():
    """
    chat 전용 서브그래프: ReAct Tool-calling 에이전트
    - search_knowledge_base: Chroma 벡터 DB 검색
    - get_attached_file: 첨부 파일 전체 내용 반환
    file_context 판단을 LLM 에이전트에게 위임 → use_file_context 플래그 불필요.
    """

    def chat_agent_node(state: GraphState) -> Dict[str, Any]:
        print("--- [NODE] Chat ReAct Agent ---")

        trace_id = (state.get("trace_id") or "")
        user_input = (state.get("input_data") or "").strip()
        chat_history = list(state.get("messages") or [])
        file_context = (state.get("file_context") or "").strip()
        file_context_name = (state.get("file_context_name") or "첨부 파일").strip()
        k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

        trace_buffer.push(trace_id, node="chat_agent", event="enter", label="execute",
                          data={"input": user_input[:200], "has_file": bool(file_context)})

        # ── 클로저로 런타임 State 캡처 ──

        @tool
        def search_knowledge_base(query: str) -> str:
            """사내 지식 베이스에서 관련 문서를 검색합니다. 사용자 질문과 관련된 정보를 찾을 때 사용하세요."""
            print(f"DEBUG: [Tool] search_knowledge_base query={query[:100]}")
            trace_buffer.push(trace_id, node="tool:search_knowledge_base", event="call",
                              label="execute", data={"query": query[:100]})
            docs = _search_chroma(query, k=k)
            trace_buffer.push(trace_id, node="tool:search_knowledge_base", event="exit",
                              label="execute", data={"docs_found": len(docs)})
            return _format_search_result(docs)

        @tool
        def get_attached_file() -> str:
            """사용자가 현재 세션에서 첨부한 파일의 전체 내용을 반환합니다. 파일 내용에 대한 질문일 때 사용하세요."""
            if not file_context:
                return "현재 첨부된 파일이 없습니다."
            print(f"DEBUG: [Tool] get_attached_file name={file_context_name}")
            trace_buffer.push(trace_id, node="tool:get_attached_file", event="call",
                              label="execute", data={"file_name": file_context_name})
            return f"[파일명: {file_context_name}]\n\n{file_context}"

        # ── 시스템 프롬프트 ──
        file_hint = (
            f"2) 현재 첨부된 파일이 있습니다: [{file_context_name}].\n"
            "   - 파일 자체의 내용(요약, 설명, 분석)이 필요할 때: get_attached_file 사용\n"
            "   - 파일 내용과 관련된 사내 문서/정보를 검색할 때: get_attached_file로 파일을 읽은 뒤 "
            "핵심 키워드를 추출하여 search_knowledge_base를 추가로 호출하세요.\n"
            if file_context
            else "2) 현재 첨부된 파일이 없습니다.\n"
        )

        system_content = (
            "당신은 사내 AI 어시스턴트입니다. 아래 지침을 따르세요.\n\n"
            "1) 도구 사용 원칙:\n"
            "   - search_knowledge_base: 사내 지식 베이스(Chroma DB)에서 관련 문서를 검색할 때 사용\n"
            "   - get_attached_file: 사용자가 업로드한 파일의 원본 내용을 읽을 때 사용\n"
            "   - 두 도구를 순차적으로 사용하는 것이 허용됩니다.\n"
            + file_hint
            + "3) 도구 사용 판단 원칙:\n"
            "   - 자기소개, 서비스 안내, 사용 방법 등 AI 어시스턴트 자신에 관한 질문은 도구 없이 직접 답하세요.\n"
            "   - 사내 정보·문서가 필요한 질문은 search_knowledge_base를 먼저 호출하세요.\n"
            "   - 검색 결과가 없는 경우에만 '관련 사내 문서를 찾을 수 없습니다'라고 답하세요.\n"
            "   - search_knowledge_base 호출 시, 현재 질문이 모호하거나 문서명·주제가 명시되지 않은 경우\n"
            "     이전 대화 히스토리에서 언급된 문서명·주제를 쿼리에 포함하여 구체화하세요.\n"
            "     예) 이전 대화에서 'Claude Code 인사이트 보고서'를 논의 중이었다면,\n"
            "         'PR 리뷰는 몇 개?'라는 질문은 'Claude Code 인사이트 보고서 PR 리뷰 세션 수'로 쿼리를 구성하세요.\n"
            "4) 범위 외 질문 처리 원칙:\n"
            "   - 날씨, 주식, 스포츠 결과, 개인 일정 등 사내 업무와 무관한 질문은 도구를 호출하지 말고,\n"
            "     '해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요.'라고 안내하세요.\n"
            "5) 인용 원칙 (매우 중요):\n"
            "   - 도구에서 가져온 내용은 가능한 한 원문 표현을 그대로 유지하세요.\n"
            "   - 임의로 요약하거나 재구성하지 마세요. 원문에 있는 수치, 날짜, 고유명사는 반드시 원문 그대로 사용하세요.\n"
            "   - 문장 끝에 [1], [2] 형식으로 출처를 표시하세요.\n"
            "6) 정중한 비즈니스 어투로 답하고, 가독성을 위해 불렛(•)을 활용하세요.\n"
        )

        tools = [search_knowledge_base, get_attached_file]
        agent = create_react_agent(get_llm(), tools)

        agent_input = {
            "messages": [SystemMessage(content=system_content)]
            + chat_history
            + [HumanMessage(content=user_input)]
        }

        result = agent.invoke(agent_input)
        final_message = result["messages"][-1]

        response_len = len(str(final_message.content))
        print(f"DEBUG: [Chat Agent] 응답 생성 완료 ({response_len}자)")
        trace_buffer.push(trace_id, node="chat_agent", event="exit", label="execute",
                          data={"response_len": response_len})

        return {
            **state,
            "messages": [HumanMessage(content=user_input), final_message],
            "citations_used": [],
        }

    g = StateGraph(GraphState)
    g.add_node("chat_agent", chat_agent_node)
    g.set_entry_point("chat_agent")
    g.add_edge("chat_agent", END)
    return g.compile()