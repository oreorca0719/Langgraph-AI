"""
LLM 2차 의도 분류기.

semantic router가 'unknown'을 반환한 경우에만 실행됩니다.
LLM_FALLBACK_ENABLED=0 으로 비활성화 가능.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_llm

_ALLOWED = {"chat", "email_draft", "file_extract", "rfp_draft"}
_CONFIDENCE_MIN = 0.7  # 이 값 미만이면 unknown으로 처리

_PROMPT = ChatPromptTemplate.from_template(
    "당신은 사용자 메시지의 의도를 분류하는 분류기입니다.\n"
    "아래 태스크 중 하나를 선택하고, 확신도(0.0~1.0)를 함께 반환하세요.\n\n"
    "태스크 종류:\n"
    "- chat: 인사, 일상 대화, 시스템 소개 질문, 일반 질문, 정보 조회, 문서 요약\n"
    "- email_draft: 이메일·메일 작성/수정 요청\n"
    "- file_extract: PDF·DOCX 등 파일에서 텍스트 추출\n"
    "- rfp_draft: RFP(제안요청서) 문서 작성\n"
    "- unknown: 위 태스크와 무관하거나 의도를 파악할 수 없는 경우\n\n"
    "규칙:\n"
    "- 확신도가 0.7 미만이면 반드시 unknown을 반환하세요.\n"
    "- 짧거나 대명사만 있는 입력('이거', '저거', '그거', '해줘')은 unknown으로 처리하세요.\n\n"
    "반드시 아래 형식으로만 출력:\n"
    "{{\"task\": \"<태스크>\", \"confidence\": <0.0~1.0>, \"reason\": \"<한 줄 이유>\"}}\n\n"
    "사용자 메시지: {user_input}"
)


def llm_intent_fallback(user_input: str) -> Tuple[str, Dict[str, Any]]:
    """
    LLM으로 의도를 재분류합니다.

    Returns:
        (task, debug_dict) — task는 _ALLOWED 중 하나 또는 'unknown'
    """
    enabled = (os.getenv("LLM_FALLBACK_ENABLED", "1") or "1").strip()
    if enabled in ("0", "false", "False", "NO", "no"):
        return "unknown", {"invoked": False, "reason": "disabled"}

    timeout_sec = float(os.getenv("LLM_FALLBACK_TIMEOUT_SEC", "8"))

    def _call() -> Dict[str, Any]:
        chain = _PROMPT | get_llm() | JsonOutputParser()
        return chain.invoke({"user_input": user_input}) or {}

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            result = future.result(timeout=timeout_sec)

        raw_task = (result.get("task") or "").strip().lower()
        confidence = float(result.get("confidence", 0.0))

        if raw_task not in _ALLOWED or confidence < _CONFIDENCE_MIN:
            task = "unknown"
        else:
            task = raw_task

        debug: Dict[str, Any] = {
            "invoked": True,
            "source": "llm_fallback",
            "raw_task": raw_task,
            "confidence": confidence,
            "final_task": task,
            "reason": result.get("reason", ""),
        }
        return task, debug

    except FuturesTimeoutError:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": "timeout"}
    except Exception as e:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": str(e)}