from __future__ import annotations

import os
import re
import uuid

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

from app.core import log_buffer as _log_buffer
_log_buffer.setup()

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langgraph.graph import END, StateGraph
from langgraph.types import Command
from app.checkpointer.dynamo_checkpointer import DynamoDBCheckpointer, ensure_checkpoints_table

from app.core.config import has_gemini_api_key
from app.graph.states.state import GraphState
from app.graph.nodes.input_guard import input_guard_node
from app.graph.nodes.task_router import task_router_node, route_by_task, rejection_node, route_after_input_guard
from app.graph.nodes.clarification import clarification_node, route_after_clarification
from app.graph.nodes.human_review import human_review_node, route_after_review
from app.graph.nodes.knowledge_search import (
    search_node, quality_check_node, route_after_quality, rewrite_node, answer_node,
)
from app.graph.nodes.detail_search import detail_search_node
from app.graph.nodes.ai_guide import ai_guide_node
from app.graph.nodes.file_chat import file_chat_node
from app.graph.nodes.file_extractor import file_extractor_node
from app.graph.nodes.email_agent import email_draft_node
from app.graph.nodes.rfp_agent import (
    rfp_research_node, rfp_draft_node, rfp_review_node, route_after_rfp_review,
)
from app.graph.nodes.planner import planner_node, route_after_planner
from app.security.content_sanitizer import sanitize as sanitize_content
from app.security.output_validator import validate as validate_output
from app.knowledge.ingest import auto_ingest_if_enabled

# Auth (DynamoDB)
from app.auth.deps import get_current_user, require_approved_user, require_admin_user
from app.auth.dynamo import ensure_admin_user, ensure_users_table_if_enabled
from app.auth.routes import router as auth_router, set_templates, set_graph_app
from app.auth.security import hash_password
from app.auth.routing_log import ensure_routing_log_table, save_routing_log
from app.core import trace_buffer as _trace_buffer
from app.auth.intent_samples import ensure_intent_samples_table, seed_intent_samples


# =========================
# LangGraph 구성 (플랫 그래프)
# =========================
memory = DynamoDBCheckpointer()

workflow = StateGraph(GraphState)

# ── 노드 등록 ────────────────────────────────────────────────
workflow.add_node("input_guard",   input_guard_node)
workflow.add_node("task_router",   task_router_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("rejection",     rejection_node)
workflow.add_node("search",        search_node)
workflow.add_node("quality_check", quality_check_node)
workflow.add_node("rewrite",       rewrite_node)
workflow.add_node("answer",        answer_node)
workflow.add_node("ai_guide",      ai_guide_node)
workflow.add_node("file_chat",     file_chat_node)
workflow.add_node("file_extract",  file_extractor_node)
workflow.add_node("email_draft",   email_draft_node)
workflow.add_node("human_review",  human_review_node)
workflow.add_node("rfp_research",  rfp_research_node)
workflow.add_node("rfp_draft",     rfp_draft_node)
workflow.add_node("rfp_review",    rfp_review_node)
workflow.add_node("detail_search", detail_search_node)
workflow.add_node("planner",       planner_node)

workflow.set_entry_point("input_guard")

# ── 조건부 엣지 ──────────────────────────────────────────────
workflow.add_conditional_edges(
    "input_guard",
    route_after_input_guard,
    {"rejection": "rejection", "task_router": "task_router"},
)

workflow.add_conditional_edges(
    "task_router",
    route_by_task,
    {
        "knowledge_search": "search",
        "detail_search":    "detail_search",
        "ai_guide":         "ai_guide",
        "file_chat":        "file_chat",
        "file_extract":     "file_extract",
        "email_draft":      "email_draft",
        "rfp_draft":        "rfp_research",
        "planner":          "planner",
        "rejection":        "rejection",
        "clarification":    "clarification",
    },
)

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge_search": "search",
        "rfp_draft":        "rfp_research",
        "email_draft":      "email_draft",
        "ai_guide":         "ai_guide",
        "detail_search":    "detail_search",
        "file_chat":        "file_chat",
    },
)

# clarification → rejection or task_router or knowledge_search (루프 방어 fallback)
workflow.add_conditional_edges(
    "clarification",
    route_after_clarification,
    {
        "rejection":        "rejection",
        "task_router":      "task_router",
        "knowledge_search": "search",  # 루프 방어 fallback 경로
    },
)

# knowledge search 순환: search → quality_check → (answer | rewrite → search)
workflow.add_edge("search", "quality_check")
workflow.add_conditional_edges(
    "quality_check",
    route_after_quality,
    {"answer": "answer", "rewrite": "rewrite"},
)
workflow.add_edge("rewrite", "search")

# RFP 3-에이전트 체인: research → draft → review → (rfp_draft | human_review)
workflow.add_edge("rfp_research", "rfp_draft")
workflow.add_edge("rfp_draft", "rfp_review")
workflow.add_conditional_edges(
    "rfp_review",
    route_after_rfp_review,
    {"rfp_draft": "rfp_draft", "human_review": "human_review"},
)

# email/rfp → human_review → (end | task_router | email_draft | rfp_draft)
workflow.add_edge("email_draft", "human_review")
workflow.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "end":        END,
        "task_router": "task_router",
        "email_draft": "email_draft",
        "rfp_draft":   "rfp_draft",
    },
)

# ── 종료 엣지 ────────────────────────────────────────────────
workflow.add_edge("detail_search", "answer")
workflow.add_edge("answer",      END)
workflow.add_edge("ai_guide",    END)
workflow.add_edge("file_chat",   END)
workflow.add_edge("file_extract", END)
workflow.add_edge("rejection",   END)

graph_app = workflow.compile(checkpointer=memory)
set_graph_app(graph_app)


# =========================
# 템플릿 / 정적 파일
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# 초기 관리자 자동 생성 (DynamoDB)
# =========================
def ensure_initial_admin() -> None:
    """환경변수로 초기 관리자 계정을 DynamoDB에 자동 생성/보정합니다."""
    admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    admin_name = (os.getenv("ADMIN_NAME", "Admin") or "Admin").strip()

    if not admin_email or not admin_password:
        return

    try:
        pw_hash = hash_password(admin_password)
    except Exception as e:
        print(f"[INIT_ADMIN] SKIP: invalid ADMIN_PASSWORD ({e})")
        return

    try:
        ensure_admin_user(email=admin_email, name=admin_name, password_hash=pw_hash)
        print(f"[INIT_ADMIN] OK: {admin_email}")
    except Exception as e:
        print(f"[INIT_ADMIN] SKIP: ensure_admin_user failed ({e})")


# =========================
# Lifespan (startup/shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _secret = os.getenv("SESSION_SECRET", "")
    if not _secret or _secret == "dev-secret-change-me":
        raise RuntimeError(
            "[SECURITY] SESSION_SECRET 환경변수가 설정되지 않았거나 기본값입니다. "
            "안전한 무작위 문자열로 설정 후 서버를 재시작하세요."
        )
    ensure_users_table_if_enabled()
    ensure_routing_log_table()
    ensure_intent_samples_table()
    ensure_checkpoints_table()
    seed_intent_samples()
    ensure_initial_admin()
    auto_ingest_if_enabled()
    yield


# =========================
# FastAPI 앱 구성
# =========================
limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-secret-change-me"),
    max_age=int(os.getenv("SESSION_MAX_AGE", "28800")),
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.state.templates = templates

set_templates(templates)
app.include_router(auth_router)


# =========================
# Health / Status
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    api_key_present = has_gemini_api_key()
    return {
        "ok": True,
        "llm_ready": api_key_present,
        "reason": None if api_key_present else "missing_api_key",
        "auto_ingest": os.getenv("AUTO_INGEST", "1"),
        "knowledge_dir": os.getenv("KNOWLEDGE_DIR", "./knowledge_data"),
        "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "my_knowledge"),
        "users_table": os.getenv("USERS_TABLE", "langgraph_users"),
        "aws_region": os.getenv("AWS_REGION", "ap-northeast-1"),
        "thread_context_scope": os.getenv("THREAD_CONTEXT_SCOPE", "user"),
    }


def _ensure_llm_ready_or_503():
    if not has_gemini_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM API 키가 설정되지 않아 요청을 처리할 수 없습니다. "
                "GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정해 주세요."
            ),
        )


# =========================
# Home Page
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        user = get_current_user(request)
        require_approved_user(user)
    except HTTPException as e:
        if e.status_code == 401:
            return RedirectResponse(url="/login", status_code=303)
        if e.status_code == 403:
            return RedirectResponse(url="/pending", status_code=303)
        raise

    from app.auth.dynamo import get_user_by_email
    from app.auth.routing_log import scan_user_recent_logs
    from datetime import datetime

    full_user = get_user_by_email(user.get("email", "")) or user
    user_id = str(full_user.get("user_id") or full_user.get("email", ""))
    recent_logs = scan_user_recent_logs(user_id, limit=5)
    for log in recent_logs:
        ts = log.get("timestamp", 0)
        log["ts_str"] = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else ""

    last_login = full_user.get("updated_at")
    last_login_str = datetime.fromtimestamp(float(last_login)).strftime("%Y-%m-%d %H:%M") if last_login else "-"

    return templates.TemplateResponse(request, "home.html", {
        "request": request,
        "user": full_user,
        "recent_logs": recent_logs,
        "last_login_str": last_login_str,
    })


# =========================
# Chat Page
# =========================
@app.get("/chat", response_class=HTMLResponse)
async def index(request: Request):
    try:
        user = get_current_user(request)
        require_approved_user(user)
    except HTTPException as e:
        if e.status_code == 401:
            return RedirectResponse(url="/login", status_code=303)
        if e.status_code == 403:
            return RedirectResponse(url="/pending", status_code=303)
        raise

    return templates.TemplateResponse(request, "index.html", {"request": request, "user": user})


# =========================
# Chat Reset Endpoint
# =========================
@app.post("/chat/reset")
async def chat_reset(request: Request):
    user = get_current_user(request)
    thread_id = str(user.get("user_id") or user.get("email"))
    memory.delete(thread_id)
    return {"ok": True}


# =========================
# Chat Endpoint (interrupt 패턴)
# =========================
@app.post("/chat")
async def chat_endpoint(request: Request):
    user = get_current_user(request)
    require_approved_user(user)

    data = await request.json()
    user_input = data.get("message", "")

    _max_input = int(os.getenv("CHAT_MAX_INPUT_CHARS", "4000"))
    if len(user_input) > _max_input:
        raise HTTPException(status_code=400, detail=f"입력이 너무 깁니다. 최대 {_max_input}자까지 허용됩니다.")

    _ensure_llm_ready_or_503()

    trace_id  = str(uuid.uuid4())
    thread_id = str(user.get("user_id") or user.get("email"))
    config    = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15,
    }

    # ── interrupt 재개 여부 확인 ──────────────────────────────
    _INTERRUPT_NODES = {"human_review", "clarification"}
    has_interrupt = False
    try:
        current_state = graph_app.get_state(config)
        if current_state:
            # 1차: tasks.interrupts 확인 (put_writes 정상 동작 시)
            has_interrupt = any(t.interrupts for t in (current_state.tasks or []))
            # 2차: state.next가 interrupt 노드 중 하나를 가리키면 interrupt 상태
            if not has_interrupt:
                has_interrupt = any(n in _INTERRUPT_NODES for n in (current_state.next or ()))
    except Exception:
        pass

    # ── 그래프 실행 ───────────────────────────────────────────
    if has_interrupt:
        result = graph_app.invoke(Command(resume=user_input), config=config)
    else:
        inputs = {
            "trace_id":   trace_id,
            "input_data": user_input,
            "task_args":  {},
        }
        result = graph_app.invoke(inputs, config=config)

    # ── interrupt 발생 확인 (invoke 후) ───────────────────────
    try:
        new_state = graph_app.get_state(config)
        pending = []
        if new_state and new_state.tasks:
            for task in new_state.tasks:
                pending.extend(task.interrupts or [])
    except Exception:
        pending = []

    if pending:
        iv = pending[0]
        interrupt_data = iv.value if hasattr(iv, "value") else iv
        if not isinstance(interrupt_data, dict):
            interrupt_data = {}
        # 구조 데이터(draft)를 함께 포함해 JS가 직접 렌더링 가능하도록 함
        current_task = (result.get("current_task") or result.get("task_type") or "").strip()
        return JSONResponse({
            "type":           "interrupt",
            "interrupt_type": interrupt_data.get("type", ""),
            "message":        interrupt_data.get("message", ""),
            "hint":           interrupt_data.get("hint", ""),
            "current_task":   current_task,
            "draft_email":    result.get("draft_email"),
            "draft_rfp":      result.get("draft_rfp") or "",
            "sources":        [],
        })

    # ── 라우팅 로그 저장 ──────────────────────────────────────
    save_routing_log(
        user_id=str(user.get("user_id") or user.get("email")),
        input_text=user_input,
        final_task=(result.get("task_type") or "unknown"),
        routing_debug=(result.get("task_args") or {}).get("routing_debug", {}),
    )

    # ── 응답 포맷팅 ───────────────────────────────────────────
    task_type = (result.get("task_type") or "").strip()

    if task_type == "file_extract":
        text = (result.get("extracted_text") or "")[:20000]
        return {
            "type":   "file_extract",
            "meta":   result.get("extracted_meta"),
            "text":   text,
            "answer": "파일 추출이 완료되었습니다.",
            "sources": [],
        }

    if task_type == "email_draft":
        draft = result.get("draft_email") or {"to": "", "cc": "", "subject": "", "body": ""}
        preview = (
            f"[이메일 초안]\n"
            f"- To: {draft.get('to','')}\n"
            f"- CC: {draft.get('cc','')}\n"
            f"- Subject: {draft.get('subject','')}\n\n"
            f"{draft.get('body','')}\n\n"
            f"---\n수정이 필요하시면 '제목 바꿔줘', '수신자 변경해줘' 등으로 말씀해 주세요."
        )
        return {
            "type":   "email_draft",
            "draft":  draft,
            "answer": preview,
            "sources": [],
        }

    if task_type == "rfp_draft":
        draft = result.get("draft_rfp") or ""
        return {
            "type":   "rfp_draft",
            "draft":  draft,
            "answer": "RFP 초안이 생성되었습니다.\n\n---\n수정이 필요하시면 '일정 섹션 수정해줘' 등으로 말씀해 주세요.",
            "sources": [],
        }

    # 기본 chat 응답
    raw_answer = result.get("messages", [])[-1].content if result.get("messages") else ""

    if isinstance(raw_answer, list) and len(raw_answer) > 0:
        final_text = raw_answer[0].get("text", "응답을 처리할 수 없습니다.")
    else:
        final_text = str(raw_answer)

    _, final_text = validate_output(final_text)

    if not final_text.strip():
        final_text = "응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."

    def _extract_cited_ids(text: str) -> set[int]:
        return {int(x) for x in re.findall(r"\[(\d{1,3})\]", text or "")}

    cited_ids = _extract_cited_ids(final_text)
    all_sources = result.get("citations_used") or result.get("citations") or []
    filtered_sources = [
        s for s in all_sources
        if isinstance(s, dict) and isinstance(s.get("id"), int) and s.get("id") in cited_ids
    ]

    return {
        "type":    "chat",
        "answer":  final_text,
        "sources": filtered_sources or [],
    }


# =========================
# 파일 업로드 / 파일 QA
# =========================
_UPLOAD_ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}
_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20MB
_FILE_CONTEXT_MAX_CHARS = int(os.getenv("FILE_CONTEXT_MAX_CHARS", "8000"))

_MAGIC_SIGNATURES: dict[str, list[tuple[int, bytes]]] = {
    ".pdf":  [(0, b"%PDF")],
    ".docx": [(0, b"PK\x03\x04")],
    ".xlsx": [(0, b"PK\x03\x04")],
    ".xlsm": [(0, b"PK\x03\x04")],
    ".pptx": [(0, b"PK\x03\x04")],
    ".txt":  [],
    ".md":   [],
}


def _verify_mime(content: bytes, suffix: str) -> bool:
    sigs = _MAGIC_SIGNATURES.get(suffix, [])
    if not sigs:
        return True
    return any(content[off: off + len(magic)] == magic for off, magic in sigs)


def _get_chat_thread_config(user: dict) -> dict:
    base_thread = str(user.get("user_id") or user.get("email"))
    scope = (os.getenv("THREAD_CONTEXT_SCOPE", "user") or "user").strip().lower()
    thread_id = f"{base_thread}:chat" if scope == "task" else base_thread
    return {"configurable": {"thread_id": thread_id}}


def _extract_file_to_text(content: bytes, filename: str) -> tuple[str, dict]:
    import tempfile
    from pathlib import Path
    from app.graph.nodes.file_extractor import extract_text_from_file

    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        text, meta = extract_text_from_file(Path(tmp_path))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text, meta


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """파일만 첨부 시: 텍스트 추출 후 LLM 요약 반환."""
    user = get_current_user(request)
    require_approved_user(user)

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {suffix}")

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail="파일 크기는 20MB 이하여야 합니다.")

    if not _verify_mime(content, suffix):
        raise HTTPException(status_code=400, detail="파일 내용이 확장자와 일치하지 않습니다.")

    try:
        text, meta = _extract_file_to_text(content, file.filename or "file")
        meta["name"] = file.filename
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 추출 실패: {e}")

    text = sanitize_content(text, source=f"upload:{file.filename}")

    summary = None
    keywords: list[str] = []
    if has_gemini_api_key() and text.strip():
        try:
            import json as _json
            from app.core.config import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            context = text.strip()[:_FILE_CONTEXT_MAX_CHARS]
            llm = get_llm()
            resp = llm.invoke([
                SystemMessage(content=(
                    "당신은 문서 분석 전문가입니다.\n"
                    "아래 문서를 분석하여 반드시 다음 JSON 형식으로만 응답하세요. "
                    "JSON 외 다른 텍스트는 절대 출력하지 마세요.\n\n"
                    "{\n"
                    '  "summary": "**문서 유형**: ...\\n**핵심 내용**:\\n• ...\\n**주요 수치/일정**: ...\\n**특이사항**: ...",\n'
                    '  "keywords": ["키워드1", "키워드2", ...]\n'
                    "}\n\n"
                    "keywords: 이 문서의 내용을 내부 문서 DB에서 유사 문서를 찾는 데 활용할 핵심 개념·용어 10개 이내."
                )),
                HumanMessage(content=context),
            ])
            raw = resp.content
            if isinstance(raw, list):
                raw = " ".join(p.get("text", "") for p in raw if isinstance(p, dict))
            raw = str(raw).strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = _json.loads(raw)
            summary = str(parsed.get("summary") or "").strip() or None
            keywords = [str(k) for k in (parsed.get("keywords") or []) if k]
        except Exception as e:
            print(f"[UPLOAD] LLM 분석 실패: {type(e).__name__}: {e}")

    try:
        config = _get_chat_thread_config(user)
        graph_app.update_state(config, {
            "file_context": text.strip()[:_FILE_CONTEXT_MAX_CHARS],
            "file_context_name": file.filename,
        })
    except Exception as e:
        print(f"[UPLOAD] file_context State 저장 실패 (non-fatal): {e}")

    return JSONResponse({"ok": True, "summary": summary, "text": text[:20000], "meta": meta})


@app.post("/chat-with-file")
async def chat_with_file(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(...),
):
    """파일 + 질문: 파일 내용을 컨텍스트로 LLM 답변 반환."""
    user = get_current_user(request)
    require_approved_user(user)
    _ensure_llm_ready_or_503()

    if not message.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {suffix}")

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail="파일 크기는 20MB 이하여야 합니다.")

    if not _verify_mime(content, suffix):
        raise HTTPException(status_code=400, detail="파일 내용이 확장자와 일치하지 않습니다.")

    try:
        raw_text, _ = _extract_file_to_text(content, file.filename or "file")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 추출 실패: {e}")

    from app.core.config import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_text = sanitize_content(raw_text, source=f"chat-with-file:{file.filename}")

    file_context = raw_text.strip()
    truncated = len(file_context) > _FILE_CONTEXT_MAX_CHARS
    if truncated:
        file_context = file_context[:_FILE_CONTEXT_MAX_CHARS]

    system_prompt = (
        "당신은 사용자가 첨부한 파일 내용을 분석하는 AI 어시스턴트입니다.\n"
        "아래 [파일 내용]만을 근거로 사용자의 질문에 답하세요.\n"
        "파일에 없는 내용은 '파일에서 확인할 수 없습니다'라고 답하세요.\n"
        "정중한 비즈니스 어투로 답하고, 필요 시 불렛(•)을 활용하세요.\n\n"
        f"[파일 내용]\n{file_context}"
    )

    try:
        llm = get_llm()
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message.strip()),
        ])
        answer = resp.content
        if isinstance(answer, list):
            answer = " ".join(p.get("text", "") for p in answer if isinstance(p, dict))
        answer = str(answer)
        if truncated:
            answer += f"\n\n※ 파일이 길어 앞부분 {_FILE_CONTEXT_MAX_CHARS}자만 참조했습니다."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")

    try:
        config = _get_chat_thread_config(user)
        graph_app.update_state(config, {
            "file_context": raw_text.strip()[:_FILE_CONTEXT_MAX_CHARS],
            "file_context_name": file.filename,
        })
    except Exception as e:
        print(f"[CHAT-WITH-FILE] file_context State 저장 실패 (non-fatal): {e}")

    return JSONResponse({"type": "file_qa", "answer": answer, "sources": []})


@app.get("/admin/api/traces")
def admin_traces_api(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    from datetime import datetime
    traces = _trace_buffer.get_recent_traces(50)
    for trace in traces:
        ts = trace.get("ts_start", 0)
        trace["ts_str"] = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S") if ts else ""
    return traces


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
