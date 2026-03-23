from __future__ import annotations

import os
import re

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
from app.checkpointer.dynamo_checkpointer import DynamoDBCheckpointer, ensure_checkpoints_table

from app.core.config import has_gemini_api_key
from app.graph.nodes.task_router import task_router_node, route_by_task
from app.graph.states.state import GraphState
from app.knowledge.ingest import auto_ingest_if_enabled
from app.graph.subgraphs.chat_graph import build_chat_subgraph
from app.graph.subgraphs.file_graph import build_file_subgraph
from app.graph.subgraphs.email_graph import build_email_subgraph
from app.graph.subgraphs.rfp_graph import build_rfp_subgraph

# Auth (DynamoDB)
import uuid

from app.auth.deps import get_current_user, require_approved_user
from app.auth.dynamo import ensure_admin_user, ensure_users_table_if_enabled
from app.auth.routes import router as auth_router, set_templates
from app.auth.security import hash_password
from app.auth.routing_log import ensure_routing_log_table, save_routing_log
from app.core import trace_buffer as _trace_buffer
from app.auth.intent_samples import ensure_intent_samples_table, seed_intent_samples


# =========================
# LangGraph 구성 (Subgraph 기반)
# =========================
memory = DynamoDBCheckpointer()

workflow = StateGraph(GraphState)

# 1) Router
workflow.add_node("task_router", task_router_node)

# 2) Subgraphs (각 서브그래프를 runnable로 등록)
workflow.add_node("chat_subgraph", build_chat_subgraph())
workflow.add_node("file_subgraph", build_file_subgraph())
workflow.add_node("email_subgraph", build_email_subgraph())
workflow.add_node("rfp_subgraph", build_rfp_subgraph())

workflow.set_entry_point("task_router")

# 3) Router -> task_type에 따라 subgraph로 분기
workflow.add_conditional_edges(
    "task_router",
    route_by_task,  # 분기 함수를 그대로 사용
    {
        "chat": "chat_subgraph",
        "file_extract": "file_subgraph",
        "email_draft": "email_subgraph",
        "rfp_draft": "rfp_subgraph",
    },
)

# 4) 각 subgraph 완료 후 END로 연결
workflow.add_edge("chat_subgraph", END)
workflow.add_edge("file_subgraph", END)
workflow.add_edge("email_subgraph", END)
workflow.add_edge("rfp_subgraph", END)

graph_app = workflow.compile(checkpointer=memory)

# =========================
# 템플릿 / 정적 파일
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# 초기 관리자 자동 생성 (DynamoDB)
# =========================
def ensure_initial_admin() -> None:
    """환경변수로 초기 관리자 계정을 DynamoDB에 자동 생성/보정합니다.

    - ADMIN_EMAIL: 필수
    - ADMIN_PASSWORD: 필수
    - ADMIN_NAME: 선택 (기본값 'Admin')

    동작:
    - 해당 계정이 없으면 생성 (approved=True, is_admin=True)
    - 이미 존재하면 approved/is_admin을 True로 보정 (비밀번호는 변경하지 않음)
    """

    admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    admin_name = (os.getenv("ADMIN_NAME", "Admin") or "Admin").strip()

    if not admin_email or not admin_password:
        return

    # bcrypt 72 bytes 제한으로 인해 해시가 실패할 수 있어서 미리 검증
    try:
        pw_hash = hash_password(admin_password)
    except Exception as e:
        print(f"[INIT_ADMIN] SKIP: invalid ADMIN_PASSWORD ({e})")
        return

    try:
        ensure_admin_user(email=admin_email, name=admin_name, password_hash=pw_hash)
        print(f"[INIT_ADMIN] OK: {admin_email}")
    except Exception as e:
        # DynamoDB 연결 불가 등 예외 시 요청 흐름에 영향 없도록 무시
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
    max_age=int(os.getenv("SESSION_MAX_AGE", "28800")),  # 기본 8시간
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.state.templates = templates

# auth 라우터에 템플릿 주입 (routes.py가 set_templates 호출)
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


# Chat Page (메인 사용자 화면)
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
    checkpointer.delete(thread_id)
    return {"ok": True}


# =========================
# Chat Endpoint (메인 사용자 화면)
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

    # -------------------------
    # 1) task_hint 계산 (LLM 호출 및 thread_id 결정용)
    # -------------------------

    # 체크포인트에서 file_context 존재 여부를 미리 확인 (unknown → chat fallback 판단용)
    _base_thread_early = str(user.get("user_id") or user.get("email"))
    _scope_early = (os.getenv("THREAD_CONTEXT_SCOPE", "user") or "user").strip().lower()
    _early_thread_id = (
        f"{_base_thread_early}:chat" if _scope_early == "task" else _base_thread_early
    )
    _file_context_in_checkpoint = False
    try:
        _snap = graph_app.get_state({"configurable": {"thread_id": _early_thread_id}})
        _snap_values = getattr(_snap, "values", None) or {}
        _file_context_in_checkpoint = bool((_snap_values.get("file_context") or "").strip())
    except Exception:
        pass

    trace_id = str(uuid.uuid4())

    preview = task_router_node({
        "trace_id": trace_id,
        "input_data": user_input,
        "task_type": "",
        "task_args": {"_trace_label": "preview"},
        "messages": [],
        "draft_email": None,
        "draft_rfp": None,
        "extracted_text": None,
        "extracted_meta": None,
        "citations": [],
        "citations_used": [],
    })
    effective_task = (preview.get("task_type") or "chat").strip() or "chat"

    # unknown + 파일 컨텍스트 존재 시 chat으로 fallback
    if effective_task == "unknown" and _file_context_in_checkpoint:
        effective_task = "chat"

    # unknown → chat으로 fallback (범위 외 판단은 chat 에이전트에 위임)
    if effective_task == "unknown":
        effective_task = "chat"

    # LLM이 필요한 작업만 API 키 점검
    if effective_task in {"chat", "email_draft", "rfp_draft"}:
        _ensure_llm_ready_or_503()

    # -------------------------
    # 2) thread_id 계산
    #    - 기본: 사용자 단일 컨텍스트 윈도우 (task 간 맥락 공유)
    #    - 옵션: THREAD_CONTEXT_SCOPE=task 이면 기존처럼 task별 분리
    # -------------------------
    base_thread = str(user.get("user_id") or user.get("email"))
    scope = (os.getenv("THREAD_CONTEXT_SCOPE", "user") or "user").strip().lower()
    if scope == "task":
        thread_id = f"{base_thread}:{effective_task}"
    else:
        thread_id = base_thread
    config = {"configurable": {"thread_id": thread_id}}

    # 이전 task와 달라지는 경우, 산출물 상태만 리셋해 task 간 오염을 방지
    previous_task = None
    try:
        snap = graph_app.get_state(config)
        prev_values = getattr(snap, "values", None) or {}
        previous_task = (prev_values.get("task_type") or "").strip() or None
    except Exception:
        previous_task = None

    # -------------------------
    # 3) 새 메시지마다 초기값 + 사용자 입력 주입
    # -------------------------
    preview_debug = (preview.get("task_args") or {}).get("routing_debug", {})
    inputs = {
        "trace_id": trace_id,
        "input_data": user_input,
        "task_type": "",
        "task_args": {
            "_trace_label": "execute",
            "task_type": effective_task,
            "routing_debug": preview_debug,
        },
    }

    if previous_task and previous_task != effective_task:
        inputs.update(
            {
                "draft_email": None,
                "draft_rfp": None,
                "extracted_text": None,
                "extracted_meta": None,
                "citations": [],
                "citations_used": [],
            }
        )

    # -------------------------
    # 4) 그래프 실행
    # -------------------------
    result = graph_app.invoke(inputs, config=config)

    # 라우팅 로그 저장 (execute 기준 — 실제 실행된 라우팅 데이터)
    save_routing_log(
        user_id=str(user.get("user_id") or user.get("email")),
        input_text=user_input,
        final_task=(result.get("task_type") or effective_task),
        routing_debug=(result.get("task_args") or {}).get("routing_debug", {}),
    )

    # -------------------------
    # 5) 응답의 "최종 결정"값으로 task_type 재확정 (안전장치)
    # -------------------------
    task_type = (result.get("task_type") or "").strip() or effective_task

    if task_type == "file_extract":
        text = (result.get("extracted_text") or "")[:20000]
        return {
            "type": "file_extract",
            "meta": result.get("extracted_meta"),
            "text": text,
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
            f"{draft.get('body','')}"
        )
        return {
            "type": "email_draft",
            "draft": draft,
            "answer": preview,
            "sources": [],
        }

    if task_type == "rfp_draft":
        draft = result.get("draft_rfp") or ""
        return {
            "type": "rfp_draft",
            "draft": draft,
            "answer": "RFP 초안이 생성되었습니다.",
            "sources": [],
        }

    # -------------------------
    # 6) 기본 chat 응답
    # -------------------------
    raw_answer = result.get("messages", [])[-1].content if result.get("messages") else ""

    if isinstance(raw_answer, list) and len(raw_answer) > 0:
        final_text = raw_answer[0].get("text", "응답을 처리할 수 없습니다.")
    else:
        final_text = str(raw_answer)

    def _extract_cited_ids(text: str) -> set[int]:
        return {int(x) for x in re.findall(r"\[(\d{1,3})\]", text or "")}

    cited_ids = _extract_cited_ids(final_text)
    all_sources = result.get("citations_used") or result.get("citations") or []

    filtered_sources = [
        s for s in all_sources
        if isinstance(s, dict) and isinstance(s.get("id"), int) and s.get("id") in cited_ids
    ]

    return {
        "type": "chat",
        "answer": final_text,
        "sources": filtered_sources or [],
    }


# =========================
# 파일 업로드 / 파일 QA
# =========================
_UPLOAD_ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}
_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20MB
_FILE_CONTEXT_MAX_CHARS = int(os.getenv("FILE_CONTEXT_MAX_CHARS", "8000"))

# 확장자 → 허용 매직 바이트 (오프셋, 바이트열) 목록
_MAGIC_SIGNATURES: dict[str, list[tuple[int, bytes]]] = {
    ".pdf":  [(0, b"%PDF")],
    ".docx": [(0, b"PK\x03\x04")],
    ".xlsx": [(0, b"PK\x03\x04")],
    ".xlsm": [(0, b"PK\x03\x04")],
    ".pptx": [(0, b"PK\x03\x04")],
    ".txt":  [],   # 텍스트는 매직 바이트 없음 → 검사 생략
    ".md":   [],
}


def _verify_mime(content: bytes, suffix: str) -> bool:
    """파일 앞부분 매직 바이트가 확장자와 일치하는지 확인합니다."""
    sigs = _MAGIC_SIGNATURES.get(suffix, [])
    if not sigs:
        return True
    return any(content[off: off + len(magic)] == magic for off, magic in sigs)


def _get_chat_thread_config(user: dict) -> dict:
    """사용자의 chat 세션 thread config 반환 (file_context 저장 시 사용)."""
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

            # JSON 블록 추출 (```json ... ``` 감싸진 경우 대응)
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

    # 파일 컨텍스트 + 검색 키워드를 사용자 thread State에 저장
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

    # 키워드 추출 (QA용 LLM 호출과 별개로 소형 호출)
    kw_list: list[str] = []
    try:
        import json as _json
        from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM
        kw_resp = get_llm().invoke([
            _SM(content=(
                "아래 문서에서 내부 문서 DB 검색에 활용할 핵심 개념·용어를 10개 이내로 추출하세요.\n"
                "반드시 JSON 배열 형식으로만 응답하세요. 예: [\"키워드1\", \"키워드2\"]"
            )),
            _HM(content=raw_text.strip()[:_FILE_CONTEXT_MAX_CHARS]),
        ])
        kw_raw = kw_resp.content
        if isinstance(kw_raw, list):
            kw_raw = " ".join(p.get("text", "") for p in kw_raw if isinstance(p, dict))
        kw_raw = str(kw_raw).strip()
        if kw_raw.startswith("```"):
            kw_raw = kw_raw.split("```")[1]
            if kw_raw.startswith("json"):
                kw_raw = kw_raw[4:]
            kw_raw = kw_raw.strip()
        kw_list = [str(k) for k in _json.loads(kw_raw) if k]
    except Exception as e:
        print(f"[CHAT-WITH-FILE] 키워드 추출 실패 (non-fatal): {e}")

    # 파일 컨텍스트 + 검색 키워드를 사용자 thread State에 저장
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
    from app.auth.deps import require_admin_user
    user = get_current_user(request)
    require_admin_user(user)
    from datetime import datetime
    traces = _trace_buffer.get_recent_traces(50)
    for trace in traces:
        ts = trace.get("ts_start", 0)
        trace["ts_str"] = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S") if ts else ""
    return traces


# favicon 404 방지 (선택)
@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)