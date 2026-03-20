from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from datetime import datetime
from typing import Optional

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_NAME_RE  = re.compile(r"^[가-힣a-zA-Z]{2,20}$")

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from jinja2 import TemplateNotFound
from slowapi import Limiter
from slowapi.util import get_remote_address

from .dynamo import (
    approve_user,
    create_user_if_not_exists,
    get_user_by_email,
    list_users,
    update_login_timestamp,
    set_admin,
    delete_user,
    set_department,
)
from .deps import get_current_user, require_admin_user
from .security import hash_password, verify_password
from .routing_log import scan_recent_logs, delete_routing_log, delete_all_routing_logs
from .intent_samples import add_sample, load_all_samples

router = APIRouter()
_limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")

_templates = None


# ── CSRF 헬퍼 ────────────────────────────────────────────

def _get_csrf_token(request: Request) -> str:
    """세션에서 CSRF 토큰을 가져오거나 신규 생성합니다."""
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        request.session["csrf_token"] = token
    return token


def _verify_csrf(request: Request, form_token: str) -> None:
    """Form에서 전달된 토큰이 세션 토큰과 일치하지 않으면 403을 반환합니다."""
    session_token = request.session.get("csrf_token") or ""
    if not secrets.compare_digest(session_token, form_token or ""):
        raise HTTPException(status_code=403, detail="CSRF 토큰이 유효하지 않습니다.")


def set_templates(templates):
    global _templates
    _templates = templates


def _render(request: Request, name: str, context: dict) -> HTMLResponse:
    if _templates is None:
        return HTMLResponse("templates not set", status_code=500)
    try:
        return _templates.TemplateResponse(name, {"request": request, **context})
    except TemplateNotFound:
        return HTMLResponse(f"Template not found: {name}", status_code=500)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _render(request, "login.html", {"error": None})


@router.post("/login")
@_limiter.limit(os.getenv("LOGIN_RATE_LIMIT", "10/minute"))
def login_action(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: Optional[str] = Form(None),
):
    email_l = email.strip().lower()

    if not _EMAIL_RE.match(email_l):
        return _render(request, "login.html", {"error": "올바른 이메일 형식을 입력해 주세요."})

    if name and not _NAME_RE.match(name.strip()):
        return _render(request, "login.html", {"error": "이름은 한글 또는 영문 2~20자만 입력 가능합니다."})

    user = get_user_by_email(email_l)

    if user is None:
        pw_hash = hash_password(password)
        user = create_user_if_not_exists(email_l, name or "", pw_hash)
    else:
        if not verify_password(password, user.get("password_hash", "")):
            return _render(request, "login.html", {"error": "이메일 또는 비밀번호가 올바르지 않습니다."})

    update_login_timestamp(email_l)

    request.session["user"] = {
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "name": user.get("name"),
        "approved": bool(user.get("approved")),
        "is_admin": bool(user.get("is_admin")),
    }

    if not user.get("approved"):
        return RedirectResponse(url="/pending", status_code=303)

    return RedirectResponse(url="/", status_code=303)


@router.get("/pending", response_class=HTMLResponse)
def pending_page(request: Request):
    user = request.session.get("user")
    return _render(request, "pending.html", {"user": user})


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@router.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    users = list_users(limit=300)
    users.sort(key=lambda x: (bool(x.get("approved", False)), x.get("email", "")))
    csrf_token = _get_csrf_token(request)
    return _render(request, "admin_users.html", {"user": user, "users": users, "csrf_token": csrf_token})


@router.post("/admin/users/approve")
def admin_approve(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    _verify_csrf(request, csrf_token)
    approve_user(email=email, approved=True)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/reject")
def admin_reject(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    _verify_csrf(request, csrf_token)
    if user.get("email") == email.strip().lower():
        return RedirectResponse(url="/admin/users", status_code=303)
    approve_user(email=email, approved=False)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/toggle-admin")
def admin_toggle_admin(
    request: Request,
    email: str = Form(...),
    make_admin: str = Form(...),
    csrf_token: str = Form(...),
):
    user = get_current_user(request)
    require_admin_user(user)
    _verify_csrf(request, csrf_token)
    target_email = email.strip().lower()
    if user.get("email") == target_email and make_admin.strip() != "1":
        return RedirectResponse(url="/admin/users", status_code=303)
    set_admin(target_email, is_admin=(make_admin.strip() == "1"))
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/set-department")
def admin_set_department(
    request: Request,
    email: str = Form(...),
    department: str = Form(...),
    csrf_token: str = Form(...),
):
    user = get_current_user(request)
    require_admin_user(user)
    _verify_csrf(request, csrf_token)
    set_department(email=email, department=department)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/delete")
def admin_delete(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    _verify_csrf(request, csrf_token)
    target_email = email.strip().lower()
    if user.get("email") == target_email:
        return RedirectResponse(url="/admin/users", status_code=303)
    delete_user(target_email)
    return RedirectResponse(url="/admin/users", status_code=303)


# ── 관리자 홈 ────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    return _render(request, "admin_home.html", {"user": user})


# ── 모니터 대시보드 ─────────────────────────────────────

@router.get("/admin/monitor", response_class=HTMLResponse)
def admin_monitor(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    return _render(request, "admin_monitor.html", {"user": user})


@router.get("/admin/api/routing-logs")
def admin_routing_logs_api(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    logs = scan_recent_logs(limit=50)
    for log in logs:
        ts = log.get("timestamp", 0)
        log["ts_str"] = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S") if ts else ""
    return logs



@router.delete("/admin/api/routing-logs/{log_id}")
def admin_delete_routing_log(log_id: str, request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    ok = delete_routing_log(log_id)
    return {"ok": ok}


@router.delete("/admin/api/routing-logs")
def admin_delete_all_routing_logs(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    count = delete_all_routing_logs()
    return {"ok": True, "deleted": count}


@router.get("/admin/api/routing-stats")
def admin_routing_stats_api(request: Request):
    user = get_current_user(request)
    require_admin_user(user)

    logs = scan_recent_logs(limit=500)

    source_counts: dict = {}
    task_counts: dict = {}
    score_sum = 0.0
    score_cnt = 0
    margin_sum = 0.0
    margin_cnt = 0
    unknown_cases: list = []

    for log in logs:
        # final_source 집계: routing_debug JSON 파싱
        debug_raw = log.get("routing_debug") or "{}"
        try:
            debug = json.loads(debug_raw) if isinstance(debug_raw, str) else debug_raw
        except Exception:
            debug = {}

        final_source = debug.get("final_source") or log.get("mode") or "unknown"
        source_counts[final_source] = source_counts.get(final_source, 0) + 1

        # task 분포
        task = log.get("final_task") or "unknown"
        task_counts[task] = task_counts.get(task, 0) + 1

        # 평균 score / margin
        try:
            s = float(log.get("top1_score") or 0)
            if s > 0:
                score_sum += s
                score_cnt += 1
        except Exception:
            pass
        try:
            m = float(log.get("margin") or 0)
            if m >= 0:
                margin_sum += m
                margin_cnt += 1
        except Exception:
            pass

        # unknown 케이스 목록
        if task == "unknown" or final_source in ("unknown", "semantic_unknown"):
            ts = log.get("timestamp", 0)
            unknown_cases.append({
                "log_id": log.get("log_id", ""),
                "ts_str": datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S") if ts else "",
                "input_text": log.get("input_text", ""),
            })

    total = len(logs)
    return {
        "total": total,
        "source_counts": source_counts,
        "task_counts": task_counts,
        "avg_top1_score": round(score_sum / score_cnt, 4) if score_cnt else None,
        "avg_margin": round(margin_sum / margin_cnt, 4) if margin_cnt else None,
        "unknown_cases": unknown_cases[:50],
    }


@router.post("/admin/api/intent-samples")
async def admin_add_intent_sample(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    body = await request.json()
    task = (body.get("task") or "").strip()
    text = (body.get("text") or "").strip()
    if not task or not text:
        raise HTTPException(status_code=400, detail="task와 text가 필요합니다.")
    from app.graph.nodes.task_router import invalidate_sample_cache
    added = add_sample(task, text, source="admin")
    if added:
        invalidate_sample_cache()
    return {"ok": True, "added": added}


@router.post("/admin/api/reingest")
def admin_reingest(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    try:
        from app.knowledge.ingest import auto_ingest_if_enabled
        auto_ingest_if_enabled()
        return {"ok": True}
    except Exception as e:
        import traceback
        msg = f"{type(e).__name__}: {e}" or traceback.format_exc()
        print(f"[REINGEST] 오류: {msg}")
        return {"ok": False, "error": msg}


@router.get("/admin/logs/stream")
async def admin_logs_stream(request: Request):
    user = get_current_user(request)
    require_admin_user(user)

    from app.core import log_buffer

    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    log_buffer.subscribe(queue)

    async def event_generator():
        # 기존 버퍼 먼저 전송
        for record in log_buffer.get_recent(200):
            payload = json.dumps(record, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=15.0)
                    payload = json.dumps(record, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"ts\":\"\",\"level\":\"PING\",\"msg\":\"\"}\n\n"
        finally:
            log_buffer.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )