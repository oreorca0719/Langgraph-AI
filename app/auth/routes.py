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

_READONLY_EMAIL = "testuser@test.co.kr"


def _is_readonly(user: dict) -> bool:
    return (user.get("email") or "").strip().lower() == _READONLY_EMAIL

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
from .intent_samples import add_sample, load_all_samples, reset_seed_samples

router = APIRouter()
_limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")

_templates = None
_graph_app = None


def set_graph_app(app) -> None:
    global _graph_app
    _graph_app = app


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
        return _templates.TemplateResponse(request, name, {"request": request, **context})
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
    return _render(request, "admin_users.html", {"user": user, "users": users, "csrf_token": csrf_token, "read_only": _is_readonly(user)})


@router.post("/admin/users/approve")
def admin_approve(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    approve_user(email=email, approved=True)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/reject")
def admin_reject(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
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
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
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
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    set_department(email=email, department=department)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/delete")
def admin_delete(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
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
    return _render(request, "admin_home.html", {"user": user, "read_only": _is_readonly(user)})


