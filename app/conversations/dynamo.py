"""대화 메타 테이블 (langgraph_conversations) — 사용자별 thread를 N개 관리.

체크포인트 본체는 langgraph_checkpoints에 두고, 이 테이블은 목록 조회·정렬·
20개 캡 관리를 위한 인덱스 역할만 한다. thread_id는 '{user_id}:{conversation_id}'.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from app.auth.dynamo import _region


CONVERSATIONS_TABLE = os.getenv("CONVERSATIONS_TABLE", "langgraph_conversations")
MAX_CONVERSATIONS_PER_USER = int(os.getenv("MAX_CONVERSATIONS_PER_USER", "20"))


def _table():
    return boto3.resource("dynamodb", region_name=_region()).Table(CONVERSATIONS_TABLE)


def derive_thread_id(user_id: str, conversation_id: str) -> str:
    return f"{user_id}:{conversation_id}"


def ensure_conversations_table() -> None:
    client = boto3.client("dynamodb", region_name=_region())
    try:
        client.describe_table(TableName=CONVERSATIONS_TABLE)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            print(f"[CONVERSATIONS] 테이블 확인 실패 (non-fatal): {e}")
            return

    try:
        client.create_table(
            TableName=CONVERSATIONS_TABLE,
            AttributeDefinitions=[
                {"AttributeName": "user_id",         "AttributeType": "S"},
                {"AttributeName": "conversation_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "user_id",         "KeyType": "HASH"},
                {"AttributeName": "conversation_id", "KeyType": "RANGE"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        client.get_waiter("table_exists").wait(TableName=CONVERSATIONS_TABLE)
        print(f"[CONVERSATIONS] 테이블 생성 완료: {CONVERSATIONS_TABLE}")
    except Exception as e:
        print(f"[CONVERSATIONS] 테이블 생성 실패 (non-fatal): {e}")


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    item["updated_at"]    = int(item.get("updated_at") or 0)
    item["created_at"]    = int(item.get("created_at") or 0)
    item["message_count"] = int(item.get("message_count") or 0)
    return item


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    """사용자 대화 목록을 updated_at 내림차순으로 반환 (최대 MAX 개)."""
    if not user_id:
        return []
    try:
        resp = _table().query(KeyConditionExpression=Key("user_id").eq(user_id))
        items = [_normalize_item(it) for it in resp.get("Items", [])]
        items.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return items[:MAX_CONVERSATIONS_PER_USER]
    except Exception as e:
        print(f"[CONVERSATIONS] list 실패 (non-fatal): {e}")
        return []


def get_conversation(user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    if not user_id or not conversation_id:
        return None
    try:
        resp = _table().get_item(
            Key={"user_id": user_id, "conversation_id": conversation_id}
        )
        item = resp.get("Item")
        return _normalize_item(item) if item else None
    except Exception as e:
        print(f"[CONVERSATIONS] get 실패 (non-fatal): {e}")
        return None


def create_conversation(user_id: str, title: str = "") -> str:
    """새 대화 생성. 20개 초과 시 가장 오래된 항목과 그 체크포인트를 함께 삭제."""
    cid = str(uuid.uuid4())
    now = int(time.time())
    try:
        _table().put_item(Item={
            "user_id":         user_id,
            "conversation_id": cid,
            "title":           (title or "(새 대화)")[:60],
            "created_at":      now,
            "updated_at":      now,
            "message_count":   0,
        })
    except Exception as e:
        print(f"[CONVERSATIONS] create 실패 (non-fatal): {e}")
        return cid

    _prune_oldest_if_over_limit(user_id)
    return cid


def _prune_oldest_if_over_limit(user_id: str) -> None:
    try:
        resp = _table().query(KeyConditionExpression=Key("user_id").eq(user_id))
        items = [_normalize_item(it) for it in resp.get("Items", [])]
        if len(items) <= MAX_CONVERSATIONS_PER_USER:
            return
        items.sort(key=lambda x: x.get("updated_at", 0))
        excess = items[: len(items) - MAX_CONVERSATIONS_PER_USER]
        for it in excess:
            delete_conversation(user_id, it["conversation_id"])
    except Exception as e:
        print(f"[CONVERSATIONS] prune 실패 (non-fatal): {e}")


def update_conversation(
    user_id: str,
    conversation_id: str,
    *,
    title: Optional[str] = None,
    increment_messages: int = 0,
) -> None:
    if not user_id or not conversation_id:
        return
    try:
        parts = ["updated_at = :ts"]
        values: Dict[str, Any] = {":ts": int(time.time())}
        if title is not None:
            parts.append("title = :title")
            values[":title"] = (title or "")[:60]
        if increment_messages:
            parts.append("message_count = if_not_exists(message_count, :zero) + :inc")
            values[":zero"] = 0
            values[":inc"]  = int(increment_messages)
        _table().update_item(
            Key={"user_id": user_id, "conversation_id": conversation_id},
            UpdateExpression="SET " + ", ".join(parts),
            ExpressionAttributeValues=values,
        )
    except Exception as e:
        print(f"[CONVERSATIONS] update 실패 (non-fatal): {e}")


def delete_conversation(user_id: str, conversation_id: str) -> None:
    """메타 + 체크포인트 동시 삭제."""
    if not user_id or not conversation_id:
        return
    try:
        _table().delete_item(
            Key={"user_id": user_id, "conversation_id": conversation_id}
        )
    except Exception as e:
        print(f"[CONVERSATIONS] delete (meta) 실패 (non-fatal): {e}")
    try:
        from app.checkpointer.dynamo_checkpointer import DynamoDBCheckpointer
        DynamoDBCheckpointer().delete(derive_thread_id(user_id, conversation_id))
    except Exception as e:
        print(f"[CONVERSATIONS] delete (checkpoint) 실패 (non-fatal): {e}")


def load_conversation_messages(user_id: str, conversation_id: str) -> List[Dict[str, str]]:
    """체크포인트에서 messages를 읽어 [{role, content}, ...] 형태로 반환."""
    if not user_id or not conversation_id:
        return []
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from app.checkpointer.dynamo_checkpointer import DynamoDBCheckpointer

        thread_id = derive_thread_id(user_id, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}
        tup = DynamoDBCheckpointer().get_tuple(config)
        if not tup or not tup.checkpoint:
            return []

        channel_values = tup.checkpoint.get("channel_values") or {}
        raw_messages = channel_values.get("messages") or []

        out: List[Dict[str, str]] = []
        for msg in raw_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            out.append({"role": role, "content": str(content)})
        return out
    except Exception as e:
        print(f"[CONVERSATIONS] load_messages 실패 (non-fatal): {e}")
        return []
