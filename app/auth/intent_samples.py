from __future__ import annotations

import os
import time
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

from app.auth.dynamo import _region

# INTENT_SAMPLES 원본 (시드 데이터용)
SEED_SAMPLES: Dict[str, List[str]] = {
    "chat": [
        "HR 문의 메일 알려줘",
        "IT 지원팀 연락처가 뭐야",
        "사내 계정 권한 신청은 어디로 문의해",
        "복리후생 담당 부서 알려줘",
        "인사팀이 하는 업무가 뭐야",
        "네트워크 장애 나면 누구한테 연락해",
        "소프트웨어 설치 요청은 어디로 해",
        "평가 제도 관련 문의처 알려줘",
        "채용 관련 질문은 어디에 해야 해",
        "문서 내용을 요약해줘",
        "근거 문서 기반으로 답해줘",
        "사내 지원 조직 구조를 설명해줘",
        "안녕하세요",
        "안녕",
        "반가워요",
        "넌 누구야?",
        "자기소개 해줘",
        "뭐 할 수 있어?",
        "어떤 기능이 있어?",
        "사용 방법 알려줘",
        "여긴 뭐 하는 곳이야?",
        "도움말",
    ],
    "email_draft": [
        "이메일 초안 작성해줘",
        "메일 문안 만들어줘",
        "고객사 회신 메일 작성",
        "to subject body 형식으로 이메일 써줘",
        "영문 이메일 draft 작성",
        "계정 권한 요청 메일 작성해줘",
        "사내 공지 메일 초안",
        "정중한 톤으로 이메일 작성",
        "협업 요청 이메일 써줘",
        "제목과 본문 포함 이메일 작성",
        "cc 포함 메일 초안",
        "클레임 회신 이메일 초안",
    ],
    "file_extract": [
        "파일 추출해줘 ./sample.pdf",
        "문서 텍스트 추출해줘",
        "pdf에서 텍스트 뽑아줘",
        "docx 파일 내용 추출",
        "xlsx 문서 파싱",
        "pptx 텍스트 추출",
        "file_path로 문서 읽어줘",
        "경로의 파일 내용만 추출",
        "첨부 문서 원문 텍스트 뽑아줘",
        "filepath ./a.pdf extract",
        "문서에서 글자만 뽑아줘",
        "로컬 파일 텍스트 추출",
    ],
    "rfp_draft": [
        "RFP 초안 작성해줘",
        "제안요청서 만들어줘",
        "요구사항 정의서 작성",
        "입찰용 RFP 문서 작성",
        "배경 범위 요구사항 일정 포함 RFP",
        "신규 시스템 구축 RFP",
        "기능 비기능 요구사항 포함 RFP",
        "SI 제안요청서 초안",
        "평가기준 포함 RFP 문서",
        "요구사항 정의 문서 초안",
        "프로젝트 제안요청서 작성",
        "RFP 형식으로 정리해줘",
    ],
}


def _table_name() -> str:
    return (os.getenv("INTENT_SAMPLES_TABLE") or "langgraph_intent_samples").strip()


def _table():
    return boto3.resource("dynamodb", region_name=_region()).Table(_table_name())


def ensure_intent_samples_table() -> None:
    """CREATE_INTENT_SAMPLES_TABLE=1 일 때 테이블을 생성합니다."""
    create_enabled = (os.getenv("CREATE_INTENT_SAMPLES_TABLE") or "").strip() in ("1", "true", "True", "YES", "yes")
    if not create_enabled:
        return

    client = boto3.client("dynamodb", region_name=_region())
    name = _table_name()

    try:
        client.describe_table(TableName=name)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

    # PK: task, SK: text
    client.create_table(
        TableName=name,
        AttributeDefinitions=[
            {"AttributeName": "task", "AttributeType": "S"},
            {"AttributeName": "text", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "task", "KeyType": "HASH"},
            {"AttributeName": "text", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=name)
    print(f"[INTENT_SAMPLES] Table created: {name}")


def seed_intent_samples() -> None:
    """SEED_SAMPLES를 항상 upsert합니다. 신규 항목만 실질적으로 추가됩니다."""
    tbl = _table()
    now = int(time.time())
    try:
        with tbl.batch_writer() as batch:
            for task, texts in SEED_SAMPLES.items():
                for text in texts:
                    batch.put_item(Item={
                        "task": task,
                        "text": text,
                        "source": "seed",
                        "created_at": now,
                    })
        print(f"[INTENT_SAMPLES] Seeded {sum(len(v) for v in SEED_SAMPLES.values())} samples.")
    except Exception as e:
        print(f"[INTENT_SAMPLES] seed failed (non-fatal): {e}")


def load_all_samples() -> Dict[str, List[str]]:
    """전체 샘플을 DynamoDB에서 로드합니다."""
    tbl = _table()
    result: Dict[str, List[str]] = {}

    try:
        paginator = boto3.client("dynamodb", region_name=_region()).get_paginator("scan")
        for page in paginator.paginate(TableName=_table_name()):
            for raw in page.get("Items", []):
                task = raw.get("task", {}).get("S", "")
                text = raw.get("text", {}).get("S", "")
                if task and text:
                    result.setdefault(task, []).append(text)
    except Exception as e:
        print(f"[INTENT_SAMPLES] load failed, using seed fallback: {e}")
        return SEED_SAMPLES

    return result if result else SEED_SAMPLES


def add_sample(task: str, text: str, source: str = "llm_fallback") -> bool:
    """
    LLM fallback 성공 케이스를 샘플로 추가합니다.
    이미 존재하면 무시합니다. 실패해도 요청 흐름에 영향 없음.

    Returns:
        True: 신규 추가됨 / False: 중복 또는 실패
    """
    if not task or not text:
        return False

    text = text.strip()[:500]  # DynamoDB SK 길이 제한 대비

    try:
        _table().put_item(
            Item={
                "task": task,
                "text": text,
                "source": source,
                "created_at": int(time.time()),
            },
            ConditionExpression="attribute_not_exists(#t)",
            ExpressionAttributeNames={"#t": "text"},
        )
        print(f"[INTENT_SAMPLES] Added: [{task}] {text[:60]}")
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False  # 중복
        print(f"[INTENT_SAMPLES] add failed (non-fatal): {e}")
        return False
    except Exception as e:
        print(f"[INTENT_SAMPLES] add failed (non-fatal): {e}")
        return False