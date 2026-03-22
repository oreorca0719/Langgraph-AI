# Langgraph-Rag

사내 업무 보조 AI 어시스턴트 — LangGraph + RAG 기반 FastAPI 백엔드

**개발자: 김범준**
---

## 개요

사내 임직원을 위한 AI 어시스턴트 웹 애플리케이션 프로토타입입니다.
질문 의도를 시맨틱 라우터로 자동 분류하여, RAG 문서 검색 / 이메일 초안 / RFP 초안 / 파일 분석 / 일반 대화 등의 기능을 단일 채팅 인터페이스에서 제공합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **사내 문서 검색 (RAG)** | Chroma 벡터 DB + Gemini 임베딩으로 사내 지식 베이스 검색 · 출처 페이지 번호 표시 |
| **이메일 초안 작성** | 수신자·주제·본문 구조의 이메일 초안 자동 생성 |
| **RFP 초안 작성** | 제안요청서(RFP) 형식의 문서 초안 자동 생성 |
| **파일 분석** | PDF·DOCX·XLSX·PPTX·TXT 첨부파일 텍스트 추출 및 요약 / Q&A |
| **일반 대화** | 사내 업무 범위 내 자유 질의응답 및 AI 기능 안내 |
| **시맨틱 라우팅** | 코사인 유사도 기반 의도 분류 + LLM fallback |
| **대화 컨텍스트 영속화** | DynamoDB 체크포인터로 배포 후에도 대화 히스토리 유지 |

---

## 기술 스택

- **Backend**: FastAPI, Python 3.11
- **AI Orchestration**: LangGraph (`StateGraph`, `DynamoDBCheckpointer`)
- **LLM**: Google Gemini (`gemini-3-flash-preview`)
- **Embedding**: Google `gemini-embedding-001`
- **Vector DB**: Chroma (로컬 영속 스토리지)
- **Document Store**: Amazon DynamoDB (ap-northeast-1)
- **Auth**: 세션 쿠키 + CSRF 토큰
- **Infrastructure**: AWS App Runner + Amazon ECR
- **File Storage**: Amazon S3 (사내 문서 원본)

---

## 아키텍처 개요

```
사용자 요청
    │
    ▼
[시맨틱 라우터]  ← DynamoDB(intent_samples) 기반 코사인 유사도 분류
    │                + LLM intent fallback (unknown 시)
    │
    ├── chat     → Chroma 검색 → Gemini 답변 생성 (출처 + 페이지 번호 포함)
    ├── email    → 이메일 초안 생성 (To / CC / Subject / Body)
    ├── rfp      → RFP 초안 생성
    └── file     → 파일 텍스트 추출 → 요약 / Q&A

대화 상태 (히스토리 · 첨부파일 컨텍스트)
    └── DynamoDBCheckpointer → langgraph_checkpoints 테이블 (TTL 7일)
```

---

## 디렉토리 구조

```
Langgraph-Rag/
├── main.py                        # FastAPI 앱 엔트리포인트
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .github/workflows/deploy.yml  # CI/CD 파이프라인
│
├── app/
│   ├── checkpointer/
│   │   └── dynamo_checkpointer.py # DynamoDB 기반 LangGraph 체크포인터
│   │
│   ├── auth/
│   │   ├── dynamo.py              # DynamoDB 사용자 CRUD
│   │   ├── routes.py              # 인증·관리자 API 라우터
│   │   ├── intent_samples.py      # 시맨틱 라우터 seed 데이터 관리
│   │   ├── routing_log.py         # 라우팅 이력 기록 / 조회
│   │   ├── deps.py                # 인증 의존성 (get_current_user)
│   │   └── security.py            # 세션·CSRF 유틸
│   │
│   ├── graph/
│   │   ├── states/state.py        # GraphState 정의
│   │   ├── nodes/
│   │   │   ├── task_router.py     # 시맨틱 라우터 노드
│   │   │   ├── llm_intent_fallback.py  # LLM 의도 분류 fallback
│   │   │   ├── email_agent.py     # 이메일 초안 노드
│   │   │   ├── rfp_agent.py       # RFP 초안 노드
│   │   │   └── file_extractor.py  # 파일 추출 노드
│   │   └── subgraphs/
│   │       ├── chat_graph.py      # RAG + 일반 대화 서브그래프
│   │       ├── email_graph.py     # 이메일 서브그래프
│   │       ├── rfp_graph.py       # RFP 서브그래프
│   │       └── file_graph.py      # 파일 분석 서브그래프
│   │
│   ├── knowledge/
│   │   └── ingest.py              # S3 → Chroma 문서 인제스트 (PDF 페이지 단위 청킹)
│   │
│   └── core/
│       ├── config.py              # 환경변수 중앙 관리
│       ├── trace_buffer.py        # 인메모리 트레이스 버퍼
│       └── log_buffer.py          # 로그 버퍼
│
├── templates/
│   ├── home.html                  # 홈페이지 (사용자 정보 + 최근 이용)
│   ├── index.html                 # 채팅 UI
│   ├── login.html                 # 로그인 페이지
│   ├── admin_home.html            # 관리자 홈
│   ├── admin_users.html           # 사용자 관리
│   └── admin_monitor.html         # 라우팅 성능 모니터링 + 실시간 로그
│
└── static/
    ├── css/style.css
    └── js/chat.js
```

---

## 환경변수

`.env` 파일을 프로젝트 루트에 생성하여 아래 변수를 설정합니다.
(`.env`는 `.gitignore`에 포함되어 있으므로 저장소에 커밋되지 않습니다.)

```env
# AWS
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# DynamoDB 테이블명 (기본값 사용 시 생략 가능)
USERS_TABLE=langgraph_users
INTENT_SAMPLES_TABLE=langgraph_intent_samples
ROUTING_LOG_TABLE=langgraph_routing_logs
CHECKPOINT_TABLE=langgraph_checkpoints

# DynamoDB 테이블 자동 생성 여부
CREATE_USERS_TABLE=1
CREATE_INTENT_SAMPLES_TABLE=1
CREATE_ROUTING_LOG_TABLE=1

# Google AI
GOOGLE_API_KEY=...

# S3 (문서 인제스트)
S3_KNOWLEDGE_BUCKET=...
S3_KNOWLEDGE_PREFIX=knowledge_data/

# 세션
SESSION_SECRET=...

# LLM
LLM_MODEL=gemini-3-flash-preview
LLM_TEMPERATURE=0
LLM_MAX_OUTPUT_TOKENS=4096

# RAG 검색 품질
RETRIEVAL_MIN_RELEVANCE=0.3
RETRIEVAL_MAX_DISTANCE=0.75
RETRIEVAL_TOP_K=5

# 인제스트
KNOWLEDGE_DIR=./knowledge_data
AUTO_INGEST=1
INGEST_CHUNK_MAX_CHARS=1200

# 라우팅 임계값
ROUTER_TOP1_MIN=0.62
ROUTER_MARGIN_MIN=0.08

# 대화 히스토리
HISTORY_MAX_MESSAGES=40
CHECKPOINT_TTL_DAYS=7
CHECKPOINT_MAX_MESSAGES=40
```

---

## 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 시작
uvicorn main:app --reload --port 8000
```

> **주의**: `knowledge_data/`와 `chroma_db/`는 로컬 테스트 전용입니다.
> 운영 환경(App Runner)에서는 S3에서 자동 인제스트되며, 재배포 시 초기화됩니다.

---

## 배포 파이프라인

`main` 브랜치에 push하면 GitHub Actions가 자동으로 빌드~배포까지 처리합니다.

```
git push origin main
      │
      ▼
GitHub Actions (.github/workflows/deploy.yml)
      │
      ├── 1. AWS 인증 (IAM 액세스 키)
      ├── 2. Docker 이미지 빌드
      ├── 3. Amazon ECR 푸시 (커밋 해시 태그 + latest 태그)
      │
      ▼
App Runner 새 이미지 감지 → 자동 재배포
```

**GitHub Secrets 설정 필요** (레포 → Settings → Secrets and variables → Actions)

| Secret | 설명 |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM 사용자 액세스 키 |
| `AWS_SECRET_ACCESS_KEY` | IAM 사용자 시크릿 키 |
| `AWS_REGION` | `ap-northeast-1` |
| `ECR_REPOSITORY` | ECR 레포지토리 이름 |
| `APP_RUNNER_SERVICE_ARN` | App Runner 서비스 ARN |

**환경변수**는 App Runner 콘솔 서비스 설정에서 별도 관리합니다. `.env`는 배포 이미지에 포함되지 않습니다.

---

## 관리자 기능

- `/admin` — 관리자 홈
- `/admin/users` — 사용자 목록 조회, 승인/거절, 관리자 권한 토글, 소속(department) 지정, 계정 삭제
- `/admin/monitor` — 라우팅 통계(소스·태스크 분포, 평균 score/margin), 라우팅 로그(사용자별 질문 이력), 실시간 서버 로그 스트리밍

---

## 라이선스

Internal use only.
