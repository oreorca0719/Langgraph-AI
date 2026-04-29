# Langgraph-AI

사내 업무 보조 AI 어시스턴트 — LangGraph + RAG 기반 FastAPI 백엔드

**개발자: 김범준**

---

## 개요

사내 임직원을 위한 AI 어시스턴트 웹 애플리케이션입니다.
사용자 질문 의도를 다단계 라우터(Semantic + LLM Fallback)로 분류하여, **사내 문서 검색 (RAG) · 심화 검색 · 파일 분석** 3가지 기능을 단일 채팅 인터페이스에서 제공합니다.

LangGraph의 순환(Cycle), interrupt(slot 기반 의도 확인 + 진행 확인), DynamoDB 체크포인터 영속화를 활용한 플랫 그래프 구조로 구현되어 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **사내 문서 검색 (RAG)** | 시맨틱 + BM25 하이브리드 검색(RRF 병합) · 출처 페이지 번호 표시 · 검색 결과 없을 시 쿼리 재작성 후 재검색(최대 2회) |
| **심화 검색** | "좀 더 자세히" 등 후속 심화 패턴 감지 시 직전 Q&A 컨텍스트로 쿼리 재구성 + 참조 문서 범위 내 확장 검색(k=15) |
| **파일 분석** | PDF·DOCX·XLSX·PPTX·TXT·MD 첨부파일 텍스트 추출 및 요약 / Q&A (XLSX/DOCX/PPTX 표는 마크다운 변환, PPTX는 슬라이드 단위 분리 + 발표자 노트 포함) |
| **AI 기능 안내** | 인사·자기소개·기능 문의에 대해 위 3가지 기능을 안내 (범위 외 기능 차단) |
| **다단계 라우터** | Semantic 임베딩 매칭 + LLM Fallback · 슬롯 기반 clarification · 무한 루프 방지 (count≥2) |
| **slot 기반 clarification** | `file_context` 누락 시 interrupt() 기반 사용자 질문 → 슬롯 충족 후 진행 확인 단계 |
| **대화 컨텍스트 영속화** | DynamoDB 체크포인터로 배포 후에도 대화 히스토리 유지 (TTL 7일) |
| **프롬프트 인젝션 방어** | 4계층 오케스트레이션 방어 시스템 (Fine-tuning 없이 코드 레벨 구현) |

---

## 기술 스택

- **Backend**: FastAPI, Python 3.11, slowapi (rate limiter), Starlette SessionMiddleware
- **AI Orchestration**: LangGraph (`StateGraph`, `interrupt`, `Command`, `DynamoDBCheckpointer`)
- **LLM**: Google Gemini (`gemini-3-flash-preview`)
- **Embedding**: Google `gemini-embedding-001` (RAG용 task_type 분기: `retrieval_document` / `retrieval_query`)
- **Vector DB**: Chroma (로컬 영속 스토리지) + BM25 (rank-bm25, 인메모리 싱글톤)
- **Document Store**: Amazon DynamoDB (ap-northeast-1) — users · intent_samples · routing_logs · checkpoints
- **Auth**: 세션 쿠키 + CSRF 토큰 (passlib bcrypt)
- **Infrastructure**: AWS App Runner + Amazon ECR
- **File Storage**: Amazon S3 (사내 문서 원본 동기화)

---

## 그래프 구조 (플랫, 12 노드)

```
사용자 요청
    │
    ▼
[input_guard] — 임베딩 유사도 injection 감지
    ├─[injection]→ rejection → END
    └─[pass]→ task_router — 다단계 라우팅
                  │
                  ├─ knowledge_search → search → quality_check
                  │                               ├─[ok]→ answer → END
                  │                               └─[no docs]→ rewrite → search  ← 순환 (최대 2회)
                  │
                  ├─ detail_search → answer → END  ← 심화 질의 전용
                  │
                  ├─ ai_guide → END
                  │
                  ├─ file_chat → END
                  │
                  ├─ rejection → END
                  │
                  └─ clarification ──interrupt──→ 사용자 슬롯 질문
                         ├─[knowledge_search]→ search                         ← 루프 방어 fallback
                         ├─[clarification_confirm]→ interrupt → task_router  ← 슬롯 충족 후 진행 확인
                         └─[task_router]→ task_router                         ← 슬롯 채워진 채로 재라우팅

대화 상태 (히스토리 · 첨부파일 컨텍스트)
    └── DynamoDBCheckpointer → langgraph_checkpoints 테이블 (TTL 7일)
```

**등록 노드 (12)**: `input_guard`, `task_router`, `clarification`, `clarification_confirm`, `rejection`, `search`, `quality_check`, `rewrite`, `answer`, `ai_guide`, `file_chat`, `detail_search`

---

## interrupt 동작 방식 (slot 기반 clarification)

`file_chat` 라우팅 시 첨부 파일(`file_context`)이 없으면 그래프가 일시정지되고 사용자 응답을 대기합니다.

```
사용자: "이 파일 분석해줘" (파일 첨부 없음)
  → task_router가 file_chat으로 분류했으나 file_context 슬롯 누락
  → clarification(slot) interrupt → "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요." 반환

사용자: 파일 업로드
  → /upload 엔드포인트 → state.file_context 채워짐
  → 다음 채팅 입력 시 task_router가 file_chat으로 정상 라우팅
```

**`/chat` API 처리 흐름**

1. `get_state()`로 활성 interrupt 여부 확인 (`tasks[].interrupts` + `state.next` 폴백)
2. interrupt 활성 → `Command(resume=user_input)` 으로 그래프 재개
3. interrupt 없음 → `input_guard`부터 신규 실행 (`recursion_limit=15` 하드코딩)
4. `invoke()` 반환 후 `get_state()` 재확인 — 새 interrupt 발생 시 메시지 포함 응답 반환
5. 완료 시 `task_type` 기반 응답 포맷 반환 (인용 ID `[1][2]`로 출처 필터링)

**interrupt 응답 구조**

```json
{
  "type": "interrupt",
  "interrupt_type": "clarification",
  "current_task": "clarification",
  "message": "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요.",
  "hint": "",
  "sources": []
}
```

---

## 라우팅 의도 카테고리

| 카테고리 | 처리 방식 |
|---|---|
| `knowledge_search` | 시맨틱+BM25 하이브리드 검색(RRF, 시맨틱 후보 4배 풀) → 결과 없을 시 쿼리 재작성 후 재검색(최대 2회) → LLM 답변 |
| `detail_search` | 직전 Q&A 기반 쿼리 재구성 + 참조 문서명 필터 확장 검색(k=15) → LLM 답변 |
| `ai_guide` | 기능 안내 전용 LLM (도구 없음) |
| `file_chat` | 첨부 파일 기반 Q&A (시스템 프롬프트에 파일 내용 직접 주입) |
| `unknown` | fallback: 짧은 질문(≤15자) · 인사 키워드 → ai_guide / 그 외 → knowledge_search |
| `clarification` | 슬롯 감지 (file_context) · `clarification_count≥2` → knowledge_search 강제 fallback |
| `injection` | input_guard / task_router에서 차단 → rejection (히스토리 누적 안 함) |

---

## 라우팅 시스템

`task_router.py`는 2단계 라우터로 구성됩니다.

```
사용자 요청
    │
    ├─ 1. _semantic_route — intent_samples 임베딩 기반 코사인 유사도 분류
    │      ├─ 신뢰도 임계값(top1 ≥ 0.62, margin ≥ 0.08) 통과 시 결정
    │      └─ unknown이면 다음 단계
    │
    ├─ 2. llm_intent_fallback — LLM 직접 분류 (timeout 8초)
    │      └─ 결과를 intent_samples에 자동 누적 (학습 효과)
    │
    └─ 3. 부가 라우팅 보정
           ├─ detail_search 감지: 직전 task가 knowledge_search/detail_search이고 입력에 "자세히/좀 더/구체적" 등 패턴 포함 시
           ├─ file_context fallback: unknown인데 첨부 파일이 있으면 file_chat으로 전환
           └─ slot 누락 감지: file_chat인데 file_context 없으면 clarification으로 전환
```

**slot 기반 clarification**

| 누락 슬롯 | 질문 내용 | 발동 조건 |
|---|---|---|
| `file_context` | 첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요. | file_chat인데 업로드된 파일이 없을 때 |

루프 방어: `clarification_count ≥ 2` → `knowledge_search` 강제 fallback
부정 응답("아니요/취소/그만" 등) 감지 시: knowledge_search로 fallback

---

## 디렉토리 구조

```
Langgraph-AI/
├── main.py                        # FastAPI 앱 엔트리포인트 + LangGraph 플랫 그래프 구성
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example                   # 환경변수 템플릿
├── policy-dynamo-users.json       # DynamoDB IAM 정책 파일
├── trust-apprunner-instance.json  # App Runner IAM 신뢰 정책 파일
├── .github/workflows/deploy.yml   # CI/CD 파이프라인
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
│   │   └── nodes/
│   │       ├── input_guard.py     # 그래프 진입점 보안 필터 (injection 차단)
│   │       ├── task_router.py     # 다단계 라우터 (Semantic + LLM Fallback) · rejection_node 포함
│   │       ├── clarification.py   # 슬롯 누락 시 interrupt + 진행 확인 노드 2종
│   │       ├── knowledge_search.py # search / quality_check / rewrite / answer 4노드 (하이브리드 검색)
│   │       ├── detail_search.py   # 심화 질의 전용 노드 (쿼리 재구성 + 문서 필터 검색)
│   │       ├── ai_guide.py        # AI 기능 안내 노드
│   │       ├── file_chat.py       # 첨부 파일 Q&A 노드 (시스템 프롬프트 주입)
│   │       ├── file_extractor.py  # 파일 텍스트 추출 헬퍼 (XLSX/DOCX/PPTX 마크다운 변환, PPTX 슬라이드 단위 분리)
│   │       └── llm_intent_fallback.py  # LLM 기반 의도 분류 fallback
│   │
│   ├── knowledge/
│   │   └── ingest.py              # S3 → Chroma 문서 인제스트 (PDF 페이지 단위 청킹, 청크 오버랩)
│   │
│   ├── security/
│   │   ├── injection_detector.py  # 임베딩 유사도 + 슬라이딩 윈도우 injection 탐지
│   │   ├── content_sanitizer.py   # RAG 문서 · 파일 내용 sanitize
│   │   └── output_validator.py    # 응답 민감 정보 출력 검증
│   │
│   └── core/
│       ├── config.py              # 환경변수 중앙 관리 + LLM/Embedding factory
│       └── history_utils.py       # cosine, extract_text_content
│
├── docs/
│   └── features.md                # (참고용 / 일부 항목은 구버전 명세)
│
├── templates/
│   ├── home.html                  # 홈페이지 (사용자 정보 + 최근 이용)
│   ├── index.html                 # 채팅 UI
│   ├── login.html                 # 로그인 페이지
│   ├── admin_home.html            # 관리자 홈
│   ├── admin_users.html           # 사용자 관리
│   └── pending.html               # 승인 대기 안내 페이지
│
└── static/
    ├── css/style.css
    └── js/chat.js
```

---

## 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET`  | `/health` | 헬스 체크 |
| `GET`  | `/status` | 시스템 상태 (관리자 전용) |
| `GET`  | `/` | 홈 (최근 이용 내역) |
| `GET`  | `/chat` | 채팅 UI |
| `POST` | `/chat` | 채팅 메시지 처리 (interrupt 패턴 지원) |
| `POST` | `/chat/reset` | 현재 thread 체크포인트 삭제 |
| `POST` | `/upload` | 파일 업로드 + LLM 요약/키워드 추출 + `file_context` State 저장 |
| `POST` | `/chat-with-file` | 파일 + 질문 동시 입력 → 파일 컨텍스트 기반 LLM 답변 |
| `GET`  | `/admin` | 관리자 홈 |
| `GET`  | `/admin/users` | 사용자 관리 (승인/거절·관리자 토글·소속 지정·삭제) |

**파일 업로드 검증** (`/upload`, `/chat-with-file`)
- 허용 확장자: `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.pptx`
- 최대 크기: 20MB
- Magic byte 검증: PDF=`%PDF`, OOXML=`PK\x03\x04` (확장자 위조 차단)

**Chat 입력 제한**: `CHAT_MAX_INPUT_CHARS` (기본 4000자)

---

## GraphState 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `input_data` | str | 사용자 입력 (clarification 재개 시 combined 값으로 교체) |
| `input_embedding` | List[float] | 사용자 입력 임베딩 캐시 |
| `task_type` | str | 라우팅 결과 카테고리 |
| `task_args` | Dict | 라우팅 디버그 정보, 검색 문서, 누락 슬롯 등 |
| `messages` | Sequence[BaseMessage] | 대화 히스토리 (누적 append) |
| `citations_used` | List[Dict] | 인용된 출처 목록 |
| `file_context` | str | 업로드 파일 텍스트 (State에 영속화) |
| `file_context_name` | str | 업로드 파일명 |
| `retry_count` | int | knowledge search 재시도 횟수 |
| `clarification_count` | int | clarification 발동 횟수 (루프 방어용, ≥2 → knowledge_search fallback) |
| `interrupt_type` | str | "clarification" |
| `pending_confirm_msg` | str | 슬롯 수집 후 확인 메시지 |
| `trace_id` | str | 요청별 트레이스 ID |

---

## RAG 인제스트 (`auto_ingest_if_enabled`)

앱 startup 시 자동 수행 (`AUTO_INGEST=1`):

1. **S3 동기화**: `S3_KNOWLEDGE_BUCKET` 설정 시 S3 → `KNOWLEDGE_DIR` 다운로드
2. **해시 비교**: SHA256 해시 비교로 변경 파일만 재인제스트 (`.ingest_state.json`)
3. **삭제 동기화**: 로컬에 없는 파일은 Chroma에서 제거
4. **포맷별 처리**:
   - PDF → 페이지 단위 추출 후 청킹 (메타데이터에 `page_number` 기록)
   - DOCX/XLSX/PPTX → 표는 마크다운 표로 변환, PPTX는 슬라이드 단위 분리 + 제목 헤딩 + 발표자 노트 포함
   - TXT/MD → 그대로 읽기
5. **청킹**: 단락 기반 + 슬라이딩 윈도우, 청크 간 오버랩으로 단락 경계 누락 방지
6. **BM25 재인덱싱**: 인제스트 변경 시 자동 무효화 → 다음 검색에서 재구축

---

## 프롬프트 인젝션 방어

Fine-tuning 없이 오케스트레이션 레벨에서 4계층 방어를 구현합니다. 추가 LLM 호출 비용 없이 동작합니다.

| 레이어 | 위치 | 방식 | 차단 대상 |
|---|---|---|---|
| **1차** | `input_guard_node` (그래프 첫 노드) | 임베딩 유사도 + 슬라이딩 윈도우 | 알려진 패턴 · 분할 인젝션 · 소셜 엔지니어링 |
| **2차** | `task_router` → `rejection_node` | 라우팅 `injection` task_type | Semantic 변형 우회 · 범위 외 질문 |
| **3차** | `search_node` | RAG 문서 sanitize | 문서 경유 간접 인젝션 |
| **4차** | `answer_node` | 응답 출력 규칙 기반 검증 | 1~3차 통과 후 민감 정보 노출 |

**거부 turn 히스토리 격리**

`rejection_node`는 인젝션으로 거부된 turn을 `messages`에 추가하지 않습니다. 거부 응답 텍스트는 `/chat` 엔드포인트가 `task_type=="injection"`을 감지해 직접 반환합니다. 이로써 인젝션 텍스트가 thread history에 누적되어 슬라이딩 윈도우 검사에서 false positive를 유발하는 것을 방지합니다.

**요청 간 컨텍스트 격리**

각 LLM 호출 노드(knowledge_search, file_chat, ai_guide)는 이전 대화 히스토리를 LLM에 주입하지 않습니다. 요청마다 독립적인 컨텍스트로 처리하여 이전 대화 내용이 현재 답변에 오염되는 것을 방지합니다. 첨부 파일 컨텍스트는 DynamoDB 체크포인터를 통해 별도로 유지됩니다.

---

## 환경변수

`.env` 파일을 프로젝트 루트에 생성하여 아래 변수를 설정합니다. `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY` 중 하나는 필수입니다 (코드에서 `GEMINI_API_KEY`를 `GOOGLE_API_KEY`로 자동 승격).

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

# LLM (둘 중 하나 필수)
GEMINI_API_KEY=...
GOOGLE_API_KEY=...

# Admin 초기 계정 (없으면 미생성)
ADMIN_EMAIL=
ADMIN_PASSWORD=
ADMIN_NAME=Admin

# Session (필수: 안전한 무작위 문자열)
SESSION_SECRET=...
SESSION_MAX_AGE=28800

# S3 (문서 인제스트, 비우면 로컬 인제스트만 수행)
S3_KNOWLEDGE_BUCKET=
S3_KNOWLEDGE_PREFIX=knowledge_data/

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
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION=my_knowledge
AUTO_INGEST=1
INGEST_CHUNK_MAX_CHARS=1200
INGEST_CHUNK_OVERLAP=200

# Chat 입력 / 파일 컨텍스트 제한
CHAT_MAX_INPUT_CHARS=4000
FILE_CONTEXT_MAX_CHARS=8000

# 대화 히스토리
HISTORY_MAX_MESSAGES=40
HISTORY_RELEVANCE_THRESHOLD=0.40
HISTORY_ALWAYS_KEEP_LAST_N=0
CHECKPOINT_TTL_DAYS=7
CHECKPOINT_MAX_MESSAGES=40

# Thread 컨텍스트 스코프 (user | task)
THREAD_CONTEXT_SCOPE=user

# 프롬프트 인젝션 방어 임계값
INJECTION_THRESHOLD_SINGLE=0.80
INJECTION_THRESHOLD_COMBINED=0.76
INJECTION_WINDOW_TURNS=3

# 라우터 임계값
ROUTER_TOP1_MIN=0.62
ROUTER_MARGIN_MIN=0.08

# LLM Intent Fallback
LLM_FALLBACK_ENABLED=1
LLM_FALLBACK_TIMEOUT_SEC=8
LLM_FALLBACK_CONFIDENCE_MIN=0.5

# LangSmith (선택)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=langgraph-rag

# 그래프 순환 제한
# recursion_limit=15 (main.py 하드코딩, 변경 시 코드 수정)
```

---

## DevOps 전체 플로우

### [0단계] 인프라 사전 준비 (최초 1회)

```
AWS Console / CLI
  ├── Amazon ECR       — Docker 이미지 레포지토리 생성
  ├── AWS App Runner   — ECR 이미지 기반 서비스 생성 + 환경변수 설정
  ├── Amazon DynamoDB  — 앱 기동 시 테이블 자동 생성 (CREATE_*_TABLE=1)
  └── Amazon S3        — 사내 문서 원본 업로드 (S3_KNOWLEDGE_BUCKET)

GitHub 레포 → Settings → Secrets and variables → Actions
  ├── AWS_ACCESS_KEY_ID
  ├── AWS_SECRET_ACCESS_KEY
  ├── AWS_REGION
  ├── ECR_REPOSITORY
  └── APP_RUNNER_SERVICE_ARN
```

### [1단계] 로컬 개발

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

> `knowledge_data/`와 `chroma_db/`는 로컬 테스트 전용입니다.
> 운영 환경(App Runner)에서는 S3에서 자동 인제스트되며, 재배포 시 초기화됩니다.

### [2단계] 배포 트리거

```bash
git push origin main
```

`main` 브랜치에 push하면 GitHub Actions (`.github/workflows/deploy.yml`)가 자동으로 기동됩니다.

### [3단계] GitHub Actions (ubuntu-latest runner)

```
runner VM 기동
    ├── 1. 코드 체크아웃
    ├── 2. AWS 자격증명 구성
    ├── 3. Amazon ECR 로그인
    ├── 4. Docker 이미지 빌드
    ├── 5. ECR 푸시 (커밋 SHA 태그 + latest)
    ├── 6. App Runner RUNNING 상태 대기 (최대 10분)
    └── 7. App Runner 배포 트리거
```

### [4단계] 컨테이너 초기화 (앱 기동 시 자동 수행)

```
FastAPI 앱 시작 (main.py lifespan)
    ├── SESSION_SECRET 검증 (미설정/기본값이면 기동 거부)
    ├── DynamoDB 테이블 자동 생성 (users · routing_logs · intent_samples · checkpoints)
    ├── intent_samples seed 데이터 upsert
    ├── 초기 관리자 계정 보정 (ADMIN_EMAIL/ADMIN_PASSWORD 설정 시)
    ├── S3 → Chroma 문서 인제스트 (AUTO_INGEST=1)
    └── FastAPI 서버 Ready
```

---

## 관리자 기능

- `/admin` — 관리자 홈
- `/admin/users` — 사용자 목록 조회, 승인/거절, 관리자 권한 토글, 소속(department) 지정, 계정 삭제

라우팅 이력은 DynamoDB `routing_logs` 테이블에 요청별로 기록됩니다 (사용자별 조회 지원).

---

## 라이선스

Internal use only.
