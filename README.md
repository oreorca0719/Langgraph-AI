# Langgraph-Rag

사내 업무 보조 AI 어시스턴트 — LangGraph + RAG 기반 FastAPI 백엔드

**개발자: 김범준**
---

## 개요

사내 임직원을 위한 AI 어시스턴트 웹 애플리케이션입니다.
질문 의도를 시맨틱 라우터로 자동 분류하여, RAG 문서 검색 / 이메일 초안 / RFP 초안 / 파일 분석 / AI 기능 안내 등의 기능을 단일 채팅 인터페이스에서 제공합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **사내 문서 검색 (RAG)** | Chroma 벡터 DB + Gemini 임베딩으로 사내 지식 베이스 검색 · 출처 페이지 번호 표시 |
| **이메일 초안 작성** | 수신자·주제·본문 구조의 이메일 초안 자동 생성 |
| **RFP 초안 작성** | 제안요청서(RFP) 형식의 문서 초안 자동 생성 |
| **파일 분석** | PDF·DOCX·XLSX·PPTX·TXT 첨부파일 텍스트 추출 및 요약 / Q&A |
| **AI 기능 안내** | 인사·자기소개·기능 문의에 대해 5가지 제공 기능만 안내 (범위 외 기능 언급 차단) |
| **시맨틱 라우팅** | 코사인 유사도 기반 의도 분류 + LLM fallback / out_of_scope 즉시 거절 |
| **히스토리 관련성 필터링** | 임베딩 유사도 기반으로 현재 질문과 무관한 대화 턴을 LLM 컨텍스트에서 제거 |
| **대화 컨텍스트 영속화** | DynamoDB 체크포인터로 배포 후에도 대화 히스토리 유지 |
| **프롬프트 인젝션 방어** | 4계층 오케스트레이션 방어 시스템 (Fine-tuning 없이 코드 레벨 구현) |

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
[1차 방어] 임베딩 유사도 사전 차단 (슬라이딩 윈도우 포함)
    │  인젝션 감지 시 즉시 거절 반환 (graph 실행 없음)
    ▼
[2차 방어] 시맨틱 라우터  ← DynamoDB(intent_samples) 기반 코사인 유사도 분류
    │  + LLM intent fallback (unknown 시)
    │  injection / out_of_scope / unknown → 즉시 거절 반환
    │
    ├── knowledge_search → [3차: RAG 결과 sanitize]
    │                    → Chroma 검색 → Gemini 요약 (출처 + 페이지 번호 포함)
    │                    → [4차: 응답 출력 검증]
    │
    ├── ai_guide         → 5가지 기능 안내 전용 LLM 응답 (도구 없음)
    │
    ├── file_chat        → 첨부 파일 Q&A (get_attached_file 단일 도구)
    │
    ├── email_draft      → 이메일 초안 생성 (To / CC / Subject / Body)
    ├── rfp_draft        → RFP 초안 생성
    └── file_extract     → 파일 텍스트 추출

히스토리 관련성 필터 (모든 서브그래프 공통)
    └── 현재 질문과 코사인 유사도 < 0.40인 이전 대화 턴을 LLM 컨텍스트에서 제거
        → 컨텍스트 오염(범위 외 답변의 히스토리 잔류) 차단

대화 상태 (히스토리 · 첨부파일 컨텍스트)
    └── DynamoDBCheckpointer → langgraph_checkpoints 테이블 (TTL 7일)
```

---

## 라우팅 의도 카테고리

시맨틱 라우터는 아래 7개 카테고리로 의도를 분류합니다.

| 카테고리 | 처리 방식 |
|---|---|
| `knowledge_search` | RAG 검색 후 LLM 요약 |
| `ai_guide` | 기능 안내 전용 LLM (도구 없음) |
| `file_chat` | 첨부 파일 기반 Q&A (ReAct) |
| `email_draft` | 이메일 초안 생성 |
| `rfp_draft` | RFP 초안 생성 |
| `file_extract` | 파일 경로 지정 텍스트 추출 |
| `injection` / `out_of_scope` / `unknown` | LLM 호출 없이 즉시 거절 |

> **out_of_scope** 분류 기준: 소셜 엔지니어링(권위 사칭, 테스트·검수 사칭) + 업무 범위 외 질문(날씨·주식·뉴스 등)
> 단, `out_of_scope`로 분류되더라도 검색 의도 키워드("찾아줘", "탐색", "문서" 등)가 포함된 경우 `knowledge_search`로 강제 전환합니다.

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
│   │   │                          # 카테고리: knowledge_search / ai_guide / file_chat
│   │   │                          #           email_draft / file_extract / rfp_draft
│   │   │                          #           injection / out_of_scope
│   │   ├── routing_log.py         # 라우팅 이력 기록 / 조회
│   │   ├── deps.py                # 인증 의존성 (get_current_user)
│   │   └── security.py            # 세션·CSRF 유틸
│   │
│   ├── graph/
│   │   ├── states/state.py        # GraphState 정의 (13개 필드)
│   │   ├── nodes/
│   │   │   ├── task_router.py     # 시맨틱 라우터 노드 + preview_route()
│   │   │   ├── llm_intent_fallback.py  # LLM 의도 분류 fallback
│   │   │   ├── email_agent.py     # 이메일 초안 노드
│   │   │   ├── rfp_agent.py       # RFP 초안 노드
│   │   │   └── file_extractor.py  # 파일 추출 노드
│   │   └── subgraphs/
│   │       ├── knowledge_search_graph.py  # RAG 검색 서브그래프 (ReAct 없음)
│   │       ├── ai_guide_graph.py          # AI 기능 안내 서브그래프 (도구 없음)
│   │       ├── chat_graph.py              # 첨부 파일 Q&A 서브그래프 (file_chat)
│   │       ├── email_graph.py             # 이메일 서브그래프
│   │       ├── rfp_graph.py               # RFP 서브그래프
│   │       └── file_graph.py              # 파일 추출 서브그래프
│   │
│   ├── knowledge/
│   │   └── ingest.py              # S3 → Chroma 문서 인제스트 (PDF 페이지 단위 청킹)
│   │
│   ├── security/
│   │   ├── injection_detector.py  # 1차: 임베딩 유사도 + 슬라이딩 윈도우
│   │   ├── content_sanitizer.py   # 3차: RAG 문서 · 파일 내용 sanitize
│   │   └── output_validator.py    # 4차: 응답 민감 정보 출력 검증
│   │
│   └── core/
│       ├── config.py              # 환경변수 중앙 관리
│       ├── history_utils.py       # 공통 유틸 (cosine, filter_history_by_relevance)
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
HISTORY_MAX_MESSAGES=40          # LLM에 전달할 최대 메시지 수
HISTORY_RELEVANCE_THRESHOLD=0.40 # 히스토리 필터 임계값 (미만 턴 제거)
HISTORY_ALWAYS_KEEP_LAST_N=0     # 항상 유지할 마지막 N턴 (0=모든 턴 필터링 대상)
CHECKPOINT_TTL_DAYS=7
CHECKPOINT_MAX_MESSAGES=40

# 프롬프트 인젝션 방어 임계값
INJECTION_THRESHOLD_SINGLE=0.80
INJECTION_THRESHOLD_COMBINED=0.76
INJECTION_WINDOW_TURNS=3
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
  ├── AWS_ACCESS_KEY_ID       IAM 사용자 액세스 키
  ├── AWS_SECRET_ACCESS_KEY   IAM 사용자 시크릿 키
  ├── AWS_REGION              ap-northeast-1
  ├── ECR_REPOSITORY          ECR 레포지토리 이름
  └── APP_RUNNER_SERVICE_ARN  App Runner 서비스 ARN
```

> **환경변수** (`.env` 내용)는 App Runner 콘솔 서비스 설정에서 별도 관리합니다.
> `.env`는 `.gitignore`에 포함되어 있어 배포 이미지에 포함되지 않습니다.

---

### [1단계] 로컬 개발

```bash
# 의존성 설치
pip install -r requirements.txt

# 로컬 서버 실행 (.env 파일 필요)
uvicorn main:app --reload --port 8000
```

> `knowledge_data/`와 `chroma_db/`는 로컬 테스트 전용입니다.
> 운영 환경(App Runner)에서는 S3에서 자동 인제스트되며, 재배포 시 초기화됩니다.

---

### [2단계] 배포 트리거

```bash
git push origin main
```

`main` 브랜치에 push하면 GitHub Actions (`.github/workflows/deploy.yml`)가 자동으로 기동됩니다.

---

### [3단계] GitHub Actions (ubuntu-latest runner)

> Docker Desktop 불필요 — GitHub 서버(runner VM)에서 빌드되므로 로컬 환경과 무관합니다.

```
runner VM 기동 (ubuntu-latest)
    │
    ├── 1. 코드 체크아웃 (actions/checkout@v4)
    ├── 2. AWS 자격증명 구성 (IAM 액세스 키)
    ├── 3. Amazon ECR 로그인
    ├── 4. Docker 이미지 빌드
    │       docker build -t <ECR>/<REPO>:<커밋 SHA> .
    ├── 5. ECR 푸시 (커밋 SHA 태그 + latest 태그 동시)
    ├── 6. App Runner RUNNING 상태 대기 (최대 10분, 30초 간격 폴링)
    └── 7. App Runner 배포 트리거 (start-deployment)
```

---

### [4단계] App Runner 자동 재배포

```
ECR latest 이미지 감지
    │
    ▼
기존 컨테이너 교체 → 새 컨테이너 기동
```

---

### [5단계] 컨테이너 초기화 (앱 기동 시 자동 수행)

```
FastAPI 앱 시작 (main.py)
    │
    ├── DynamoDB 테이블 자동 생성 (CREATE_*_TABLE=1 환경변수 기준)
    │       langgraph_users / langgraph_intent_samples
    │       langgraph_routing_logs / langgraph_checkpoints
    │
    ├── intent_samples seed 데이터 upsert
    │       knowledge_search / ai_guide / file_chat / email_draft
    │       file_extract / rfp_draft / injection / out_of_scope
    │
    ├── S3 → Chroma 문서 인제스트 (AUTO_INGEST=1 환경변수 기준)
    │       S3_KNOWLEDGE_BUCKET 에서 PDF 다운로드
    │       → 페이지 단위 청킹 → Gemini 임베딩 → chroma_db/ 저장
    │
    └── FastAPI 서버 Ready — 사용자 요청 수신 시작
```

---

## 프롬프트 인젝션 방어

Fine-tuning 없이 오케스트레이션 레벨에서 4계층 방어를 구현합니다. 추가 LLM 호출 비용 없이 동작합니다.

| 레이어 | 방식 | 차단 대상 |
|---|---|---|
| **1차** | 임베딩 유사도 + 슬라이딩 윈도우 | 알려진 패턴 · 분할 인젝션 · 다단계 공격 · 소셜 엔지니어링 프레이밍 |
| **2차** | 라우터 `injection` / `out_of_scope` task_type | 소셜 엔지니어링 · 범위 외 질문 · Semantic 변형 우회 |
| **3차** | RAG 문서 · 파일 내용 sanitize | 문서/파일 경유 간접 인젝션 |
| **4차** | 응답 출력 규칙 기반 검증 | 1~3차 통과 후 민감 정보 노출 |

**컨텍스트 오염 방어 (히스토리 관련성 필터)**

1차 방어를 통과한 소셜 엔지니어링 응답이 히스토리에 잔류하더라도, 이후 요청에서 현재 질문과 코사인 유사도 임계값(기본 0.40) 미만인 과거 턴을 LLM 컨텍스트에서 제거합니다. `HISTORY_ALWAYS_KEEP_LAST_N=0` 설정으로 직전 턴도 필터링 대상에 포함됩니다.

**시스템 프롬프트 보안 구조**

- `【절대 금지 원칙】` 섹션을 최상단에 배치하여 기술적 내부 구현(모델명·API 키·DB 구조·방어 메커니즘) 노출을 차단합니다.
- 각 서브그래프별로 전용 시스템 프롬프트를 분리 적용하여 LLM 판단 범위를 최소화합니다.

---

## 관리자 기능

- `/admin` — 관리자 홈
- `/admin/users` — 사용자 목록 조회, 승인/거절, 관리자 권한 토글, 소속(department) 지정, 계정 삭제
- `/admin/monitor` — 라우팅 통계(소스·태스크 분포, 평균 score/margin), 라우팅 로그(사용자별 질문 이력 + 전체 재분류 기능), 실시간 서버 로그 스트리밍

**라우팅 정확도 개선 루프**

관리자 모니터 페이지에서 라우팅 결정 이력의 모든 쿼리를 올바른 카테고리로 재분류할 수 있습니다. 재분류된 샘플은 DynamoDB `intent_samples` 테이블에 즉시 반영되고, 시맨틱 라우터의 샘플 벡터 캐시가 무효화되어 다음 요청부터 반영됩니다.

---

## 라이선스

Internal use only.
