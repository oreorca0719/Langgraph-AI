# Langgraph-AI

사내 업무 보조 AI 어시스턴트 — LangGraph + RAG 기반 FastAPI 백엔드

**개발자: 김범준**

---

## 개요

사내 임직원을 위한 AI 어시스턴트 웹 애플리케이션입니다.
질문 의도를 LLM 구조화 출력 라우터로 자동 분류하여, RAG 문서 검색 / 이메일 초안 / RFP 초안 / 파일 분석 / AI 기능 안내 기능을 단일 채팅 인터페이스에서 제공합니다.

LangGraph의 순환(Cycle), interrupt(Human-in-the-loop), 다중 에이전트 체인, 복합 요청 오케스트레이션(Planner)을 완전하게 활용하는 플랫 그래프 구조로 구현되어 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **사내 문서 검색 (RAG)** | 시맨틱 + BM25 하이브리드 검색(RRF 병합)으로 사내 지식 베이스 검색 · 출처 페이지 번호 표시 · 검색 결과 없을 시 쿼리 재작성 후 재검색(최대 2회) |
| **심화 검색** | "좀 더 자세히" 등 후속 심화 질의 감지 시 직전 Q&A 컨텍스트로 쿼리 재구성 + 참조 문서 범위 내 확장 검색 |
| **복합 요청 처리 (Planner)** | "검색해서 RFP 작성해줘" 같은 복합 요청을 LLM으로 분해 → 선행 검색 직접 실행 → 결과를 컨텍스트로 주입 → 최종 작성 노드로 라우팅 |
| **이메일 초안 작성** | 수신자·주제·본문 구조의 이메일 초안 생성 → interrupt로 사용자 검토 → 수정/승인 루프 |
| **RFP 초안 작성** | 사내 문서 조사 → 초안 작성 → 8개 섹션 자동 검토(재작성 최대 1회) → interrupt로 사용자 검토 |
| **파일 분석** | PDF·DOCX·XLSX·PPTX·TXT 첨부파일 텍스트 추출 및 요약 / Q&A |
| **AI 기능 안내** | 인사·자기소개·기능 문의에 대해 5가지 제공 기능만 안내 (범위 외 기능 언급 차단) |
| **LLM 라우터 (구조화 출력)** | `with_structured_output(RouteDecision)` — LLM이 직접 태스크 분류 + missing_slots 감지 · 슬롯 기반 clarification · 무한 루프 방지 (count≥2) |
| **Human-in-the-loop** | interrupt() 기반 실시간 사용자 검토 — 이메일·RFP 초안 승인/수정/전환 |
| **대화 컨텍스트 영속화** | DynamoDB 체크포인터로 배포 후에도 대화 히스토리 유지 |
| **프롬프트 인젝션 방어** | 4계층 오케스트레이션 방어 시스템 (Fine-tuning 없이 코드 레벨 구현) |

---

## 기술 스택

- **Backend**: FastAPI, Python 3.11
- **AI Orchestration**: LangGraph (`StateGraph`, `interrupt`, `Command`, `DynamoDBCheckpointer`)
- **LLM**: Google Gemini (`gemini-3-flash-preview`)
- **Embedding**: Google `gemini-embedding-001`
- **Vector DB**: Chroma (로컬 영속 스토리지) + BM25 (rank-bm25, 인메모리 싱글톤)
- **Document Store**: Amazon DynamoDB (ap-northeast-1)
- **Auth**: 세션 쿠키 + CSRF 토큰
- **Infrastructure**: AWS App Runner + Amazon ECR
- **File Storage**: Amazon S3 (사내 문서 원본)

---

## 아키텍처 개요

### 그래프 구조 (플랫, 19 노드)

```
사용자 요청
    │
    ▼
[input_guard] — 임베딩 유사도 injection 감지
    ├─[injection]→ rejection → END
    └─[pass]→ task_router — LLM 구조화 출력 라우팅 (with_structured_output)
                  │
                  ├─ knowledge_search → search → quality_check
                  │                               ├─[ok]→ answer → END
                  │                               └─[no docs]→ rewrite → search  ← 순환
                  │
                  ├─ detail_search → answer → END  ← 심화 질의 전용
                  │
                  ├─ planner ─────────────────────────────────── 복합 요청 전용
                  │    │  LLM으로 task 분해 → 선행 검색 직접 실행
                  │    ├─[rfp_draft]→  rfp_research → ...
                  │    ├─[email_draft]→ email_draft → ...
                  │    └─[knowledge_search]→ search → ...
                  │
                  ├─ ai_guide → END
                  │
                  ├─ file_chat → END
                  │
                  ├─ file_extract → END
                  │
                  ├─ email_draft → human_review ──interrupt──→ 사용자 검토
                  │                    ├─[approve]→ END
                  │                    ├─[revise]→ email_draft  ← 순환
                  │                    └─[switch]→ task_router  ← 루프백
                  │
                  ├─ rfp_research → rfp_draft → rfp_review
                  │                               ├─[notes]→ rfp_draft  ← 순환
                  │                               └─[pass]→ human_review (위와 동일)
                  │
                  ├─ rejection → END
                  │
                  └─ clarification ──interrupt──→ 사용자 의도 확인
                         ├─[rejection]→ rejection → END
                         └─[resumed]→ task_router  ← 루프백

대화 상태 (히스토리 · 첨부파일 컨텍스트)
    └── DynamoDBCheckpointer → langgraph_checkpoints 테이블 (TTL 7일)
```

---

## Planner 노드 (복합 요청 오케스트레이터)

"브릿지 프로토콜을 검색해서 RFP 초안을 작성해줘" 같이 **선행 검색 + 문서 작성**이 동시에 요구되는 복합 요청을 처리합니다.

```
복합 요청 감지 (task_router — LLM 구조화 출력)
    └─ "~검색해서 ~작성해줘" 처럼 선행 조사 + 후행 작성이 명확히 결합된 경우
         → LLM RouteDecision.task = "planner"

[planner_node]
    1. LLM | JsonOutputParser → {"tasks": ["knowledge_search", "rfp_draft"]}
    2. 선행 task (knowledge_search) 직접 실행
         └─ _search_hybrid() → sanitize_docs() → _format_docs()
         → planner_context에 누적
    3. 최종 task (rfp_draft) → task_type 설정 → route_after_planner → rfp_research 라우팅

[rfp_research_node / email_draft_node]
    └─ planner_context를 LLM 입력에 추가 컨텍스트로 주입
         → 사내 문서 검색 결과를 반영한 고품질 초안 생성
```

**감지 패턴**

| 선행 키워드 | 후행 키워드 | 결과 |
|---|---|---|
| 검색, 조회, 찾아, 알아봐, 확인해 | rfp, 제안요청서, 이메일, 메일, 초안, 작성해 | planner |
| 분석, 읽어, 요약 | rfp, 이메일, 초안, 작성해 | planner |

---

## interrupt 동작 방식 (Human-in-the-loop)

이메일·RFP 초안 작성, 의도 불명 입력 처리 시 그래프가 중간에 일시정지되고 사용자 응답을 대기합니다.

```
사용자: "인사팀에 휴가 신청 이메일 작성해줘"
  → [email_draft 실행] → interrupt → {"type": "interrupt", "draft_email": {...}} 반환

사용자: "제목 바꿔줘"
  → Command(resume="제목 바꿔줘") → [email_draft 재실행] → interrupt → 수정본 반환

사용자: "완료"
  → Command(resume="완료") → [approve 감지] → END
```

**`/chat` API 처리 흐름**

1. `get_state()`로 활성 interrupt 여부 확인
2. interrupt 활성 → `Command(resume=user_input)` 으로 그래프 재개
3. interrupt 없음 → `input_guard`부터 신규 실행
4. `invoke()` 반환 후 `get_state()` 재확인 — 새 interrupt 발생 시 구조 데이터(`draft_email` / `draft_rfp`) 포함 응답 반환
5. 완료 시 `task_type` 기반 응답 포맷 반환

**interrupt 응답 구조**

```json
{
  "type": "interrupt",
  "current_task": "email_draft",
  "message": "이메일 초안을 확인해 주세요.",
  "hint": "수정 요청 또는 '완료'로 승인하세요.",
  "draft_email": { "to": "hr@example.com", "cc": "", "subject": "...", "body": "..." },
  "draft_rfp": ""
}
```

> `draft_email` / `draft_rfp` 구조 데이터를 항상 포함하여, DynamoDB 직렬화 상태와 무관하게 프론트엔드 렌더링을 보장합니다.

**human_review 응답 분류**

| 사용자 응답 | 분류 기준 | 다음 노드 |
|------------|-----------|-----------|
| "완료", "확인", "ok" 등 | 승인 키워드 | END |
| "제목 바꿔줘", "수신자 수정" 등 | 수정 키워드 | email_draft / rfp_draft 루프백 |
| 전혀 다른 요청 (e.g. "RFP 작성해줘") | semantic route 재분류 | task_router 루프백 |

---

## RFP 3-에이전트 체인

```
rfp_research_node  — 사내 문서 ChromaDB 검색 + LLM으로 요구사항 컨텍스트 정리
                     (planner_context 있을 경우 추가 컨텍스트로 주입)
        ↓
rfp_draft_node     — rfp_research + rfp_review_notes 반영해 8개 섹션 초안 작성
        ↓
rfp_review_node    — 8개 필수 섹션 완성도 LLM 자동 검토
        ├─[미흡]→ rfp_draft (재작성 최대 1회)
        └─[통과]→ human_review (interrupt)
```

**8개 필수 섹션**: 배경/목적 · 범위(Scope) · 요구사항(기능/비기능) · 데이터/연동/보안 · 일정/마일스톤 · 산출물 · 평가 기준 · 가정/제약

---

## 라우팅 의도 카테고리

| 카테고리 | 처리 방식 |
|---|---|
| `knowledge_search` | 시맨틱+BM25 하이브리드 검색(RRF) → 결과 없을 시 쿼리 재작성 후 재검색(최대 2회) → LLM 답변 |
| `detail_search` | 직전 Q&A 기반 쿼리 재구성 + 참조 문서 필터 확장 검색 → LLM 답변 |
| `planner` | 복합 요청 분해 → 선행 검색 직접 실행 → planner_context 주입 → 최종 task 라우팅 |
| `ai_guide` | 기능 안내 전용 LLM (도구 없음) |
| `file_chat` | 첨부 파일 기반 Q&A (ReAct 에이전트) |
| `email_draft` | 이메일 초안 생성 → interrupt 검토 루프 |
| `rfp_draft` | 3-에이전트 체인 → interrupt 검토 루프 |
| `file_extract` | 파일 텍스트 추출 |
| `unknown` | fallback: ai_guide (짧은 질문/안내 요청) 또는 knowledge_search (일반 질문) |
| `clarification` | 슬롯 감지 (파일경로·수신자·범위) 또는 의도 재확인 · clarification_count≥2 → knowledge_search 강제 fallback |
| `injection` | input_guard / task_router에서 차단 → rejection |

---

## 라우팅 시스템

### 현재 방식: LLM 구조화 출력 라우팅

`task_router.py`는 LLM의 `with_structured_output(RouteDecision)` API를 사용해 사용자 요청을 분류합니다.

```
사용자 요청
    └─ LLM.with_structured_output(RouteDecision)
         → RouteDecision.task: 9개 카테고리 중 하나
         → RouteDecision.missing_slots: 실행에 필요한 누락 슬롯 목록
              ├─ slots 없음 → task_type 직접 설정
              └─ slots 있음 → task_type = "clarification" → 슬롯 질문 후 재라우팅
```

**RouteDecision 스키마**

```python
class RouteDecision(BaseModel):
    task: Literal[
        "email_draft", "rfp_draft", "knowledge_search",
        "file_chat", "file_extract", "ai_guide",
        "detail_search", "planner", "rejection",
    ]
    missing_slots: List[Literal["to", "project_scope", "file_path", "file_context"]]
```

**슬롯 기반 clarification**

| 누락 슬롯 | 질문 내용 | 발동 조건 |
|---|---|---|
| `to` | 이메일 수신자 또는 부서 | email_draft인데 수신자 특정 불가 |
| `project_scope` | RFP 프로젝트명·범위 | rfp_draft인데 범위 전혀 없을 때 |
| `file_path` | 파일 경로·파일명 | file_extract인데 경로 없을 때 |
| `file_context` | 파일 업로드 요청 | file_chat인데 파일 없다고 명시된 경우 |

루프 방어: `clarification_count ≥ 2` → `knowledge_search` 강제 fallback

### 레거시 보존: 시맨틱 라우팅 (`task_router_semantic_legacy.py`)

기존 키워드 매칭 + 임베딩 코사인 유사도 + LLM fallback 3단계 라우팅은 현재 프로덕션에 연결되어 있지 않으며, **고도화 참조 목적**으로 `app/graph/nodes/task_router_semantic_legacy.py`에 보존되어 있습니다.

재도입 고려 시나리오:
- LLM 호출 비용 절감이 최우선인 고트래픽 환경
- 라우팅 카테고리가 10개 이상으로 확장될 때 속도 우선 레이어로 활용
- 오프라인·저지연 환경에서 LLM 없이 라우팅이 필요할 때

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
│   │       ├── task_router.py     # LLM 구조화 출력 라우터 (RouteDecision, with_structured_output)
│   │       ├── planner.py         # 복합 요청 오케스트레이터 (task 분해 + 선행 검색 실행)
│   │       ├── clarification.py   # 의도 불명 시 interrupt 기반 의도 확인
│   │       ├── human_review.py    # email/rfp 공용 interrupt 검토 노드
│   │       ├── knowledge_search.py # search / quality_check / rewrite / answer 4노드 (하이브리드 검색)
│   │       ├── detail_search.py   # 심화 질의 전용 노드 (쿼리 재구성 + 문서 필터 검색)
│   │       ├── ai_guide.py        # AI 기능 안내 노드
│   │       ├── file_chat.py       # 첨부 파일 Q&A 노드 (ReAct)
│   │       ├── file_extractor.py  # 파일 텍스트 추출 노드
│   │       ├── email_agent.py     # 이메일 초안 노드 (planner_context 반영)
│   │       ├── rfp_agent.py       # RFP 3-에이전트 (research / draft / review, planner_context 반영)
│   │       └── task_router_semantic_legacy.py  # [레거시 보존] 키워드+시맨틱 라우팅 (프로덕션 미연결, 고도화 참조용)
│   │
│   ├── knowledge/
│   │   └── ingest.py              # S3 → Chroma 문서 인제스트 (PDF 페이지 단위 청킹)
│   │
│   ├── security/
│   │   ├── injection_detector.py  # 임베딩 유사도 + 슬라이딩 윈도우 injection 탐지
│   │   ├── content_sanitizer.py   # RAG 문서 · 파일 내용 sanitize
│   │   └── output_validator.py    # 응답 민감 정보 출력 검증
│   │
│   └── core/
│       ├── config.py              # 환경변수 중앙 관리
│       ├── history_utils.py       # cosine, filter_history_by_relevance
│       ├── trace_buffer.py        # 인메모리 트레이스 버퍼
│       └── log_buffer.py          # 로그 버퍼
│
├── templates/
│   ├── home.html                  # 홈페이지 (사용자 정보 + 최근 이용)
│   ├── index.html                 # 채팅 UI
│   ├── login.html                 # 로그인 페이지
│   ├── admin_home.html            # 관리자 홈
│   ├── admin_users.html           # 사용자 관리
│   ├── admin_monitor.html         # 라우팅 성능 모니터링 + 실시간 로그
│   ├── admin_graph.html           # LangGraph 구조 시각화 + 실시간 실행 흐름
│   └── pending.html               # 승인 대기 안내 페이지
│
└── static/
    ├── css/style.css
    └── js/chat.js
```

---

## GraphState 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `input_data` | str | 사용자 입력 (clarification 재개 시 combined 값으로 교체) |
| `task_type` | str | 라우팅 결과 카테고리 |
| `task_args` | Dict | 라우팅 디버그 정보, 검색 문서 등 |
| `messages` | Sequence[BaseMessage] | 대화 히스토리 (누적 append) |
| `citations_used` | List[Dict] | 인용된 출처 목록 |
| `extracted_text` | str | 파일 추출 원문 텍스트 |
| `extracted_meta` | Dict | 파일 메타데이터 |
| `file_context` | str | 업로드 파일 텍스트 (State에 영속화) |
| `file_context_name` | str | 업로드 파일명 |
| `draft_email` | Dict | 이메일 초안 (to/cc/subject/body) |
| `draft_rfp` | str | RFP 초안 텍스트 |
| `rfp_research` | str | rfp_research_node 수집 결과 |
| `rfp_review_notes` | str | rfp_review_node 검토 의견 (재작성 시 반영 후 초기화) |
| `retry_count` | int | knowledge search 재시도 횟수 |
| `clarification_count` | int | clarification 발동 횟수 (루프 방어용, ≥2 → knowledge_search fallback) |
| `review_action` | str | human_review 결과 (approve/revise/switch) |
| `current_task` | str | human_review switch 감지 기준 |
| `interrupt_type` | str | "clarification" 또는 "human_review" |
| `task_sequence` | List[str] | planner가 결정한 실행 순서 |
| `planner_context` | str | planner 선행 검색 결과 컨텍스트 |
| `trace_id` | str | 요청별 트레이스 ID |

---

## 프롬프트 인젝션 방어

Fine-tuning 없이 오케스트레이션 레벨에서 4계층 방어를 구현합니다. 추가 LLM 호출 비용 없이 동작합니다.

| 레이어 | 위치 | 방식 | 차단 대상 |
|---|---|---|---|
| **1차** | `input_guard_node` (그래프 첫 노드) | 임베딩 유사도 + 슬라이딩 윈도우 | 알려진 패턴 · 분할 인젝션 · 소셜 엔지니어링 |
| **2차** | `task_router` → `rejection_node` | 라우팅 `injection` task_type | Semantic 변형 우회 · 범위 외 질문 |
| **3차** | `search_node`, `rfp_research_node` | RAG 문서 · 파일 내용 sanitize | 문서/파일 경유 간접 인젝션 |
| **4차** | `answer_node` | 응답 출력 규칙 기반 검증 | 1~3차 통과 후 민감 정보 노출 |

**요청 간 컨텍스트 격리**

각 LLM 호출 노드(knowledge_search, email_draft, rfp_draft, file_chat, ai_guide)는 이전 대화 히스토리를 LLM에 주입하지 않습니다. 요청마다 독립적인 컨텍스트로 처리하여 이전 대화 내용이 현재 답변에 오염되는 것을 방지합니다. interrupt/resume에 필요한 draft 상태 및 파일 컨텍스트는 DynamoDB 체크포인터를 통해 별도로 유지됩니다.

---

## 환경변수

`.env` 파일을 프로젝트 루트에 생성하여 아래 변수를 설정합니다.

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

# 대화 히스토리
HISTORY_MAX_MESSAGES=40
HISTORY_RELEVANCE_THRESHOLD=0.40
HISTORY_ALWAYS_KEEP_LAST_N=0
CHECKPOINT_TTL_DAYS=7
CHECKPOINT_MAX_MESSAGES=40

# 프롬프트 인젝션 방어 임계값
INJECTION_THRESHOLD_SINGLE=0.80
INJECTION_THRESHOLD_COMBINED=0.76
INJECTION_WINDOW_TURNS=3

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
FastAPI 앱 시작 (main.py)
    ├── DynamoDB 테이블 자동 생성
    ├── intent_samples seed 데이터 upsert
    ├── S3 → Chroma 문서 인제스트 (AUTO_INGEST=1)
    └── FastAPI 서버 Ready
```

---

## 관리자 기능

- `/admin` — 관리자 홈
- `/admin/users` — 사용자 목록 조회, 승인/거절, 관리자 권한 토글, 소속(department) 지정, 계정 삭제
- `/admin/monitor` — 라우팅 통계(소스·태스크 분포, 평균 score/margin), 라우팅 로그, 실시간 서버 로그 스트리밍
- `/admin/graph` — LangGraph 그래프 구조 시각화(Mermaid.js) + 실시간 노드 실행 흐름 모니터링(SSE)

**라우팅 이력 조회**

관리자 모니터 페이지에서 라우팅 결정 이력(입력·분류 결과·LLM 라우터 디버그 정보)을 조회할 수 있습니다. DynamoDB `routing_logs` 테이블에 요청별 로그가 기록됩니다.

---

## 라이선스

Internal use only.
