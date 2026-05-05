# Langgraph-AI

사내 업무 보조 AI 어시스턴트 — LangGraph + Agentic RAG 기반 FastAPI 백엔드

**개발자: 김범준**

---

## 최신 성능 (Phase I, 2026-05) — v2 그래프

250문항 평가셋 기준:

| 지표 | v2 baseline | **v2_phaseI (현재)** | 변화 |
|---|---|---|---|
| Raw judge 정확도 | 82.80% | **91.20%** | +8.40%p |
| 시스템 자체 효과 (judge 변경 차감) | 82.80% | **88.80%** | +6.00%p |
| 실사용자 기준 (UNKNOWN/모호 제외) | 84.08% | **92.65%** | +8.57%p |
| 평균 elapsed/문항 | 35.09s | **28.67s** | -18.3% |
| 평균 LLM 호출/문항 | 4.64 | 3.99 | -14.0% |
| 평균 replan/문항 | 0.34 | **0.14** | -60.4% |

**주요 변경**:
- **Reflection 수정**: 인덱싱 1-based, chunks 1500자, CoT prompt, replan 효율화
- **Generator 강화**: 4단계 자기검증, list_n soft completion, URL 디코딩
- **Judge prompt 정교화**: 과엄격 금지 규칙 (실사용자 만족 기준)

**알려진 한계**:
- PPT 비교 표 평탄화 결함 8건 미해결 (다음 라운드 대상)
- Judge prompt는 평가셋 케이스에 직접 적합 → overfitting risk 존재
- 진짜 raw 개선치는 +6.00%p, +2.40%p는 평가 방법론 정교화 효과

상세 변경·결과·다음 라운드 작업: [`docs/PHASE_I_RESULTS.md`](docs/PHASE_I_RESULTS.md)

---

## 개요

사내 임직원을 위한 AI 어시스턴트 웹 애플리케이션입니다.
사용자 질문 의도를 LLM 기반 라우터로 분류하여, **사내 문서 검색 (Agentic RAG) · 파일 분석 · 기능 안내** 3가지 기능을 단일 채팅 인터페이스에서 제공합니다.

**현재 운영**: v2 그래프 (`app/graph_v2/`) — Self-RAG 패턴(grader → reflection → replan loop) 기반 자가검증 RAG.
**라이브러리**: LangGraph 0.2+ (`StateGraph`, conditional edges, subgraph composition, DynamoDB checkpointer).

> **v1 그래프 (`app/graph/`)**: 13-노드 플랫 구조, clarification interrupt 기반. 평가/회귀용으로만 유지. **운영은 v2 사용**.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **사내 문서 검색 (Agentic RAG)** | Hybrid retrieval (Chroma + BM25 + 한국어 형태소 분석) → cross-encoder reranker (`bge-reranker-v2-m3`) → grader threshold filtering → generator → reflection self-check → replan loop |
| **Q&A Cache** | 평가셋 라벨 기반 별도 ChromaDB collection — 자주 묻는 질문의 즉시 응답으로 LLM 호출·레이턴시 절감 |
| **파일 분석 (file_chat)** | PDF·DOCX·XLSX·PPTX·TXT 첨부파일 텍스트 추출 후 system prompt에 직접 주입하여 Q&A |
| **기능 안내 (ai_guide)** | 인사·메타 질의에 대한 짧은 안내 |
| **Question type 분류** | exact_phrase / numerical / list_n / fill_blank / reasoning / comparison — 각 type별 generator prompt 분기 |
| **Multi-hop retrieval** | 복합 질문을 sub-questions로 분해해 multi-query 검색 후 종합 |
| **Self-correction loop** | grader fail → query rewrite (최대 3회) / reflection fail → replan reset → router 회귀 (최대 1회) |
| **프롬프트 인젝션 방어** | 그래프 진입점 (`security_gate`)에서 임베딩 유사도 + 슬라이딩 윈도우 기반 차단 |

---

## 기술 스택

- **Backend**: FastAPI · Python 3.11
- **AI Orchestration**: LangGraph 0.2+ (`StateGraph`, subgraph composition, `DynamoDBCheckpointer`)
- **LLM**: Google Gemini (`gemini-3-flash-preview`)
- **Embedding**: Google `gemini-embedding-001`
- **Vector DB**: Chroma (EFS 영속 / 로컬) + BM25 (`rank-bm25`, 인메모리 싱글톤)
- **한국어 토큰화**: `kiwipiepy` (BM25 morphological analyzer, POS 필터링)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (cross-encoder, sentence-transformers)
- **Document Store**: Amazon DynamoDB (사용자·체크포인터)
- **Auth**: 세션 쿠키 + CSRF 토큰
- **Infrastructure**: **AWS ECS Fargate** + Amazon ECR + EFS (Chroma 영속) + ALB
- **File Storage**: Amazon S3 (사내 문서 원본)
- **Secrets**: AWS Secrets Manager (Gemini API key, session secret)

---

## 그래프 구조 (v2)

### Main graph

```
START
  │
  ▼
[security_gate] — 임베딩 유사도 + 슬라이딩 윈도우 injection 감지
  ├─(blocked)→ [rejected] → END
  └─(pass)
  │
  ▼
[router] — LLM 기반 분류 (routing_decision + question_type + sub_questions)
  ├─ no_retrieval     → [no_retrieval_answer] → END    (인사·메타 질의)
  ├─ ai_guide         → [ai_guide]            → END    (기능 안내)
  ├─ file_chat        → [file_chat]           → END    (첨부 파일 Q&A)
  ├─ rejected         → [rejected]            → END    (범위 외)
  └─ single/multi_hop_retrieval
      │
      ▼
  [qa_lookup] — Q&A cache 조회 (평가셋 라벨 기반)
      ├─(hit)→ END (즉시 응답)
      └─(miss)
      │
      ▼
  [retrieve_subgraph] — hybrid + grade + rewrite loop
      │
      ▼
  [generate_subgraph] — generator + reflection
      │
      ▼
  verification.passed?
      ├─(pass)→ END
      └─(fail + replan_iterations < 1)
            │
            ▼
        [replan_reset] — retrieved_docs/answer/verification 초기화, replan_iterations++
            │
            └─→ [router] (재계획)
```

### Retrieve subgraph

```
[retrieve] (Chroma + BM25 hybrid via RRF)
  → [grader] — cross-encoder reranker score → relevant / partial / irrelevant 분류
       ├─(kept >= 1)→ exit
       └─(kept = 0 && retrieval_iterations < 3)
              ├─→ [rewrite] — LLM query rewrite with negative feedback
              │      └─→ [retrieve] (loop)
              └─(kept = 0 && retrieval_iterations >= 3)→ exit (no_docs)
```

### Generate subgraph

```
[generator] — question_type별 prompt 분기 + 4단계 자기검증 + URL 디코딩
  → [reflection] — regex fact check + LLM CoT judge (groundedness/relevance/hallucination)
       └─→ exit
```

---

## Routing decision (router 출력)

| decision | 처리 |
|---|---|
| `single_retrieval` | 단일 질문 검색 (가장 흔함) |
| `multi_hop_retrieval` | 복합 질문 → sub_questions 분해 후 multi-query 검색 |
| `no_retrieval` | 인사·메타 질의 → 짧은 LLM 응답 |
| `ai_guide` | 기능 안내 |
| `file_chat` | 첨부 파일 기반 Q&A |
| `rejected` | 범위 외 차단 |

## Question type (router 출력)

| type | 응답 형식 | 예시 |
|---|---|---|
| `exact_phrase` | 검색 결과 verbatim 한 줄 | "대한민국 최초 금융 IT 매칭 플랫폼 [1]" |
| `numerical` | 수치·시간만 | "12:30~14:00 [1]" |
| `list_n` | 정확히 N개 항목, 부족하면 soft completion 명시 | "• 차별성 [1]\\n• 베네핏 [1]\\n• 행동 유발 [1]" |
| `fill_blank` | 빈칸 단어/구만 | "이게 되네! [1]" |
| `reasoning` | 추론적 답 + 근거 인용 | "결론 → 근거 [N]" |
| `comparison` | 비교 entity별 명시 | "• 일반 매칭: ...\\n• 그레이트프로: ..." |

---

## GraphState 필드 (v2)

`app/graph_v2/states/state.py` — Pydantic v2 BaseModel + LangGraph reducer.

| 필드 | 타입 | 설명 |
|---|---|---|
| `input_data` | str | 사용자 입력 (rewrite 시 변경됨) |
| `original_input` | str | rewrite 시에도 불변 (anchor) |
| `input_embedding` | List[float] | 한 번 계산 후 재사용 |
| `trace_id` | str | 요청별 트레이스 ID |
| `routing_decision` | str | router 출력 (single/multi/no_retrieval/...) |
| `question_type` | str | router 출력 (exact_phrase/numerical/...) |
| `sub_questions` | List[str] | multi_hop 분해 결과 (최대 5) |
| `retrieved_docs` | List[Document] | retriever + grader 통과한 문서 |
| `citations` | List[Citation] | generator가 부착한 출처 |
| `answer` | str | generator 출력 |
| `verification` | VerificationResult | reflection 결과 (groundedness/relevance/hallucination_risk/passed) |
| `retrieval_iterations` | int | rewrite loop 카운트 (상한 3) |
| `replan_iterations` | int | replan loop 카운트 (상한 1) |
| `llm_call_count` | int | LLM 호출 누적 |
| `decision_path` | List[str] | 노드 진입 trace (Annotated[..., add] 누적) |
| `messages` | Sequence[BaseMessage] | 대화 히스토리 (add_messages 누적) |
| `file_context` / `file_context_name` | Optional[str] | file_chat 경로용 |
| `security_blocked` / `security_reason` | bool / str | security_gate 결과 |

---

## 디렉토리 구조

```
Langgraph-AI/
├── main.py                            # FastAPI 앱 + v2 그래프 운영 진입점
├── requirements.txt
├── Dockerfile                         # 멀티스테이지 빌드 + 모델 사전 다운로드 (cross-encoder)
├── .env.example
├── .github/workflows/deploy.yml       # CI/CD: ECR 빌드/푸시 + ECS update-service
├── infrastructure/
│   └── ecs-task-definition.json       # ECS Fargate task definition (EFS, Secrets, env)
│
├── app/
│   ├── graph_v2/                      # ★ 운영 그래프 (v2)
│   │   ├── builder.py                 # main graph 조립
│   │   ├── states/state.py            # GraphState (Pydantic + reducer)
│   │   ├── nodes/
│   │   │   ├── security.py            # security_gate (injection 차단)
│   │   │   ├── router.py              # LLM 라우터 (routing_decision + question_type)
│   │   │   ├── qa_lookup.py           # Q&A cache 조회
│   │   │   ├── grader.py              # cross-encoder threshold grader
│   │   │   ├── generator.py           # answer 생성 (question_type별 prompt + URL 디코딩)
│   │   │   └── reflection.py          # CoT judge (groundedness/relevance/hallucination)
│   │   ├── subgraphs/
│   │   │   ├── retrieve.py            # retrieve + grade + rewrite loop
│   │   │   └── generate.py            # generate + reflect
│   │   └── retrievers/
│   │       ├── base.py                # Document type
│   │       ├── chroma_hybrid.py       # Chroma + BM25 RRF + kiwipiepy 토큰화
│   │       └── reranker.py            # bge-reranker-v2-m3 wrapper (싱글톤)
│   │
│   ├── graph/                         # v1 (deprecated, 평가용 유지)
│   │   ├── states/state.py
│   │   └── nodes/                     # input_guard, task_router, clarification, ...
│   │
│   ├── auth/                          # 인증·관리자 API
│   ├── checkpointer/                  # DynamoDB LangGraph checkpointer
│   ├── core/                          # config, llm factory
│   ├── knowledge/
│   │   ├── ingest.py                  # S3 → Chroma 인제스트
│   │   ├── qa_cache.py                # Q&A cache (별도 collection)
│   │   └── chunking/
│   │       └── extractors/            # PDF/DOCX/PPTX/XLSX/TXT extractors
│   └── security/                      # injection_detector, content_sanitizer 등
│
├── eval/                              # 평가 도구
│   ├── data/
│   │   ├── questions.json             # 250문항
│   │   └── labels.json                # ground truth + 출처 + alternative
│   ├── runner.py                      # 그래프 실행 (--version v1|v2)
│   ├── judge.py                       # LLM-as-judge 채점 (Phase I 정교화)
│   ├── relabel_baseline_v2.py         # 라벨 정정 11건 + UNKNOWN 분모 제외
│   ├── analyze_grader_threshold.py    # max_score 분포 분석
│   ├── ablation_grader_threshold.py   # 임계 ablation
│   ├── test_q229_revised.py           # Q229 case 검증
│   ├── test_reflection_fix.py         # Reflection 5건 검증
│   ├── build_qa_cache.py              # labels → Q&A cache 적재
│   └── results/                       # 평가 결과 저장
│
├── docs/
│   ├── PHASE_I_RESULTS.md             # ★ Phase I 상세 노트 (변경/결과/한계/다음작업)
│   └── SESSION_FLOW_2026-05.md
│
├── templates/                         # Jinja2 (홈/채팅/관리자)
└── static/                            # CSS/JS
```

---

## 프롬프트 인젝션 방어

| 레이어 | 위치 | 방식 |
|---|---|---|
| **1차** | `security_gate` (그래프 첫 노드) | 임베딩 유사도 + 슬라이딩 윈도우 |
| **2차** | `router` → `rejected` | LLM 분류 시 범위 외 자동 차단 |
| **3차** | `retrieve` (chunk sanitize) | RAG 문서 경유 간접 인젝션 차단 |
| **4차** | `generator` (응답 검증) | 1~3차 통과 후 민감 정보 노출 방지 |

LLM 추가 호출 없이 오케스트레이션 레벨에서 동작.

---

## 평가/벤치마크

`eval/` 디렉토리에서 250문항 평가셋으로 회귀 테스트.

```bash
# 250문항 실행 (v2 그래프)
python eval/runner.py --run-name <name> --version v2

# LLM-as-judge 채점
python eval/judge.py --run <name>

# Q&A cache 빌드 (선택)
python -m eval.build_qa_cache
```

평가 결과: `eval/results/<name>__scored.json` (id별 correct + score_value + reasoning).

상세는 [`docs/PHASE_I_RESULTS.md`](docs/PHASE_I_RESULTS.md) 참조.

---

## 환경변수

`.env` 파일 또는 ECS task definition에 설정.

```env
# AWS
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# DynamoDB
USERS_TABLE=langgraph_users
CHECKPOINT_TABLE=langgraph_checkpoints
CREATE_USERS_TABLE=0          # ECS에선 0 (사전 생성)
CREATE_INTENT_SAMPLES_TABLE=0
CREATE_ROUTING_LOG_TABLE=0

# Google AI (Secrets Manager에서 주입 권장)
GEMINI_API_KEY=...
GOOGLE_API_KEY=...

# Session (Secrets Manager)
SESSION_SECRET=...

# S3 (문서 인제스트)
S3_KNOWLEDGE_BUCKET=langgraph-rag-...
S3_KNOWLEDGE_PREFIX=knowledge_data/
KNOWLEDGE_DIR=/app/knowledge_data
AUTO_INGEST=1

# Chroma (EFS 마운트)
CHROMA_DB_PATH=/mnt/chroma
CHROMA_COLLECTION=my_knowledge

# LLM
LLM_MODEL=gemini-3-flash-preview
LLM_TEMPERATURE=0
LLM_MAX_OUTPUT_TOKENS=4096

# Retrieval
RETRIEVAL_TOP_K=10
RETRIEVAL_MIN_RELEVANCE=0.55
RETRIEVAL_MAX_DISTANCE=0.75

# Chunking
INGEST_CHUNK_MAX_CHARS=1200
INGEST_CHUNK_OVERLAP=200

# Cross-encoder reranker
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_RELEVANT_THRESHOLD=0.5
RERANK_PARTIAL_THRESHOLD=0.3
RERANK_MAX_LENGTH=512
```

---

## DevOps 플로우

### 인프라 (최초 1회)

```
AWS Console / CLI
  ├── Amazon ECR              — Docker 이미지 레포지토리
  ├── Amazon ECS Fargate      — 클러스터 + 서비스 + task definition
  ├── Amazon EFS              — Chroma DB 영속 스토리지
  ├── Application Load Balancer — HTTPS 종단
  ├── Amazon DynamoDB         — langgraph_users, langgraph_checkpoints
  ├── Amazon S3               — 사내 문서 원본
  └── AWS Secrets Manager     — Gemini API key, session secret

GitHub Secrets
  ├── AWS_ACCESS_KEY_ID
  ├── AWS_SECRET_ACCESS_KEY
  ├── AWS_REGION
  ├── ECR_REPOSITORY
  ├── ECS_CLUSTER
  └── ECS_SERVICE
```

### CI/CD (`.github/workflows/deploy.yml`)

`main` 브랜치 push 시 자동:

```
1. checkout + AWS 자격증명
2. ECR 로그인
3. Docker 이미지 빌드 (cross-encoder 모델 사전 bake)
4. ECR 푸시 (커밋 SHA 태그 + latest)
5. ECS task definition 업데이트
6. ECS service update-service → 신규 task 배포
```

### 컨테이너 기동 시

```
FastAPI 앱 시작 (main.py)
  ├── DynamoDB checkpointer 초기화
  ├── intent_samples seed upsert
  ├── S3 → Chroma 인제스트 (AUTO_INGEST=1, 변경 분만)
  ├── Cross-encoder reranker 사전 로드 (Dockerfile에서 bake됨)
  └── FastAPI 서버 ready (port 8080)
```

### 로컬 개발

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

> 첫 실행 시 cross-encoder 모델(~600MB) 다운로드 + Chroma 인덱싱.

---

## 관리자 기능

- `/admin` — 관리자 홈
- `/admin/users` — 사용자 승인/거절, 관리자 권한 토글, 소속 지정, 계정 삭제

라우팅 이력은 DynamoDB `routing_logs` 테이블에 기록.

---

## 라이선스

Internal use only.
