# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-03
**Project:** Langgraph-Rag

## OVERVIEW
FastAPI + LangGraph RAG application with multi-subgraph workflow (chat, file_extract, email_draft, rfp_draft). Uses Google Gemini for LLM, Chroma for vector storage, DynamoDB for auth.

## STRUCTURE
```
./
├── main.py                    # FastAPI + LangGraph entry (port 8080)
├── app/
│   ├── auth/                  # DynamoDB-backed auth
│   ├── core/                  # Config, embeddings
│   ├── graph/                 # LangGraph workflow
│   │   ├── nodes/             # 8 agent nodes
│   │   ├── subgraphs/         # 4 task subgraphs
│   │   └── states/            # GraphState TypedDict
│   └── knowledge/             # RAG ingest
├── templates/                 # Jinja2 HTML
└── static/                    # CSS, JS
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new LangGraph node | `app/graph/nodes/` | Add node fn, update state |
| Modify auth flow | `app/auth/routes.py` | Login/logout/register |
| Change LLM model | `app/core/config.py` | `has_gemini_api_key` |
| Add new subgraph | `app/graph/subgraphs/` | Copy existing pattern |
| RAG pipeline | `app/knowledge/ingest.py` | Chroma ingestion |

## CONVENTIONS (THIS PROJECT)
- **LangGraph**: StateGraph with TypedDict state, subgraph composition
- **Auth**: DynamoDB user table, bcrypt password hashing, Starlette sessions
- **Korean comments**: Code uses Korean for important warnings ("수정 금지" = do not modify)
- **Entry point**: `python main.py` (no `__main__.py`)
- **No tests**: Project has zero test coverage

## ANTI-PATTERNS (THIS PROJECT)
- **DO NOT** modify `task_type` or `next_node` in analyzer (per Korean comment)
- **DO NOT** commit hardcoded API keys (see `app/core/config.py`)
- **DO NOT** include `chroma_db/` in git (runtime data)
- **DO NOT** use English TODO/FIXME markers (not used in this project)

## COMMANDS
```bash
# Dev
python main.py

# Docker
docker build -t langgraph-rag .
docker run -p 8080:8080 langgraph-rag
```

## NOTES
- Requires: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Auth requires: `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `SESSION_SECRET`, `USERS_TABLE` (DynamoDB)
- Default session secret: "dev-secret-change-me" (MUST change in prod)
- Chroma DB at `./chroma_db/` (add to .gitignore)
