# app/graph/subgraphs

## OVERVIEW
4 task-specific LangGraph subgraphs.

## STRUCTURE
```
subgraphs/
├── chat_graph.py      # General chat workflow
├── file_graph.py      # File extraction workflow
├── email_graph.py     # Email drafting workflow
└── rfp_graph.py       # RFP drafting workflow
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Modify chat flow | `chat_graph.py` | Default subgraph |
| Add new subgraph | Copy existing pattern | Follow StateGraph composition |
| Change subgraph routing | `task_router.py` (parent nodes/) | Controls which subgraph executes |

## CONVENTIONS
- Each subgraph builds its own StateGraph
- Subgraphs composed into main workflow in main.py
- Uses TypedDict for state (defined in app/graph/states/)

## ANTI-PATTERNS
- **DO NOT** duplicate shared logic across subgraphs
- Extract common nodes to app/graph/nodes/
