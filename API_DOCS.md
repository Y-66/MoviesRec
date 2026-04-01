# Movies Recommendation Agent API Documentation

## 1. Overview
This document defines the backend API contract for the MoviesRec conversational recommendation system.

The API is implemented with FastAPI and organized with enterprise-style layering in `src/movies/api`:
- Routers: request/response boundary
- Schemas: strongly typed contracts
- Services: reusable business logic
- Dependencies: app container and shared runtime resources

Core pipeline:
- Intent analysis
- Conditional SQL hard filtering (if hard filters are present)
- Collaborative filtering (placeholder interface)
- Diversity filtering (placeholder interface)
- Summarization

## 2. Base URL and Versioning
Base URL (local):
```text
http://localhost:8000
```

API prefix:
```text
/api/v1
```

## 3. Memory and Session Model
In-session memory:
- Managed by LangGraph checkpointer (`thread_id = session_id`)
- Backend only sends current user turn to graph invoke
- Previous context is restored internally by LangGraph runtime

History persistence:
- Chat history is saved to root `chat_history/` as JSON files
- Purpose: frontend rendering and audit trail
- File history is not manually concatenated into each model call

## 4. Unified Endpoint List
### 4.1 System Endpoints
1. `GET /api/v1/system/health`
2. `GET /api/v1/system/capabilities`

### 4.2 Chat Endpoints
1. `POST /api/v1/chat`
2. `POST /api/v1/chat/stream` (SSE)
3. `GET /api/v1/chat/sessions`
4. `GET /api/v1/chat/history/{session_id}`
5. `DELETE /api/v1/chat/sessions/{session_id}`

## 5. Detailed API Contracts
### 5.1 Health Check
Method and path:
```text
GET /api/v1/system/health
```

Response 200:
```json
{
  "status": "ok",
  "service": "movies-recommendation-agent",
  "version": "1.1.0"
}
```

### 5.2 Capability Discovery
Method and path:
```text
GET /api/v1/system/capabilities
```

Response 200:
```json
{
  "service": "movies-recommendation-agent",
  "memory_mode": "langgraph-thread-memory + file-history-for-frontend",
  "features": [
    "intent-analysis",
    "conditional-sql-filter",
    "collaborative-filter-placeholder",
    "diversity-filter-placeholder",
    "response-summarization",
    "session-history-list",
    "session-history-detail",
    "session-history-delete"
  ]
}
```

### 5.3 Chat Turn
Method and path:
```text
POST /api/v1/chat
```

Request body:
```json
{
  "user_input": "Recommend a sci-fi movie after 2018.",
  "session_id": "user_42"
}
```

Field definitions:
- `user_input` (string, required): current user utterance
- `session_id` (string, optional, default `default`): conversation thread id

Response 200:
```json
{
  "session_id": "user_42",
  "request_id": "3f3c796f-6cb6-4f69-9f64-1d50ad2a6df4",
  "created_at": "2026-03-31T12:08:14.132211+00:00",
  "response": "You may enjoy Dune and Interstellar...",
  "latency_ms": 824,
  "message_count": 20,
  "stage_trace": [
    "intent_analyzer",
    "sql_filter",
    "collaborative_filter",
    "diversity_filter",
    "summarizer"
  ],
  "recommendation_count": 2,
  "recommendation_cards": [
    {
      "id": 101,
      "title": "Dune"
    },
    {
      "id": 233,
      "title": "Interstellar"
    }
  ],
  "intent_data": {
    "intent": "recommendation",
    "hard_filters": {
      "genre": "Sci-Fi",
      "year": 2018
    },
    "response": null
  }
}
```

Error 500:
```json
{
  "detail": "<runtime error message>"
}
```

### 5.4 Streaming Chat (SSE)
Method and path:
```text
POST /api/v1/chat/stream
```

Headers:
```text
Accept: text/event-stream
```

Request body:
```json
{
  "user_input": "Give me a relaxing movie for tonight",
  "session_id": "user_42"
}
```

SSE event flow (ordered):
1. `start`: request/session metadata
2. `stage`: node-level progress (`intent_analyzer`, `sql_filter`, etc.)
3. `token`: frontend incremental text chunks
4. `final`: full structured response payload
5. `done`: stream end marker
6. `error`: error payload (if any)

SSE response headers:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache, no-transform`
- `Connection: keep-alive`
- `X-Accel-Buffering: no`
- `Content-Encoding: identity`

Example SSE frame:
```text
event: stage
data: {"request_id":"...","session_id":"user_42","stage":"intent_analyzer"}
```

Buffering troubleshooting checklist:
1. If using Nginx, disable proxy buffering for this location.
2. Do not apply `GZipMiddleware` to SSE routes.
3. Avoid middlewares that consume full response body before sending.
4. Ensure frontend uses streaming readers (`EventSource` or fetch+ReadableStream) and does not wait for full body.

### 5.5 List Sessions
Method and path:
```text
GET /api/v1/chat/sessions
```

Response 200:
```json
[
  {
    "session_id": "user_42",
    "updated_at": "2026-03-31T08:20:43.832211+00:00",
    "message_count": 14,
    "file": "user_42.json"
  }
]
```

### 5.6 Session History Detail
Method and path:
```text
GET /api/v1/chat/history/{session_id}
```

Response 200:
```json
{
  "session_id": "user_42",
  "updated_at": "2026-03-31T08:20:43.832211+00:00",
  "message_count": 14,
  "messages": [
    {
      "type": "human",
      "data": {
        "content": "Recommend a sci-fi movie after 2018.",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": null,
        "id": null,
        "example": false
      }
    }
  ]
}
```

Behavior:
- If the session file does not exist, an empty payload is returned with `message_count = 0`

### 5.7 Delete Session History
Method and path:
```text
DELETE /api/v1/chat/sessions/{session_id}
```

Response 200:
```json
{
  "session_id": "user_42",
  "removed": true
}
```

Behavior notes:
- `removed = false` when file does not exist
- This endpoint deletes local persisted history file only

## 6. Runtime Entry
Server entry remains in root `main.py` and only starts the service.

Run:
```bash
python main.py
```

Alternative:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 7. Frontend Integration Notes
Recommended frontend call sequence:
1. On app startup: call `GET /api/v1/system/health`
2. Load sidebar sessions: call `GET /api/v1/chat/sessions`
3. Load selected history: call `GET /api/v1/chat/history/{session_id}`
4. Send messages in streaming mode: call `POST /api/v1/chat/stream` (SSE)
5. Fallback to standard request-response: call `POST /api/v1/chat`
6. Delete session: call `DELETE /api/v1/chat/sessions/{session_id}`

## 8. Source Code Map
- App factory: `src/movies/api/app.py`
- Chat router: `src/movies/api/routers/chat.py`
- System router: `src/movies/api/routers/system.py`
- API schemas: `src/movies/api/schemas/chat.py`, `src/movies/api/schemas/system.py`
- Startup entry: `main.py`
