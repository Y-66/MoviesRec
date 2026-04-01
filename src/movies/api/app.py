from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver

from movies.api.routers.chat import router as chat_router
from movies.api.routers.system import router as system_router
from movies.graph import build_graph


def create_app(base_dir: Path | None = None) -> FastAPI:
    # app.py is under src/movies/api, so parents[4] is the repository root.
    project_dir = base_dir or Path(__file__).resolve().parents[3]

    app = FastAPI(
        title="Movies Recommendation Agent API",
        description="Enterprise-ready API for Collaborative Filtering-based conversational movie recommendations",
        version="1.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.base_dir = project_dir
    app.state.agent = build_graph(checkpointer=MemorySaver())

    app.include_router(system_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    return app


app = create_app()
