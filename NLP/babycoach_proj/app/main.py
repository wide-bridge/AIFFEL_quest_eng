from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.activity import router as activity_router
from .api.baby_profile import router as baby_profile_router
from .api.babydoc import router as babydoc_router
from .api.chat import router as chat_router
from .api.recommend import router as recommend_router
from .db import init_db
from .graph import get_compiled_graph
from .ui.app_ui import get_ui_html


def create_app() -> FastAPI:
    app = FastAPI(title="BabyCoach PoC")
    init_db()

    # Static assets (icons/images). Path is fixed and used in UI only.
    assets_dir = os.path.join(os.path.dirname(__file__), "ui", "assets")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    app.include_router(recommend_router)
    app.include_router(babydoc_router)
    app.include_router(chat_router)
    app.include_router(baby_profile_router)
    app.include_router(activity_router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return get_ui_html()

    hitl_path = Path(__file__).resolve().parent / "ui" / "babycoach_hitl.html"

    @app.get("/hitl", response_class=HTMLResponse)
    def hitl_review_ui() -> str:
        if hitl_path.exists():
            return hitl_path.read_text(encoding="utf-8")
        return "<!doctype html><meta charset=utf-8><title>HITL</title><p>babycoach_hitl.html not found.</p>"

    @app.get("/health")
    def health() -> dict:
        # Ensure graph can compile at runtime (smoke check).
        get_compiled_graph()
        return {"status": "ok"}

    return app


app = create_app()

