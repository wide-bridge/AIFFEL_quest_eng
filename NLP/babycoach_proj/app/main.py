from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.chat import router as chat_router
from .api.recommend import router as recommend_router
from .graph import get_compiled_graph
from .ui.app_ui import get_ui_html


def create_app() -> FastAPI:
    app = FastAPI(title="BabyCoach PoC")

    # Static assets (icons/images). Path is fixed and used in UI only.
    assets_dir = os.path.join(os.path.dirname(__file__), "ui", "assets")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    app.include_router(recommend_router)
    app.include_router(chat_router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return get_ui_html()

    @app.get("/health")
    def health() -> dict:
        # Ensure graph can compile at runtime (smoke check).
        get_compiled_graph()
        return {"status": "ok"}

    return app


app = create_app()

