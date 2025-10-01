from __future__ import annotations

from fastapi import FastAPI

from src.app.config import get_settings
from src.app.routes import router
from src.utils.logging import configure_logging

settings = get_settings()
configure_logging()

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(router)