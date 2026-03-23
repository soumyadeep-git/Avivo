"""
FastAPI application for Telegram webhook processing and deployment health checks.
"""

import asyncio
import logging
from typing import Any, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from telegram import Update
from telegram.ext import Application

from bot import build_application, get_rag_engine
from config import settings
from logging_utils import configure_logging, log_event


configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Avivo Telegram RAG Bot", version="2.0.0")

_telegram_application: Application | None = None
_application_lock = asyncio.Lock()


async def get_telegram_application() -> Application:
    """Initialize the Telegram application lazily for webhook/serverless execution."""
    global _telegram_application

    if _telegram_application is not None:
        return _telegram_application

    async with _application_lock:
        if _telegram_application is None:
            application = build_application(use_webhook_transport=True)
            await application.initialize()
            await application.start()
            _telegram_application = application
            log_event(
                logger,
                logging.INFO,
                "telegram_webhook_application_initialized",
                deployment_mode=settings.deployment_mode,
            )
    return _telegram_application


@app.on_event("startup")
async def startup_event() -> None:
    """Optionally register the webhook without hard-failing startup."""
    if settings.deployment_mode == "webhook" and settings.auto_set_webhook:
        application = await get_telegram_application()
        await application.bot.set_webhook(
            url=settings.telegram_webhook_url,
            secret_token=settings.telegram_webhook_secret,
            allowed_updates=Update.ALL_TYPES,
        )
        log_event(
            logger,
            logging.INFO,
            "telegram_webhook_registered",
            webhook_url=settings.telegram_webhook_url,
        )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Gracefully stop the Telegram application."""
    global _telegram_application
    if _telegram_application is not None:
        await _telegram_application.stop()
        await _telegram_application.shutdown()
        _telegram_application = None
        log_event(logger, logging.INFO, "telegram_webhook_application_stopped")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Shallow health endpoint for container and serverless checks."""
    return {
        "status": "ok",
        "environment": settings.app_env,
        "deployment_mode": settings.deployment_mode,
        "vector_backend": settings.vector_backend,
    }


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    """Deep readiness endpoint including RAG/vector backend health."""
    try:
        rag_engine = get_rag_engine()
        return {
            "status": "ready",
            "rag": rag_engine.health(),
        }
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="Application is not ready") from exc


@app.post(settings.telegram_webhook_path)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> Dict[str, bool]:
    """Process Telegram webhook requests."""
    if settings.telegram_webhook_secret and (
        x_telegram_bot_api_secret_token != settings.telegram_webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")

    application = await get_telegram_application()
    payload = await request.json()
    update = Update.de_json(payload, application.bot)
    try:
        await application.process_update(update)
    except Exception as exc:
        logger.exception("Failed to process webhook update")
        raise HTTPException(status_code=500, detail="Failed to process update") from exc

    log_event(
        logger,
        logging.INFO,
        "telegram_webhook_processed",
        update_id=payload.get("update_id"),
    )
    return {"ok": True}
