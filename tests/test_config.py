import pytest

from config import Settings


def test_settings_builds_webhook_url() -> None:
    test_settings = Settings(
        TELEGRAM_BOT_TOKEN="token",
        GROQ_API_KEY="groq",
        DEPLOYMENT_MODE="webhook",
        TELEGRAM_WEBHOOK_BASE_URL="https://example.com",
        TELEGRAM_WEBHOOK_SECRET="secret",
    )

    assert test_settings.telegram_webhook_url == "https://example.com/telegram/webhook"


def test_production_requires_qdrant_url() -> None:
    with pytest.raises(ValueError):
        Settings(
            TELEGRAM_BOT_TOKEN="token",
            GROQ_API_KEY="groq",
            APP_ENV="production",
            QDRANT_URL=None,
        )
