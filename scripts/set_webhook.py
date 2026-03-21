"""
Register the Telegram webhook for deployed environments.
"""

import asyncio
import sys
from pathlib import Path

from telegram import Bot, Update

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


async def main() -> None:
    bot = Bot(token=settings.telegram_bot_token)
    await bot.set_webhook(
        url=settings.telegram_webhook_url,
        secret_token=settings.telegram_webhook_secret,
        allowed_updates=Update.ALL_TYPES,
    )
    print(f"Webhook registered: {settings.telegram_webhook_url}")


if __name__ == "__main__":
    asyncio.run(main())
