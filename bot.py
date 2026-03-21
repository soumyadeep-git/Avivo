"""
Main Telegram Bot Entry Point.
Handles Telegram user interactions and exposes an application factory
that can be used for polling or webhook-based deployments.
"""

import logging
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import settings
from logging_utils import configure_logging, log_event
from rag_engine import RAGEngine


logger = logging.getLogger(__name__)

EXAMPLE_QUERIES = [
    "What are path parameters in FastAPI?",
    "How do I declare query parameters?",
    "When should I use async def in FastAPI?",
]


def format_examples() -> str:
    """Return a short evaluator-friendly list of demo prompts."""
    return "\n".join(f"- /ask {query}" for query in EXAMPLE_QUERIES)


def build_answer_message(result: Dict[str, Any]) -> str:
    """Format query responses in a polished evaluator-friendly layout."""
    answer = (result.get("answer") or "").strip()
    sources: List[str] = result.get("sources", [])
    source_snippets: List[str] = result.get("source_snippets", [])
    is_cached = result.get("cached", False)
    is_grounded = result.get("grounded", False)

    response_sections = ["Answer", answer or "No answer generated."]

    evidence_label = (
        "Grounded in retrieved documentation"
        if is_grounded
        else "Partially grounded; retrieved context was limited"
    )
    if is_cached:
        evidence_label += " | Served from semantic cache"
    response_sections.extend(["", f"Evidence: {evidence_label}"])

    if sources:
        response_sections.append("")
        response_sections.append("Sources:")
        response_sections.extend(f"- {source}" for source in sources[:3])

    if source_snippets:
        response_sections.append("")
        response_sections.append("Retrieved snippets:")
        response_sections.extend(f"- {snippet}" for snippet in source_snippets[:2])

    return "\n".join(response_sections).strip()


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    """Create a single RAG engine instance for the process lifetime."""
    return RAGEngine()


def get_rag_from_context(context: ContextTypes.DEFAULT_TYPE) -> RAGEngine:
    """Fetch the shared RAG engine from bot context."""
    return context.application.bot_data["rag_engine"]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}!\n\n"
        "I am a retrieval-augmented Telegram assistant with document QA and image understanding.\n\n"
        "Commands:\n"
        "- /ask <query> : ask a question against the knowledge base\n"
        "- /summarize : summarize recent conversation history\n"
        "- /help : show instructions and demo prompts\n\n"
        "Try one of these demo questions:\n"
        f"{format_examples()}\n\n"
        "You can also upload an image and I will describe it."
    )
    await update.message.reply_text(welcome_msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    help_text = (
        "Usage guide\n\n"
        "1. Text queries\n"
        "Use /ask followed by a question.\n"
        "Example prompts:\n"
        f"{format_examples()}\n\n"
        "2. Image understanding\n"
        "Upload an image directly in chat to receive a caption and tags.\n\n"
        "3. Conversation memory\n"
        "Use /summarize to get a concise summary of recent interactions.\n\n"
        "Response format\n"
        "- Direct answer\n"
        "- Evidence quality note\n"
        "- Source citations\n"
        "- Retrieved snippet preview when available"
    )
    await update.message.reply_text(help_text)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes text queries through the RAG engine."""
    user_id = update.effective_user.id
    rag_engine = get_rag_from_context(context)

    if not context.args:
        await update.message.reply_text(
            "Please provide a query after /ask.\n\nExample prompts:\n"
            f"{format_examples()}"
        )
        return

    query_text = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        result = await rag_engine.query(user_id=user_id, query_text=query_text)
        log_event(
            logger,
            logging.INFO,
            "telegram_query_processed",
            user_id=user_id,
            cached=result.get("cached", False),
            grounded=result.get("grounded", False),
            sources=len(result.get("sources", [])),
        )
        await update.message.reply_text(build_answer_message(result))
    except Exception:
        logger.exception("Failed to process query")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your request. Please try again."
        )


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarizes the user's recent conversation."""
    user_id = update.effective_user.id
    rag_engine = get_rag_from_context(context)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        summary = await rag_engine.summarize(user_id)
        await update.message.reply_text(f"Conversation summary\n\n{summary}")
    except Exception:
        logger.exception("Failed to summarize conversation")
        await update.message.reply_text("Could not summarize at this time.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles image uploads by passing them to the vision model."""
    user_id = update.effective_user.id
    rag_engine = get_rag_from_context(context)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        image_bytes_io = BytesIO()
        await file.download_to_memory(out=image_bytes_io)
        image_bytes = image_bytes_io.getvalue()

        description = await rag_engine.describe_image(user_id, image_bytes)
        await update.message.reply_text(description)
    except Exception:
        logger.exception("Failed to process image")
        await update.message.reply_text(
            "Sorry, I had trouble processing that image. Ensure it is a valid format."
        )


async def handle_unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Guides users to proper commands for standard text."""
    await update.message.reply_text(
        "Use /ask followed by a question.\n\nExample prompts:\n"
        f"{format_examples()}"
    )


def register_handlers(application: Application) -> None:
    """Register command and message handlers."""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown_message))


def build_application(use_webhook_transport: bool = False) -> Application:
    """Create a Telegram application that can run via polling or webhook."""
    builder = Application.builder().token(settings.telegram_bot_token)
    if use_webhook_transport:
        builder = builder.updater(None)
    application = builder.build()
    application.bot_data["rag_engine"] = get_rag_engine()
    register_handlers(application)
    return application


def main() -> None:
    """Local polling entry point."""
    configure_logging(settings.log_level)
    application = build_application(use_webhook_transport=False)
    log_event(
        logger,
        logging.INFO,
        "telegram_polling_started",
        deployment_mode=settings.deployment_mode,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
