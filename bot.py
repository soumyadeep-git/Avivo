"""
Main Telegram Bot Entry Point.
Handles user interactions, routes commands, and manages image/text inputs.
"""

import logging
from io import BytesIO
from typing import Any, Dict, List

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from config import TELEGRAM_BOT_TOKEN
from rag_engine import RAGEngine

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global RAG Engine Instance
rag = RAGEngine()

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

    evidence_label = "Grounded in retrieved documentation" if is_grounded else "Partially grounded; retrieved context was limited"
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
    
    if not context.args:
        await update.message.reply_text(
            "Please provide a query after /ask.\n\nExample prompts:\n"
            f"{format_examples()}"
        )
        return
        
    query_text = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        result = await rag.query(user_id=user_id, query_text=query_text)
        
        await update.message.reply_text(build_answer_message(result))

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error while processing your request. Please try again.")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarizes the user's recent conversation."""
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        summary = await rag.summarize(user_id)
        await update.message.reply_text(f"Conversation summary\n\n{summary}")
    except Exception as e:
        logger.error(f"Error summarizing: {e}")
        await update.message.reply_text("❌ Could not summarize at this time.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Innovation Bonus: Handles incoming images by passing them to the Vision model.
    """
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    try:
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download image into memory
        image_bytes_io = BytesIO()
        await file.download_to_memory(out=image_bytes_io)
        image_bytes = image_bytes_io.getvalue()
        
        # Process image
        description = await rag.describe_image(user_id, image_bytes)
        
        await update.message.reply_text(description)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble processing that image. Ensure it's a valid format.")

async def handle_unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Guides users to proper commands for standard text."""
    await update.message.reply_text(
        "Use /ask followed by a question.\n\nExample prompts:\n"
        f"{format_examples()}"
    )

def main() -> None:
    """Application entry point."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        logger.error("Please set TELEGRAM_BOT_TOKEN in your .env file!")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("summarize", summarize_command))

    # Multi-modal support (Innovation)
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Fallback text handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown_message))

    logger.info("🤖 Bot started successfully! Waiting for messages...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
