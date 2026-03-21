"""
Main Telegram Bot Entry Point.
Handles user interactions, routes commands, and manages image/text inputs.
"""

import logging
from io import BytesIO

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

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋\n\n"
        "I am an advanced GenAI Bot featuring a **Mini-RAG Knowledge Base** and **Vision Capabilities**.\n\n"
        "🛠️ **Available Commands:**\n"
        "🔹 `/ask <query>` - Ask a question about my knowledge base.\n"
        "🔹 `/summarize` - Get a summary of our recent conversation.\n"
        "🔹 `/help` - Show detailed instructions.\n\n"
        "🖼️ **Bonus:** Send me an image, and I will describe it and generate tags for you!"
    )
    await update.message.reply_markdown(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    help_text = (
        "🤖 **Bot Usage Instructions:**\n\n"
        "**1. Text Queries (Mini-RAG)**\n"
        "Use `/ask` followed by your question.\n"
        "`/ask What is the company's remote work policy?`\n\n"
        "**2. Multi-Modal Vision (Innovation Bonus)**\n"
        "Just upload any photo directly to this chat. I will process it and return a caption + keywords.\n\n"
        "**3. Memory & Summarization**\n"
        "Use `/summarize` to summarize our recent chat history.\n\n"
        "⚡ *Efficiency Note:* I use Semantic Caching! If you ask the exact same question, I'll answer instantly without API calls."
    )
    await update.message.reply_markdown(help_text)

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes text queries through the RAG engine."""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("Please provide a query after /ask. Example: /ask What is the policy?")
        return
        
    query_text = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        result = await rag.query(user_id=user_id, query_text=query_text)
        
        answer = result["answer"]
        sources = result["sources"]
        is_cached = result.get("cached", False)
        
        # Build the final message (User Experience)
        final_message = f"{answer}\n\n"
        
        if is_cached:
            final_message += "⚡ *Answer served instantly from Semantic Cache.*\n"
            
        if sources:
            unique_sources = list(set(sources))
            final_message += f"📚 *Sources:* `{', '.join(unique_sources)}`"
        else:
            final_message += "📚 *Sources:* General Knowledge"
            
        await update.message.reply_markdown(final_message)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error while processing your request. Please try again.")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarizes the user's recent conversation."""
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        summary = await rag.summarize(user_id)
        await update.message.reply_text(f"📝 **Conversation Summary:**\n{summary}", parse_mode='Markdown')
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
        
        await update.message.reply_markdown(description)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble processing that image. Ensure it's a valid format.")

async def handle_unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Guides users to proper commands for standard text."""
    await update.message.reply_text("To ask a question, please use the /ask command. For example:\n/ask Hello")

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
