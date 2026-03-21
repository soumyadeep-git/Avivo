import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Groq Settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# LLM model for text tasks (fast, reliable)
GROQ_TEXT_MODEL = "llama-3.1-8b-instant"
# LLM model for vision/image tasks (Innovation Bonus: Multi-modal)
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"

# Vector Database / Embedding Settings
DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "knowledge_base"
CACHE_COLLECTION_NAME = "semantic_cache"

# Local Embedding Model (Efficiency: Small footprint, fast CPU execution)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG & Chunking Settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
SIMILARITY_THRESHOLD = 0.2  # L2 distance threshold for cache hits
TOP_K_RETRIEVAL = 4         # Number of chunks to retrieve for context
MAX_HISTORY_MESSAGES = 6
MAX_CONTEXT_CHARS = 4000
MIN_RELEVANCE_DISTANCE = 1.2
