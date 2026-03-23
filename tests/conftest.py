import os
import sys
from pathlib import Path


os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DEPLOYMENT_MODE", "polling")
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_VECTOR_SIZE"] = "384"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
