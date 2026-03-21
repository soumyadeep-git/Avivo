import os
import sys
from pathlib import Path


os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DEPLOYMENT_MODE", "polling")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
