from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_dotenv():
    """Load local project secrets from .env without overriding real environment variables."""
    return load_dotenv(PROJECT_ROOT / ".env", override=False)
