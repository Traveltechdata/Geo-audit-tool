import os
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your API keys."
        )
    return value

ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _require("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # optional: Perplexity engine is skipped if unset
GEMINI_API_KEY = _require("GEMINI_API_KEY")
