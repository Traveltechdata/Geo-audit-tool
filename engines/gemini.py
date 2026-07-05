import asyncio
from google import genai
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)
_MODEL = "gemini-2.5-flash"


def _sync_query(prompt: str) -> str:
    response = _client.models.generate_content(model=_MODEL, contents=prompt)
    return response.text


async def query_gemini(prompt: str) -> str:
    for attempt in range(3):
        try:
            result = await asyncio.to_thread(_sync_query, prompt)
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
            else:
                if attempt == 2:
                    return f"[ERROR gemini]: {e}"
                await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR gemini]: Max retries exceeded after rate limiting"
