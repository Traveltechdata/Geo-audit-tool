import asyncio
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-1.5-flash")


def _sync_query(prompt: str) -> str:
    response = _model.generate_content(prompt)
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
