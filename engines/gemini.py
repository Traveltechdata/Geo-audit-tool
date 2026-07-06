import asyncio
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)
_MODEL = "gemini-2.5-flash"
_CONFIG = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])


def _extract_text(response) -> str:
    parts = [response.text] if response.text else []
    for candidate in getattr(response, "candidates", None) or []:
        grounding = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(grounding, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web else None
            if uri:
                title = getattr(web, "title", "") or ""
                parts.append(f"[Fonte web] {title + ' — ' if title else ''}{uri}")
    return "\n".join(p for p in parts if p)


def _sync_query(prompt: str) -> str:
    response = _client.models.generate_content(model=_MODEL, contents=prompt, config=_CONFIG)
    return _extract_text(response)


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
