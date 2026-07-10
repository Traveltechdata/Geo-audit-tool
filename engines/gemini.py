import asyncio
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)
_MODEL = "gemini-2.5-flash"
_CONFIG = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])


def _extract_text(response) -> str:
    """
    Estrae testo dalla risposta Gemini:
    1. response.text — testo sintetizzato principale (può sollevare ValueError
       se il contenuto è bloccato dal safety filter, catturiamo l'eccezione).
    2. candidates[].content.parts — fallback se response.text è vuoto/bloccato.
    3. grounding_metadata.grounding_chunks — fonti web usate dalla Google Search,
       aggiunte come righe [Fonte web] al termine del testo.
    """
    parts = []

    # 1. Testo principale
    try:
        main_text = response.text
        if main_text and main_text.strip():
            parts.append(main_text)
    except (ValueError, AttributeError):
        pass

    # 2. Fallback via candidates se il testo principale è assente
    if not parts:
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                t = getattr(part, "text", None)
                if t and t.strip():
                    parts.append(t)

    # 3. Fonti dalla Google Search grounding
    for candidate in getattr(response, "candidates", None) or []:
        grounding = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(grounding, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            uri = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or ""
            if uri:
                parts.append(f"[Fonte web] {(title + ' — ') if title else ''}{uri}")

    result = "\n".join(p for p in parts if p)
    if not result.strip():
        result = f"[EXTRACT_EMPTY] candidates={len(getattr(response, 'candidates', None) or [])}"
    return result


def _sync_query(prompt: str) -> str:
    response = _client.models.generate_content(model=_MODEL, contents=prompt, config=_CONFIG)
    return _extract_text(response)


async def query_gemini(prompt: str) -> str:
    for attempt in range(3):
        try:
            text = await asyncio.to_thread(_sync_query, prompt)
            preview = text[:200].replace("\n", " ")
            print(f"    [gemini debug] {preview}…")
            return text
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
