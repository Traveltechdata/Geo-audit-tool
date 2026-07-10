import asyncio
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-1.5-flash")


def _extract_text(response) -> str:
    """
    Estrae testo dalla risposta Gemini combinando:
    - response.text (testo principale generato)
    - grounding_metadata.grounding_chunks (fonti web se la web search è attiva)
    """
    parts = []

    # Testo principale
    try:
        if response.text:
            parts.append(response.text)
    except Exception:
        # response.text può sollevare eccezione se il contenuto è bloccato
        pass

    # Testo dai candidates in caso response.text sia vuoto
    if not parts:
        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
        except Exception:
            pass

    # Fonti da grounding_metadata (web search tool result)
    try:
        for candidate in response.candidates:
            gm = getattr(candidate, "grounding_metadata", None)
            if gm is None:
                continue
            chunks = getattr(gm, "grounding_chunks", []) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "")
                    title = getattr(web, "title", "")
                    if title:
                        parts.append(f"[fonte: {title} — {uri}]")
    except Exception:
        pass

    result = "\n\n".join(p for p in parts if p and p.strip())
    return result if result else str(response)


def _sync_query(prompt: str) -> str:
    response = _model.generate_content(prompt)
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
