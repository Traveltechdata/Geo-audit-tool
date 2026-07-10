import asyncio
import openai
from config import OPENAI_API_KEY

_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


def _extract_text(response) -> str:
    """
    Estrae testo dalla risposta OpenAI.
    Percorso 1 (Responses API): response.output_text (campo diretto).
    Percorso 2 (Responses API): itera response.output cercando blocchi message
                                 con content di tipo output_text o text.
    Percorso 3 (Chat Completions fallback): response.choices[0].message.content.
    """
    # Percorso 1 — campo diretto della Responses API
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    # Percorso 2 — lista output della Responses API
    output = getattr(response, "output", None)
    if output:
        parts = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", None) or []:
                # tipo può essere 'output_text' (Responses API) o 'text' (Chat)
                ctype = getattr(content, "type", None)
                if ctype in ("output_text", "text"):
                    text = getattr(content, "text", "") or ""
                    if text.strip():
                        parts.append(text)
        if parts:
            return "\n".join(parts)

    # Percorso 3 — Chat Completions API classica
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg else None
        if content:
            return content

    return f"[EXTRACT_EMPTY] response_type={type(response).__name__}"


async def query_openai(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = await _client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            text = _extract_text(response)
            preview = text[:200].replace("\n", " ")
            print(f"    [openai debug] {preview}…")
            return text
        except openai.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except openai.APIError as e:
            if attempt == 2:
                return f"[ERROR openai]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR openai]: Max retries exceeded after rate limiting"
