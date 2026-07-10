import asyncio
import anthropic
from config import ANTHROPIC_API_KEY

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


def _extract_text(message) -> str:
    """
    Concatena tutto il testo leggibile dalla risposta Claude,
    inclusi blocchi type='text' e contenuto testuale estratto
    dai blocchi type='web_search_tool_result'.
    """
    parts = []
    for block in message.content:
        btype = getattr(block, "type", None)

        if btype == "text":
            parts.append(block.text)

        elif btype == "tool_use" and getattr(block, "name", "") == "web_search":
            pass  # la richiesta di ricerca non contiene testo utile

        elif btype == "tool_result" or btype == "web_search_tool_result":
            # Il risultato della ricerca web può essere una lista di blocchi o una stringa
            content = getattr(block, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if getattr(item, "type", None) == "text":
                        parts.append(item.text)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))

    return "\n\n".join(p for p in parts if p.strip())


async def query_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    for attempt in range(3):
        try:
            message = await _client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _extract_text(message)
            if not text:
                # Fallback diretto in caso di struttura inattesa
                text = str(message.content)
            preview = text[:200].replace("\n", " ")
            print(f"    [claude debug] {preview}…")
            return text
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            if attempt == 2:
                return f"[ERROR claude]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR claude]: Max retries exceeded after rate limiting"
