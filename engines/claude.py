import asyncio
import anthropic
from config import ANTHROPIC_API_KEY

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def _extract_text(content_blocks) -> str:
    """
    Concatena tutti i blocchi type='text' (risposta sintetizzata da Claude)
    e aggiunge titolo+URL dai blocchi type='web_search_tool_result'.
    Il testo sintetizzato è sempre in blocchi type='text'; i blocchi
    web_search_tool_result contengono solo metadati delle fonti (encrypted_content
    non è accessibile), quindi ne estraiamo solo titolo e URL come contesto.
    """
    parts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            text = getattr(block, "text", "") or ""
            if text.strip():
                parts.append(text)

        elif block_type == "web_search_tool_result":
            for item in getattr(block, "content", None) or []:
                # item può essere oggetto SDK o dict
                if isinstance(item, dict):
                    title = item.get("title", "")
                    url = item.get("url", "")
                else:
                    title = getattr(item, "title", "") or ""
                    url = getattr(item, "url", "") or ""
                if title or url:
                    parts.append(f"[Fonte web] {(title + ' — ') if title else ''}{url}")

    result = "\n".join(p for p in parts if p)

    # Fallback: se non abbiamo estratto nulla, restituiamo repr per debug
    if not result.strip():
        result = f"[EXTRACT_EMPTY] blocks={[getattr(b, 'type', '?') for b in content_blocks]}"

    return result


async def query_claude(prompt: str, model: str = "claude-sonnet-4-6", use_web_search: bool = True) -> str:
    for attempt in range(3):
        try:
            kwargs = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if use_web_search:
                kwargs["tools"] = [_WEB_SEARCH_TOOL]
            message = await _client.messages.create(**kwargs)
            text = _extract_text(message.content)
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
