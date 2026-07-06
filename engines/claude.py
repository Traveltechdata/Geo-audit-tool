import asyncio
import anthropic
from config import ANTHROPIC_API_KEY

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def _extract_text(content_blocks) -> str:
    parts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        elif block_type == "web_search_tool_result":
            for item in getattr(block, "content", None) or []:
                title = getattr(item, "title", None)
                url = getattr(item, "url", None)
                if title and url:
                    parts.append(f"[Fonte web] {title} — {url}")
                elif title or url:
                    parts.append(f"[Fonte web] {title or url}")
    return "\n".join(p for p in parts if p)


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
            return _extract_text(message.content)
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            if attempt == 2:
                return f"[ERROR claude]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR claude]: Max retries exceeded after rate limiting"
