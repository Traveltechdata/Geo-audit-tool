import asyncio
import openai
from config import OPENAI_API_KEY

_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


def _extract_output_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    parts = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


async def query_openai(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = await _client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            return _extract_output_text(response)
        except openai.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except openai.APIError as e:
            if attempt == 2:
                return f"[ERROR openai]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR openai]: Max retries exceeded after rate limiting"
