import asyncio
import anthropic
from config import ANTHROPIC_API_KEY

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def query_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    for attempt in range(3):
        try:
            message = await _client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            if attempt == 2:
                return f"[ERROR claude]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR claude]: Max retries exceeded after rate limiting"
