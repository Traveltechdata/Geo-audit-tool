import asyncio
import openai
from config import OPENAI_API_KEY

_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


async def query_openai(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = await _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)
        except openai.APIError as e:
            if attempt == 2:
                return f"[ERROR openai]: {e}"
            await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR openai]: Max retries exceeded after rate limiting"
