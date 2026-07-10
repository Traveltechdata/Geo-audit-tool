import asyncio
import httpx
from config import PERPLEXITY_API_KEY

_API_URL = "https://api.perplexity.ai/chat/completions"


async def query_perplexity(prompt: str) -> str | None:
    if not PERPLEXITY_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(_API_URL, headers=headers, json=payload)
                if response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                preview = text[:200].replace("\n", " ")
                print(f"    [perplexity debug] {preview}…")
                return text
            except httpx.HTTPStatusError as e:
                if attempt == 2:
                    return f"[ERROR perplexity]: HTTP {e.response.status_code}"
                await asyncio.sleep(2 ** (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    return f"[ERROR perplexity]: {e}"
                await asyncio.sleep(2 ** (attempt + 1))
    return "[ERROR perplexity]: Max retries exceeded"
