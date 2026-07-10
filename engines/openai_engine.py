import asyncio
import openai
from config import OPENAI_API_KEY

_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


def _extract_text(response) -> str:
    """
    Estrae testo dalla risposta OpenAI coprendo sia l'API
    chat.completions (response.choices) sia la Responses API
    (response.output_text / response.output).
    """
    # Responses API — campo diretto
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Responses API — lista output con blocchi message/text
    if hasattr(response, "output") and response.output:
        parts = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", None) == "output_text":
                        parts.append(block.text)
                    elif getattr(block, "type", None) == "text":
                        parts.append(block.text)
            elif item_type == "text":
                parts.append(getattr(item, "text", ""))
        if parts:
            return "\n\n".join(p for p in parts if p.strip())

    # Chat Completions API — percorso classico
    if hasattr(response, "choices") and response.choices:
        return response.choices[0].message.content or ""

    return str(response)


async def query_openai(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = await _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
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
