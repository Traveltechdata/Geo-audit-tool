import json
import asyncio
from engines.claude import query_claude

_ANALYSIS_PROMPT = """You are an expert GEO (Generative Engine Optimization) auditor evaluating how AI models describe a hotel.

HOTEL NAME: {hotel_name}
HOTEL LOCATION: {location}
VERIFIED HOTEL FACTS:
{hotel_data}

Below are AI responses to the query: "{query}"

Evaluate EACH response and return a JSON array (one object per response) with this exact structure:
[
  {{
    "engine": "claude",
    "hotel_mentioned": true,
    "description_accurate": true,
    "discovery_keywords_present": ["lake view", "wellness", "romantic"],
    "hallucinations": [],
    "score_contribution": 8
  }},
  ...
]

SCORING RULES for score_contribution (0-10):
- 0: Hotel not mentioned and response is off-topic
- 1-3: Hotel not mentioned but response is relevant to the area
- 4-5: Hotel mentioned but with major inaccuracies or missing key facts
- 6-7: Hotel mentioned with mostly accurate information
- 8-9: Hotel mentioned accurately with good descriptive keywords
- 10: Perfect mention with accurate details, strong discovery keywords, and no hallucinations

HALLUCINATION DETECTION: Flag any specific claim that contradicts the verified facts or is clearly invented (wrong star rating, wrong amenities, wrong location, invented awards, etc.).

DISCOVERY KEYWORDS: Extract meaningful keywords that would help a traveler discover this hotel (e.g., "lakeside", "boutique", "wellness", "Ticino", "romantic getaway", etc.).

For Layer 2/3/4 queries (indirect discovery), hotel_mentioned=true means the hotel was suggested even without being asked about it directly.

Responses to evaluate:
{responses}

Return ONLY valid JSON array, no markdown, no explanation."""


async def analyse_results(raw_results: dict, hotel_data: str = "") -> dict:
    hotel_name = raw_results["hotel"]
    location = raw_results["location"]
    hotel_facts = hotel_data if hotel_data else "No verified facts provided — evaluate based on general plausibility."

    enriched_results = []
    total = len(raw_results["results"])

    print(f"\n[Analyser] Starting analysis of {total} queries with Claude as judge...")

    for idx, item in enumerate(raw_results["results"], 1):
        query = item["query"]
        layer = item["layer"]
        responses = item["responses"]

        responses_text = "\n\n".join(
            f"ENGINE: {engine}\nRESPONSE:\n{text}"
            for engine, text in responses.items()
        )

        prompt = _ANALYSIS_PROMPT.format(
            hotel_name=hotel_name,
            location=location,
            hotel_data=hotel_facts,
            query=query,
            responses=responses_text,
        )

        print(f"  [Analyser] Query {idx}/{total}: analysing {len(responses)} responses...", end=" ", flush=True)

        raw_analysis = await query_claude(prompt, model="claude-sonnet-4-6")

        analysis_list = _parse_analysis(raw_analysis, list(responses.keys()))

        analysis_map = {a["engine"]: a for a in analysis_list}

        enriched_responses = {}
        for engine, text in responses.items():
            enriched_responses[engine] = {
                "response": text,
                "analysis": analysis_map.get(engine, _default_analysis(engine)),
            }

        enriched_results.append({
            "query": query,
            "layer": layer,
            "responses": enriched_responses,
        })
        print("✓")

    return {
        "hotel": hotel_name,
        "location": location,
        "timestamp": raw_results["timestamp"],
        "results": enriched_results,
    }


def _parse_analysis(raw: str, engines: list) -> list:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return [_default_analysis(e) for e in engines]
    except (json.JSONDecodeError, Exception):
        return [_default_analysis(e) for e in engines]


def _default_analysis(engine: str) -> dict:
    return {
        "engine": engine,
        "hotel_mentioned": False,
        "description_accurate": False,
        "discovery_keywords_present": [],
        "hallucinations": [],
        "score_contribution": 0,
    }
