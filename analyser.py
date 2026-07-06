import json
import asyncio
from engines.claude import query_claude

_ANALYSIS_PROMPT = """You are an expert GEO (Generative Engine Optimization) auditor evaluating how AI models describe a hotel.
You are a CALIBRATED, FAIR judge — not a strict one. Give credit for partial, generic, or incomplete
mentions as long as nothing stated is false. Only penalize what is actually wrong, never what is merely absent.

HOTEL NAME: {hotel_name}
HOTEL LOCATION: {location}
VERIFIED HOTEL FACTS:
{hotel_data}

VERIFIED HOTEL FACTS may include real guest review excerpts (from Booking.com, via Apify). Treat these
reviews as equally authoritative ground truth alongside the rest of the facts: use them to catch
hallucinations (a response contradicting what guests actually report) and blind spots (a widely-confirmed
detail from the reviews that AI responses consistently omit).

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

FIELD DEFINITIONS (apply these exactly — do not use stricter criteria):

hotel_mentioned = true if the hotel appears by full name, partial name, or any clear unambiguous
reference to it (e.g. a paraphrase, a distinctive feature that identifies it, or a description that
clearly points to this specific hotel and no other). For Layer 2/3/4 (indirect discovery) queries,
hotel_mentioned=true also when the hotel is suggested/recommended without being named directly in the
question. Only set false when the hotel is genuinely absent or a different hotel is the one described.

description_accurate = true UNLESS the response contains at least one claim that actively CONTRADICTS
the VERIFIED HOTEL FACTS. Do NOT require completeness: omitting details, being generic, or describing
only part of the hotel is still accurate. Set false only when something stated is demonstrably wrong.

hallucinations = ONLY specific, verifiable claims that DIRECTLY CONTRADICT the VERIFIED HOTEL FACTS
above (e.g. wrong star category, wrong town/location, breakfast described as à la carte when facts say
buffet included, invented awards or amenities that facts do not support). NEVER count as a hallucination:
missing information, generic/marketing language, vague descriptions, or plausible details that the facts
simply don't mention one way or the other. If VERIFIED HOTEL FACTS says no facts were provided, return
an empty hallucinations list for that response — there is nothing to verify against.

SCORING RULES for score_contribution (0-10):
- 0: Hotel not mentioned in the response at all
- 2-3: Hotel mentioned, but with serious errors (major hallucinations that misrepresent the hotel)
- 4-5: Hotel mentioned, but with some errors (one or more hallucinations, not severe)
- 6-7: Hotel mentioned correctly (no hallucinations), even if the description is generic, partial, or thin
- 8-10: Hotel mentioned as the top choice, with accurate and verified details (rich discovery keywords, no hallucinations)

DISCOVERY KEYWORDS: Extract meaningful keywords that would help a traveler discover this hotel (e.g., "lakeside", "boutique", "wellness", "Ticino", "romantic getaway", etc.).

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
