import json
import asyncio
from engines.claude import query_claude

_ANALYSIS_PROMPT = """You are an expert GEO (Generative Engine Optimization) auditor evaluating how AI models describe a hotel.

HOTEL NAME: {hotel_name}
HOTEL LOCATION: {location}
VERIFIED HOTEL FACTS:
{hotel_data}

VERIFIED HOTEL FACTS may include real guest review excerpts (from Booking.com / Google Maps, via Apify).
Treat these reviews as equally authoritative ground truth: use them to catch hallucinations (a response
contradicting what guests actually report) and blind spots (a widely-confirmed detail the AI omits).

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

hotel_mentioned = true ONLY when the hotel is explicitly named (full name, partial name, or an
unambiguous reference that could not apply to any other hotel). For Layer 2/3/4 discovery queries,
true also when the hotel is recommended/suggested without being named in the question itself.
Set false whenever the hotel is completely absent from the response — do NOT set true just because
the response mentions the region or general hotel categories.

description_accurate = true UNLESS at least one claim actively CONTRADICTS the VERIFIED HOTEL FACTS
(wrong star rating, wrong town, wrong amenity, etc.). Omitting details or being generic is still accurate.
If hotel_mentioned is false, set description_accurate to false as well.

hallucinations = list of specific claims that DIRECTLY CONTRADICT the VERIFIED HOTEL FACTS.
Do NOT flag: missing info, generic language, vague descriptions, or plausible details not mentioned
in the facts. If no verified facts were provided, always return an empty list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING SCALE — score_contribution (integer 0–10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

!! CRITICAL RULE: if hotel_mentioned is false, score_contribution MUST be 0. No exceptions. !!
Do not assign 1, 2, or any positive value when the hotel is absent from the response.

When hotel_mentioned is true, apply this scale:

  9–10  Hotel cited first or as the top recommendation; information accurate and verified against
        facts; no hallucinations; rich discovery keywords present.

  7–8   Hotel cited clearly; information substantially correct; at most minor omissions or
        imprecisions that do not contradict the facts; no hallucinations.

  5–6   Hotel cited; some inaccuracies present that are not severe (e.g. one minor error or a
        claim the facts do not support but do not explicitly contradict).

  3–4   Hotel cited but with serious errors: one or more hallucinations that materially
        misrepresent the hotel (wrong location, wrong category, invented major amenity).

  1–2   Hotel cited but the response is predominantly false or contradicts the facts on
        multiple key points.

  0     Hotel NOT cited — mandatory regardless of layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCOVERY KEYWORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract only keywords that actually appear (or are clearly implied) in the response and would help
a traveler discover this hotel (e.g. "lakeside", "boutique", "wellness", "Ticino", "romantic getaway").
Return an empty list when hotel_mentioned is false.

Responses to evaluate:
{responses}

Return ONLY a valid JSON array. No markdown fences, no explanation, no extra text."""


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

        raw_analysis = await query_claude(prompt, model="claude-sonnet-4-6", use_web_search=False)

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
            return [_enforce_rules(item) for item in data]
        return [_default_analysis(e) for e in engines]
    except (json.JSONDecodeError, Exception):
        return [_default_analysis(e) for e in engines]


def _enforce_rules(item: dict) -> dict:
    """
    Hard-enforce scoring rules that the LLM judge may occasionally violate:
    - score_contribution must be 0 when hotel_mentioned is false
    - discovery_keywords_present must be empty when hotel_mentioned is false
    - score_contribution must be clamped to [0, 10]
    """
    if not item.get("hotel_mentioned", False):
        item["score_contribution"] = 0
        item["discovery_keywords_present"] = []
        item["description_accurate"] = False
    else:
        score = item.get("score_contribution", 0)
        item["score_contribution"] = max(0, min(10, int(score)))
    return item


def _default_analysis(engine: str) -> dict:
    return {
        "engine": engine,
        "hotel_mentioned": False,
        "description_accurate": False,
        "discovery_keywords_present": [],
        "hallucinations": [],
        "score_contribution": 0,
    }
