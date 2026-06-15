from collections import defaultdict


def compute_geo_score(analysed_results: dict) -> dict:
    results = analysed_results["results"]
    engines = ["claude", "gpt4o", "perplexity", "gemini"]

    layer_scores = defaultdict(list)
    engine_stats = {e: {"mentions": 0, "accurate": 0, "hallucinations": [], "scores": [], "total": 0} for e in engines}
    all_keywords = []
    all_hallucinations = []
    query_matrix = []

    for item in results:
        layer = item["layer"]
        query = item["query"]
        row = {"query": query, "layer": layer, "engines": {}}

        for engine, data in item["responses"].items():
            analysis = data.get("analysis", {})
            score = analysis.get("score_contribution", 0)
            mentioned = analysis.get("hotel_mentioned", False)
            accurate = analysis.get("description_accurate", False)
            keywords = analysis.get("discovery_keywords_present", [])
            hallucinations = analysis.get("hallucinations", [])

            layer_scores[layer].append(score)

            if engine in engine_stats:
                engine_stats[engine]["total"] += 1
                engine_stats[engine]["scores"].append(score)
                if mentioned:
                    engine_stats[engine]["mentions"] += 1
                if accurate:
                    engine_stats[engine]["accurate"] += 1
                engine_stats[engine]["hallucinations"].extend(hallucinations)

            all_keywords.extend(keywords)
            if hallucinations:
                all_hallucinations.extend([
                    {"engine": engine, "query": query[:80], "claim": h}
                    for h in hallucinations
                ])

            if mentioned and accurate:
                status = "ok"
            elif mentioned and not accurate:
                status = "warning"
            else:
                status = "missing"

            row["engines"][engine] = {"status": status, "score": score}

        query_matrix.append(row)

    def avg(lst):
        return round(sum(lst) / len(lst) * 10, 1) if lst else 0.0

    layer1_scores = layer_scores.get(1, [])
    layer2_scores = layer_scores.get(2, [])
    layer3_scores = layer_scores.get(3, [])
    layer4_scores = layer_scores.get(4, [])

    brand_scores = layer1_scores + layer2_scores
    discovery_scores = layer2_scores + layer3_scores + layer4_scores

    brand_accuracy = avg(brand_scores)
    discovery_score = avg(discovery_scores)
    geo_score_total = round(0.5 * brand_accuracy + 0.5 * discovery_score, 1)

    per_engine = {}
    for engine, stats in engine_stats.items():
        total = stats["total"] or 1
        per_engine[engine] = {
            "mention_rate": round(stats["mentions"] / total * 100, 1),
            "accuracy_rate": round(stats["accurate"] / total * 100, 1),
            "hallucination_count": len(stats["hallucinations"]),
            "avg_score": round(sum(stats["scores"]) / len(stats["scores"]) * 10, 1) if stats["scores"] else 0.0,
            "hallucinations": stats["hallucinations"],
        }

    keyword_freq = defaultdict(int)
    for kw in all_keywords:
        keyword_freq[kw.lower().strip()] += 1

    top_keywords = sorted(keyword_freq.items(), key=lambda x: -x[1])[:20]

    expected_keywords = [
        "lake view", "lago maggiore", "boutique", "wellness", "romantic",
        "spa", "terrace", "garden", "ticino", "swiss", "restaurant",
        "pool", "breakfast", "hiking", "cycling",
    ]
    found_kw_lower = set(k.lower() for k, _ in top_keywords)
    blind_spots = [kw for kw in expected_keywords if kw.lower() not in found_kw_lower]

    strengths = []
    for item in results:
        for engine, data in item["responses"].items():
            analysis = data.get("analysis", {})
            if analysis.get("score_contribution", 0) >= 8 and analysis.get("hotel_mentioned"):
                strengths.append({
                    "engine": engine,
                    "query": item["query"][:100],
                    "keywords": analysis.get("discovery_keywords_present", []),
                })

    gap_points = len(blind_spots) * 2 + sum(
        1 for e in per_engine.values() if e["mention_rate"] < 50
    ) * 5
    projected_score = min(100.0, round(geo_score_total + gap_points, 1))

    return {
        "brand_accuracy": brand_accuracy,
        "discovery_score": discovery_score,
        "geo_score_total": geo_score_total,
        "projected_score": projected_score,
        "per_engine": per_engine,
        "query_matrix": query_matrix,
        "top_keywords": top_keywords,
        "blind_spots": blind_spots,
        "all_hallucinations": all_hallucinations,
        "strengths": strengths,
    }
