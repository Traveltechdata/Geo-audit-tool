import argparse
import asyncio
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

from analyser import analyse_results
from scorer import compute_geo_score
from report import generate_report

# recensioni.csv fields: rating, reviewTitle, likedText, dislikedText,
# reviewDate, numberOfNights, travelerType, userName
_CSV_MAX_REVIEWS = 100


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_reviews_csv(csv_path: str) -> str:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return ""

    blocks = []
    for i, row in enumerate(rows[:_CSV_MAX_REVIEWS], 1):
        rating = (row.get("rating") or "").strip()
        title = (row.get("reviewTitle") or "").strip()
        liked = (row.get("likedText") or "").strip()
        disliked = (row.get("dislikedText") or "").strip()

        header = f"### Recensione {i}" + (f" (Rating: {rating})" if rating else "")
        block = [header]
        if title:
            block.append(f"Titolo: {title}")
        if liked:
            block.append(f"Positivo: {liked}")
        if disliked:
            block.append(f"Negativo: {disliked}")
        blocks.append("\n".join(block))

    body = "\n\n".join(blocks)
    if len(rows) > _CSV_MAX_REVIEWS:
        body += f"\n\n[...{len(rows) - _CSV_MAX_REVIEWS} recensioni aggiuntive omesse per brevità...]"
    return body


_JSON_MAX_REVIEWS = 100


def _load_reviews_booking_json(json_path: str) -> str:
    """Apify dataset_booking-reviews-scraper_*.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        rows = [rows]

    blocks = []
    for i, row in enumerate(rows[:_JSON_MAX_REVIEWS], 1):
        rating = str(row.get("rating") or row.get("score") or "").strip()
        title = (row.get("title") or row.get("reviewTitle") or "").strip()
        text = (
            row.get("text") or
            row.get("reviewText") or
            row.get("body") or
            row.get("likedText") or ""
        ).strip()
        disliked = (row.get("dislikedText") or "").strip()

        header = f"### Recensione Booking {i}" + (f" (Rating: {rating})" if rating else "")
        block = [header]
        if title:
            block.append(f"Titolo: {title}")
        if text:
            block.append(f"Testo: {text}")
        if disliked:
            block.append(f"Negativo: {disliked}")
        if len(block) > 1:
            blocks.append("\n".join(block))

    if not blocks:
        return ""
    body = "\n\n".join(blocks)
    if len(rows) > _JSON_MAX_REVIEWS:
        body += f"\n\n[...{len(rows) - _JSON_MAX_REVIEWS} recensioni aggiuntive omesse per brevità...]"
    return body


def _load_reviews_google_json(json_path: str) -> str:
    """Apify dataset_Google-Maps-Reviews-Scraper_*.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        rows = [rows]

    blocks = []
    for i, row in enumerate(rows[:_JSON_MAX_REVIEWS], 1):
        rating = str(row.get("stars") or row.get("rating") or row.get("score") or "").strip()
        text = (
            row.get("text") or
            row.get("reviewText") or
            row.get("snippet") or
            row.get("body") or ""
        ).strip()
        title = (row.get("title") or row.get("reviewTitle") or "").strip()

        header = f"### Recensione Google {i}" + (f" (Rating: {rating}★)" if rating else "")
        block = [header]
        if title:
            block.append(f"Titolo: {title}")
        if text:
            block.append(f"Testo: {text}")
        if len(block) > 1:
            blocks.append("\n".join(block))

    if not blocks:
        return ""
    body = "\n\n".join(blocks)
    if len(rows) > _JSON_MAX_REVIEWS:
        body += f"\n\n[...{len(rows) - _JSON_MAX_REVIEWS} recensioni aggiuntive omesse per brevità...]"
    return body


def _glob_first(directory: str, pattern: str) -> str | None:
    """Ritorna il primo file che matcha il pattern nella directory, o None."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    return matches[0] if matches else None


def _load_queries(queries_path: str, hotel_name: str, location: str) -> list:
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_queries = data if isinstance(data, list) else data["queries"]
    queries = []
    for q in raw_queries:
        raw_text = q.get("query", q.get("text"))
        text = raw_text.replace("{hotel_name}", hotel_name).replace("{location}", location)
        queries.append({"id": q["id"], "layer": q["layer"], "text": text})
    return queries


async def _run_query(query_text: str, idx: int, total: int) -> dict:
    from engines.claude import query_claude
    from engines.openai_engine import query_openai
    from engines.perplexity import query_perplexity
    from engines.gemini import query_gemini

    short = query_text[:60] + ("…" if len(query_text) > 60 else "")
    print(f"  [{idx:02d}/{total}] {short}", flush=True)

    results = await asyncio.gather(
        query_claude(query_text),
        query_openai(query_text),
        query_perplexity(query_text),
        query_gemini(query_text),
        return_exceptions=True,
    )

    def _safe(r):
        return str(r) if isinstance(r, Exception) else r

    claude_r, openai_r, perplexity_r, gemini_r = [_safe(r) for r in results]

    responses = {"claude": claude_r, "gpt4o": openai_r}
    status = ["✓ claude", "✓ gpt4o"]
    if perplexity_r is not None:
        responses["perplexity"] = perplexity_r
        status.append("✓ perplexity")
    responses["gemini"] = gemini_r
    status.append("✓ gemini")
    print(f"         {' | '.join(status)}")
    return responses


async def main():
    parser = argparse.ArgumentParser(description="GEO Audit Tool — Analisi AI Visibility per Hotel")
    parser.add_argument("--hotel", required=True, help='Nome hotel (es. "Garden Hotel Primavera")')
    parser.add_argument("--location", required=True, help='Location (es. "Brissago, Lago Maggiore, Ticino")')
    parser.add_argument("--hotel-dir", required=True, dest="hotel_dir",
                         help="Cartella hotel con dati_hotel.txt, queries.json, recensioni.txt e recensioni.csv (es. hotels/garden_hotel_primavera/)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Usa solo le prime 3 query (debug rapido, riduce costi API)")
    args = parser.parse_args()

    hotel_name = args.hotel
    location = args.location
    hotel_dir = args.hotel_dir
    hotel_data = ""

    data_path = os.path.join(hotel_dir, "dati_hotel.txt")
    queries_path = os.path.join(hotel_dir, "queries.json")
    recensioni_path = os.path.join(hotel_dir, "recensioni.txt")
    recensioni_csv_path = os.path.join(hotel_dir, "recensioni.csv")

    if not os.path.isfile(queries_path):
        sys.exit(f"[✗] File queries.json non trovato in {hotel_dir}. Interruzione.")

    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            hotel_data = f.read()
        print(f"[✓] Dati hotel caricati da: {data_path} ({len(hotel_data)} caratteri)")
    else:
        print(f"[⚠] File dati_hotel.txt non trovato in {hotel_dir}. Procedo senza dati verificati.")

    if os.path.isfile(recensioni_path):
        with open(recensioni_path, "r", encoding="utf-8") as f:
            recensioni_data = f.read().strip()
        if recensioni_data:
            hotel_data += f"\n\n## RECENSIONI\n{recensioni_data}"
            print(f"[✓] Recensioni caricate da: {recensioni_path} ({len(recensioni_data)} caratteri)")

    if os.path.isfile(recensioni_csv_path):
        reviews_block = _load_reviews_csv(recensioni_csv_path)
        if reviews_block:
            hotel_data += f"\n\n## RECENSIONI REALI OSPITI\n{reviews_block}"
            print(f"[✓] Recensioni CSV caricate da: {recensioni_csv_path}")

    booking_json = _glob_first(hotel_dir, "dataset_booking-reviews-scraper_*.json")
    if booking_json:
        reviews_block = _load_reviews_booking_json(booking_json)
        if reviews_block:
            hotel_data += f"\n\n## RECENSIONI REALI OSPITI (Booking.com)\n{reviews_block}"
            print(f"[✓] Recensioni Booking JSON caricate da: {os.path.basename(booking_json)}")
    else:
        print(f"[  ] Nessun file dataset_booking-reviews-scraper_*.json in {hotel_dir}")

    google_json = _glob_first(hotel_dir, "dataset_Google-Maps-Reviews-Scraper_*.json")
    if google_json:
        reviews_block = _load_reviews_google_json(google_json)
        if reviews_block:
            hotel_data += f"\n\n## RECENSIONI REALI OSPITI (Google Maps)\n{reviews_block}"
            print(f"[✓] Recensioni Google JSON caricate da: {os.path.basename(google_json)}")
    else:
        print(f"[  ] Nessun file dataset_Google-Maps-Reviews-Scraper_*.json in {hotel_dir}")

    print(f"\n{'='*60}")
    print(f"  GEO AUDIT TOOL")
    print(f"  Hotel: {hotel_name}")
    print(f"  Location: {location}")
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    queries = _load_queries(queries_path, hotel_name, location)
    if args.test_mode:
        queries = queries[:3]
        print(f"[⚡] TEST MODE — usando solo le prime 3 query su {len(_load_queries(queries_path, hotel_name, location))}")
    total = len(queries)
    print(f"[→] {total} query caricate. Inizio raccolta risposte in parallelo...\n")

    timestamp = datetime.now().isoformat()
    raw_results_list = []

    # Run queries with a semaphore to avoid overwhelming APIs
    semaphore = asyncio.Semaphore(4)

    async def _bounded_query(q, idx):
        async with semaphore:
            responses = await _run_query(q["text"], idx, total)
            return {
                "query": q["text"],
                "layer": q["layer"],
                "responses": responses,
            }

    tasks = [_bounded_query(q, i + 1) for i, q in enumerate(queries)]
    raw_results_list = await asyncio.gather(*tasks)

    raw_results = {
        "hotel": hotel_name,
        "location": location,
        "timestamp": timestamp,
        "results": list(raw_results_list),
    }

    slug = _slug(hotel_name)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    raw_file = f"results_raw_{slug}_{date_str}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Risposte grezze salvate: {raw_file}")

    print(f"\n[→] Analisi con Claude come giudice...")
    analysed = await analyse_results(raw_results, hotel_data)

    analysed_file = f"results_analysed_{slug}_{date_str}.json"
    with open(analysed_file, "w", encoding="utf-8") as f:
        json.dump(analysed, f, ensure_ascii=False, indent=2)
    print(f"[✓] Risultati analizzati salvati: {analysed_file}")

    print(f"\n[→] Calcolo GEO Score...")
    scores = compute_geo_score(analysed)

    print(f"\n{'='*60}")
    print(f"  GEO SCORE TOTALE:    {scores['geo_score_total']}/100")
    print(f"  Brand Accuracy:      {scores['brand_accuracy']}/100")
    print(f"  Discovery Score:     {scores['discovery_score']}/100")
    print(f"  Score post-fix:      {scores['projected_score']}/100 (stimato)")
    print(f"{'='*60}")

    for eng, stats in scores["per_engine"].items():
        labels = {"claude": "Claude", "gpt4o": "GPT-4o Mini", "perplexity": "Perplexity", "gemini": "Gemini"}
        print(f"  {labels.get(eng, eng):15s} menzioni={stats['mention_rate']}%  "
              f"accuratezza={stats['accuracy_rate']}%  "
              f"allucinazioni={stats['hallucination_count']}")

    if scores["blind_spots"]:
        print(f"\n  ⚠ Blind Spot: {', '.join(scores['blind_spots'])}")

    print(f"\n[→] Generazione report HTML...")
    html_content = generate_report(hotel_name, location, analysed, scores)
    report_file = f"report_{slug}_{date_str}.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[✓] Report HTML generato: {report_file}")
    print(f"\n✅ Audit completato. Apri {report_file} nel browser per il report completo.\n")


if __name__ == "__main__":
    asyncio.run(main())
