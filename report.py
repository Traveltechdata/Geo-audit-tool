from datetime import datetime


def _score_color(score: float) -> str:
    if score >= 70:
        return "#27ae60"
    elif score >= 40:
        return "#e67e22"
    return "#e74c3c"


def _score_label(score: float) -> str:
    if score >= 86:
        return "Visibilità AI Eccellente"
    elif score >= 71:
        return "Visibilità AI Buona"
    elif score >= 51:
        return "Visibilità AI Parziale"
    elif score >= 31:
        return "Visibilità AI Insufficiente"
    return "Visibilità AI Critica"


def _status_icon(status: str) -> str:
    return {"ok": "✅", "warning": "⚠️", "missing": "❌"}.get(status, "—")


def generate_report(hotel_name: str, location: str, analysed_results: dict, scores: dict) -> str:
    date_str = datetime.now().strftime("%d %B %Y")
    geo = scores["geo_score_total"]
    brand = scores["brand_accuracy"]
    disc = scores["discovery_score"]
    proj = scores["projected_score"]
    color = _score_color(geo)
    label = _score_label(geo)

    engines = ["claude", "gpt4o", "perplexity", "gemini"]
    engine_labels = {"claude": "Claude", "gpt4o": "GPT-4o Mini", "perplexity": "Perplexity", "gemini": "Gemini"}

    # Engine stats cards
    engine_cards = ""
    for eng in engines:
        stats = scores["per_engine"].get(eng, {})
        mr = stats.get("mention_rate", 0)
        ar = stats.get("accuracy_rate", 0)
        hc = stats.get("hallucination_count", 0)
        avg_s = stats.get("avg_score", 0)
        mr_color = _score_color(mr)
        engine_cards += f"""
        <div class="engine-card">
          <div class="engine-name">{engine_labels.get(eng, eng)}</div>
          <div class="engine-stat"><span class="stat-label">Menzioni</span><span class="stat-value" style="color:{mr_color}">{mr}%</span></div>
          <div class="engine-stat"><span class="stat-label">Accuratezza</span><span class="stat-value" style="color:{_score_color(ar)}">{ar}%</span></div>
          <div class="engine-stat"><span class="stat-label">Score medio</span><span class="stat-value">{avg_s}/100</span></div>
          <div class="engine-stat"><span class="stat-label">Allucinazioni</span><span class="stat-value" style="color:{'#e74c3c' if hc > 0 else '#27ae60'}">{hc}</span></div>
        </div>"""

    # Query matrix table
    matrix_header = "<tr><th>Query</th><th>Layer</th>" + "".join(f"<th>{engine_labels.get(e, e)}</th>" for e in engines) + "</tr>"
    matrix_rows = ""
    for row in scores["query_matrix"]:
        cells = "".join(
            f"<td class='matrix-cell'>{_status_icon(row['engines'].get(e, {}).get('status', 'missing'))} <small>{row['engines'].get(e, {}).get('score', 0)}/10</small></td>"
            for e in engines
        )
        layer_badge = f"<span class='layer-badge layer-{row['layer']}'>L{row['layer']}</span>"
        query_short = row["query"][:90] + ("…" if len(row["query"]) > 90 else "")
        matrix_rows += f"<tr><td class='query-cell'>{query_short}</td><td>{layer_badge}</td>{cells}</tr>"

    # Blind spots
    blind_spots_html = ""
    if scores["blind_spots"]:
        items = "".join(f"<li><span class='blind-spot-tag'>{kw}</span></li>" for kw in scores["blind_spots"])
        blind_spots_html = f"<ul class='blind-list'>{items}</ul>"
    else:
        blind_spots_html = "<p class='ok-text'>✅ Nessun blind spot critico rilevato.</p>"

    # Hallucinations
    hallucinations_html = ""
    if scores["all_hallucinations"]:
        rows = "".join(
            f"<tr><td><span class='engine-tag'>{engine_labels.get(h['engine'], h['engine'])}</span></td>"
            f"<td><small>{h['query']}</small></td>"
            f"<td class='hallucination-claim'>{h['claim']}</td></tr>"
            for h in scores["all_hallucinations"]
        )
        hallucinations_html = f"""
        <table class="data-table">
          <tr><th>AI</th><th>Query</th><th>Allucinazione rilevata</th></tr>
          {rows}
        </table>"""
    else:
        hallucinations_html = "<p class='ok-text'>✅ Nessuna allucinazione documentata.</p>"

    # Strengths
    strengths_html = ""
    if scores["strengths"]:
        seen = set()
        unique_strengths = []
        for s in scores["strengths"]:
            key = (s["engine"], s["query"][:40])
            if key not in seen:
                seen.add(key)
                unique_strengths.append(s)
        item_parts = []
        for s in unique_strengths[:10]:
            engine_label = engine_labels.get(s["engine"], s["engine"])
            query_text = s["query"]
            if s["keywords"]:
                kw_html = '<span class="kw-tag">' + '</span> <span class="kw-tag">'.join(s["keywords"][:4]) + '</span>'
            else:
                kw_html = ''
            item_parts.append(f'<li><strong>{engine_label}</strong> — "{query_text}" {kw_html}</li>')
        items = "".join(item_parts)
        strengths_html = f"<ul class='strengths-list'>{items}</ul>"
    else:
        strengths_html = "<p>Nessun punto di forza eccellente rilevato. Opportunità di miglioramento significative.</p>"

    # Full responses section — no truncation
    import html as _html
    _engine_order = ["claude", "gpt4o", "perplexity", "gemini"]
    all_responses_html = ""
    for idx, item in enumerate(analysed_results["results"], 1):
        layer = item["layer"]
        query_text = item["query"]
        layer_badge = f"<span class='layer-badge layer-{layer}'>L{layer}</span>"
        blocks = ""
        for eng in _engine_order:
            eng_data = item["responses"].get(eng)
            if eng_data is None:
                continue
            raw_text = eng_data.get("response", "") if isinstance(eng_data, dict) else str(eng_data)
            analysis = eng_data.get("analysis", {}) if isinstance(eng_data, dict) else {}
            score = analysis.get("score_contribution", "—")
            mentioned = analysis.get("hotel_mentioned", False)
            status_icon = "✅" if mentioned and analysis.get("description_accurate") else ("⚠️" if mentioned else "❌")
            safe_text = _html.escape(raw_text)
            _PREVIEW_LEN = 800
            if len(raw_text) > _PREVIEW_LEN:
                safe_preview = _html.escape(raw_text[:_PREVIEW_LEN])
                blocks += f"""
            <div class="response-block engine-{eng}">
              <div class="response-engine-label">{engine_labels.get(eng, eng)}</div>
              <div class="response-text">
                <span class="resp-preview">{safe_preview}</span><span class="resp-rest" style="display:none">{safe_text[_PREVIEW_LEN:]}</span><span class="resp-ellipsis">…</span>
                <a href="#" class="read-more-link" onclick="toggleResp(this);return false;"> [leggi tutto]</a>
              </div>
              <span class="response-score">{status_icon} score {score}/10</span>
            </div>"""
            else:
                blocks += f"""
            <div class="response-block engine-{eng}">
              <div class="response-engine-label">{engine_labels.get(eng, eng)}</div>
              <div class="response-text">{safe_text}</div>
              <span class="response-score">{status_icon} score {score}/10</span>
            </div>"""
        summary_label = f"Query {idx:02d} {layer_badge} — {_html.escape(query_text[:100])}{'…' if len(query_text) > 100 else ''}"
        all_responses_html += f"""
        <details>
          <summary>{summary_label}</summary>
          <div class="response-grid">{blocks}</div>
        </details>"""

    # Top keywords
    kw_tags = "".join(
        f"<span class='kw-tag-freq'>{kw} <span class='kw-count'>×{cnt}</span></span>"
        for kw, cnt in scores["top_keywords"][:15]
    )

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GEO Audit — {hotel_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #2c3e50; line-height: 1.6; }}
    .header {{ background: #2c3e50; color: white; padding: 40px; text-align: center; }}
    .header h1 {{ font-size: 2.2em; margin-bottom: 6px; }}
    .header .location {{ color: #bdc3c7; font-size: 1.1em; margin-bottom: 4px; }}
    .header .date {{ color: #95a5a6; font-size: 0.9em; }}
    .geo-badge {{ display: inline-block; background: {color}; color: white; border-radius: 12px;
                  padding: 20px 40px; margin: 24px auto; font-size: 3em; font-weight: bold; }}
    .geo-label {{ font-size: 0.4em; display: block; letter-spacing: 2px; margin-top: 4px; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px; }}
    .section {{ background: white; border-radius: 8px; padding: 28px; margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
    .section h2 {{ font-size: 1.3em; color: #2c3e50; border-bottom: 2px solid #ecf0f1;
                   padding-bottom: 10px; margin-bottom: 20px; }}
    .score-cards {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .score-card {{ flex: 1; min-width: 180px; background: #f8f9fa; border-radius: 8px;
                   padding: 20px; text-align: center; border-top: 4px solid #2c3e50; }}
    .score-card .value {{ font-size: 2.5em; font-weight: bold; }}
    .score-card .label {{ color: #7f8c8d; font-size: 0.85em; margin-top: 4px; }}
    .engine-cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .engine-card {{ flex: 1; min-width: 200px; background: #f8f9fa; border-radius: 8px; padding: 18px;
                    border-left: 4px solid #3498db; }}
    .engine-name {{ font-weight: bold; font-size: 1.05em; margin-bottom: 10px; color: #2c3e50; }}
    .engine-stat {{ display: flex; justify-content: space-between; padding: 3px 0;
                    border-bottom: 1px solid #ecf0f1; font-size: 0.9em; }}
    .stat-label {{ color: #7f8c8d; }}
    .stat-value {{ font-weight: bold; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
    .data-table th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
    .data-table td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; vertical-align: top; }}
    .data-table tr:hover {{ background: #f8f9fa; }}
    .matrix-cell {{ text-align: center; white-space: nowrap; }}
    .query-cell {{ max-width: 320px; font-size: 0.85em; color: #555; }}
    .layer-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                    font-size: 0.75em; font-weight: bold; color: white; }}
    .layer-1 {{ background: #3498db; }}
    .layer-2 {{ background: #9b59b6; }}
    .layer-3 {{ background: #e67e22; }}
    .layer-4 {{ background: #1abc9c; }}
    .blind-list {{ list-style: none; display: flex; flex-wrap: wrap; gap: 10px; }}
    .blind-spot-tag {{ display: inline-block; background: #ffeaa7; color: #d35400;
                       border: 1px solid #f39c12; border-radius: 4px; padding: 4px 10px; font-size: 0.9em; }}
    .kw-tag {{ display: inline-block; background: #dfe6e9; color: #2c3e50;
               border-radius: 4px; padding: 2px 8px; font-size: 0.8em; margin: 2px; }}
    .kw-tag-freq {{ display: inline-block; background: #e8f4fd; color: #2980b9;
                    border: 1px solid #aed6f1; border-radius: 4px; padding: 4px 10px; margin: 4px; font-size: 0.88em; }}
    .kw-count {{ background: #2980b9; color: white; border-radius: 10px;
                 padding: 1px 6px; font-size: 0.8em; margin-left: 4px; }}
    .engine-tag {{ display: inline-block; background: #2c3e50; color: white;
                   border-radius: 4px; padding: 2px 8px; font-size: 0.8em; }}
    .hallucination-claim {{ color: #c0392b; font-size: 0.9em; }}
    .strengths-list {{ list-style: none; }}
    .strengths-list li {{ padding: 8px 0; border-bottom: 1px solid #ecf0f1; font-size: 0.92em; }}
    .ok-text {{ color: #27ae60; font-weight: bold; }}
    .projection-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; border-radius: 8px; padding: 28px; text-align: center; }}
    .projection-box .proj-score {{ font-size: 3em; font-weight: bold; margin: 10px 0; }}
    .projection-box p {{ opacity: 0.9; font-size: 0.95em; }}
    .footer {{ text-align: center; color: #95a5a6; font-size: 0.8em; padding: 20px; }}
    details {{ border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 10px; }}
    details[open] {{ border-color: #bdc3c7; }}
    summary {{ cursor: pointer; padding: 10px 14px; font-weight: bold; font-size: 0.92em;
               color: #2c3e50; list-style: none; display: flex; align-items: center; gap: 8px; }}
    summary::-webkit-details-marker {{ display: none; }}
    summary::before {{ content: "▶"; font-size: 0.7em; color: #95a5a6; transition: transform 0.15s; }}
    details[open] summary::before {{ transform: rotate(90deg); }}
    .response-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                      gap: 14px; padding: 14px; }}
    .response-block {{ background: #f8f9fa; border-radius: 6px; padding: 14px;
                       border-left: 3px solid #3498db; }}
    .response-block.engine-claude {{ border-color: #8e44ad; }}
    .response-block.engine-gpt4o {{ border-color: #27ae60; }}
    .response-block.engine-perplexity {{ border-color: #e67e22; }}
    .response-block.engine-gemini {{ border-color: #2980b9; }}
    .response-engine-label {{ font-size: 0.78em; font-weight: bold; text-transform: uppercase;
                               letter-spacing: 0.5px; color: #7f8c8d; margin-bottom: 6px; }}
    .response-text {{ font-size: 0.88em; color: #2c3e50; white-space: pre-wrap;
                      word-break: break-word; line-height: 1.55; }}
    .response-score {{ display: inline-block; margin-top: 8px; font-size: 0.78em;
                       background: #ecf0f1; border-radius: 4px; padding: 2px 7px; color: #555; }}
    .read-more-link {{ font-size: 0.82em; color: #2980b9; text-decoration: none; white-space: nowrap; }}
    .read-more-link:hover {{ text-decoration: underline; }}
    @media (max-width: 700px) {{
      .score-cards, .engine-cards {{ flex-direction: column; }}
      .geo-badge {{ font-size: 2em; padding: 15px 25px; }}
      .response-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>{hotel_name}</h1>
  <div class="location">📍 {location}</div>
  <div class="date">Report GEO Audit — {date_str}</div>
  <div>
    <div class="geo-badge">
      {geo}/100
      <span class="geo-label">GEO SCORE TOTALE — {label}</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- Score Summary -->
  <div class="section">
    <h2>📊 Riepilogo Punteggi</h2>
    <div class="score-cards">
      <div class="score-card">
        <div class="value" style="color:{_score_color(brand)}">{brand}</div>
        <div class="label">Brand Accuracy<br><small>Layer 1 + 2</small></div>
      </div>
      <div class="score-card">
        <div class="value" style="color:{_score_color(disc)}">{disc}</div>
        <div class="label">Discovery Score<br><small>Layer 2 + 3 + 4</small></div>
      </div>
      <div class="score-card">
        <div class="value" style="color:{color}">{geo}</div>
        <div class="label">GEO Score Totale<br><small>Media pesata 50/50</small></div>
      </div>
    </div>
  </div>

  <!-- Per-AI Section -->
  <div class="section">
    <h2>🤖 Performance per AI</h2>
    <div class="engine-cards">{engine_cards}</div>
  </div>

  <!-- Query Matrix -->
  <div class="section">
    <h2>🗂️ Matrice Query × AI</h2>
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:14px;">
      ✅ Menzionato e accurato &nbsp;|&nbsp; ⚠️ Menzionato con problemi &nbsp;|&nbsp; ❌ Non menzionato
    </p>
    <div style="overflow-x:auto;">
      <table class="data-table">
        {matrix_header}
        {matrix_rows}
      </table>
    </div>
  </div>

  <!-- Full Responses -->
  <div class="section">
    <h2>📄 Tutte le Risposte AI</h2>
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:14px;">
      Testo completo di ogni risposta per ogni query. Clicca su una query per espanderla.
    </p>
    {all_responses_html}
  </div>

  <!-- Top Keywords -->
  <div class="section">
    <h2>🔑 Keyword di Scoperta Rilevate</h2>
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:14px;">Keyword presenti nelle risposte AI che contribuiscono alla scoperta organica.</p>
    <div>{kw_tags if kw_tags else '<p style="color:#999">Nessuna keyword di scoperta rilevata.</p>'}</div>
  </div>

  <!-- Blind Spots -->
  <div class="section">
    <h2>🕳️ Blind Spot — Keyword Critiche Mancanti</h2>
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:14px;">Keyword attese ma assenti nelle risposte AI. Richiedono intervento di contenuto.</p>
    {blind_spots_html}
  </div>

  <!-- Hallucinations -->
  <div class="section">
    <h2>⚠️ Allucinazioni Documentate</h2>
    <p style="color:#7f8c8d; font-size:0.85em; margin-bottom:14px;">Informazioni false o non verificabili generate dalle AI sul tuo hotel.</p>
    {hallucinations_html}
  </div>

  <!-- Strengths -->
  <div class="section">
    <h2>💪 Punti di Forza — Cosa Funziona Già</h2>
    {strengths_html}
  </div>

  <!-- Projection -->
  <div class="section">
    <h2>🚀 Proiezione GEO Score Post-Intervento</h2>
    <div class="projection-box">
      <p>Se risolvi i blind spot e le inaccuratezze rilevate, il GEO Score stimato sarà:</p>
      <div class="proj-score">{proj}/100</div>
      <p>+{round(proj - geo, 1)} punti rispetto al punteggio attuale di {geo}/100<br>
         <small>Stima basata su {len(scores['blind_spots'])} blind spot e gap di visibilità per AI</small></p>
    </div>
  </div>

</div>

<div class="footer">
  GEO Audit Tool — Generato il {date_str} &nbsp;|&nbsp; Powered by Claude, GPT-4o Mini, Perplexity, Gemini
</div>

<script>
function toggleResp(link) {{
  var block = link.parentElement;
  var rest = block.querySelector('.resp-rest');
  var ellipsis = block.querySelector('.resp-ellipsis');
  if (rest.style.display === 'none') {{
    rest.style.display = 'inline';
    ellipsis.style.display = 'none';
    link.textContent = ' [mostra meno]';
  }} else {{
    rest.style.display = 'none';
    ellipsis.style.display = 'inline';
    link.textContent = ' [leggi tutto]';
  }}
}}
</script>

</body>
</html>"""

    return html
