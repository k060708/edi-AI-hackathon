"""
compare.py
Head-to-head test: plain search API results vs. search API + your algorithm
(relevance/diversity ranking + guaranteed Google Scholar sources).

Both lists are scored on the same metrics, a scorecard shows which side wins
each one, and an overall verdict is printed.

Usage:
    export SERPER_API_KEY="your_key"
    python3 compare.py "school uniforms should be compulsory"
    python3 compare.py "global warming" --limit 8 --no-html
    python3 compare.py "AI in healthcare" --min-scholar 2 --open

    # Reduce bias by scoring relevance with a different embedding model than
    # the one your algorithm ranks with:
    python3 compare.py "global warming" --eval-model all-mpnet-base-v2
"""

import argparse
import html
import inspect
import math
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

import scholar

algorithm = scholar.algorithm

# (label, key, number format, higher_is_better, tie tolerance)
SCORECARD = [
    ("Ranking quality (nDCG)", "ndcg", ".3f", True, 0.005),
    ("Average relevance", "avg_relevance", ".3f", True, 0.005),
    ("Average metadata quality", "avg_quality", ".3f", True, 0.005),
    ("Unique domains", "unique_domains", "d", True, 0.5),
    ("Average similarity between results", "avg_similarity", ".3f", False, 0.005),
    ("Scholar sources", "scholar_count", "d", True, 0.5),
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _short(text, n=80):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _url_key(page):
    return (page.get("url") or "").strip().lower().rstrip("/")


def _page_key(page):
    """Stable identity for a page, even if the URL is missing."""
    return _url_key(page) or "title:" + (page.get("title") or "").strip().lower()


def _hostname(page):
    host = urlparse(page.get("url") or "").netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or (page.get("domain") or "").lower()


def _safe_link(url):
    """Only allow http(s) links in the HTML report."""
    return url if url.lower().startswith(("http://", "https://")) else ""


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _embed_pool(query, pools, model):
    """Embed the query and every distinct page across all lists exactly once."""
    unique = {}
    for pool in pools:
        for page in pool:
            unique.setdefault(_page_key(page), page)

    keys = list(unique)
    texts = [
        f"{algorithm._text(unique[k], 'title')}. "
        f"{algorithm._text(unique[k], 'snippet')}"
        for k in keys
    ]
    vectors = model.encode([query] + texts, normalize_embeddings=True)
    return vectors[0], dict(zip(keys, vectors[1:]))


def _make_rows(pages, query_emb, emb_by_key, limit):
    rows, embs = [], []
    for page in pages[:limit]:
        emb = emb_by_key[_page_key(page)]
        rows.append({
            "page": page,
            "title": page.get("title") or "(no title)",
            "url": page.get("url") or "",
            "domain": _hostname(page),
            "rel": float(np.clip(emb @ query_emb, 0.0, 1.0)),
            "qual": algorithm.calculate_quality(page),
            "source": page.get("source", "web"),
            "forced": bool(page.get("forced_scholar")),
        })
        embs.append(emb)
    return rows, (np.array(embs) if embs else np.empty((0, 0)))


def _dcg(values):
    return sum(v / math.log2(i + 2) for i, v in enumerate(values))


def _metrics(rows, embs, ideal_rels):
    if not rows:
        return {k: 0 for _, k, *_ in SCORECARD}

    domains = {r["domain"] or r["url"] for r in rows if r["domain"] or r["url"]}

    if len(rows) > 1:
        sim = embs @ embs.T
        avg_similarity = float(np.mean(sim[np.triu_indices(len(rows), k=1)]))
    else:
        avg_similarity = 0.0

    # nDCG: how close is the ORDER of this list to the best possible order of
    # all the candidate pages we saw (1.0 = most relevant pages, best first).
    ideal = _dcg(sorted(ideal_rels, reverse=True)[: len(rows)])
    ndcg = _dcg([r["rel"] for r in rows]) / ideal if ideal > 0 else 0.0

    return {
        "ndcg": float(ndcg),
        "avg_relevance": float(np.mean([r["rel"] for r in rows])),
        "avg_quality": float(np.mean([r["qual"] for r in rows])),
        "unique_domains": len(domains),
        "avg_similarity": avg_similarity,
        "scholar_count": sum(r["source"] == "scholar" for r in rows),
    }


def _scorecard(web_m, enh_m):
    """Decide a winner for each metric and an overall verdict."""
    lines, web_wins, enh_wins = [], 0, 0

    for label, key, fmt, higher_better, tol in SCORECARD:
        diff = enh_m[key] - web_m[key]
        if abs(diff) < tol:
            winner = "Tie"
        elif (diff > 0) == higher_better:
            winner, enh_wins = "Enhanced", enh_wins + 1
        else:
            winner, web_wins = "Web only", web_wins + 1

        lines.append({
            "label": label,
            "web": format(web_m[key], fmt),
            "enh": format(enh_m[key], fmt),
            "diff": format(diff, "+" + fmt if fmt != "d" else "+d"),
            "winner": winner,
            "better": "higher" if higher_better else "lower",
        })

    total = len(SCORECARD)
    if enh_wins > web_wins:
        verdict = (f"Search API + your algorithm is better: it wins {enh_wins} of "
                   f"{total} metrics (plain search wins {web_wins}).")
    elif web_wins > enh_wins:
        verdict = (f"Plain search API is better: it wins {web_wins} of {total} "
                   f"metrics (enhanced wins {enh_wins}).")
    else:
        verdict = (f"No clear winner: enhanced wins {enh_wins}, plain search wins "
                   f"{web_wins}, the rest tie.")

    return lines, verdict, enh_wins, web_wins


# --------------------------------------------------------------------------
# Build the comparison
# --------------------------------------------------------------------------
def build_comparison(query, raw_pages, ranked_pages, info, limit, min_relevance,
                     min_quality, eval_model=None):
    model = eval_model or algorithm.model
    query_emb, emb_by_key = _embed_pool(query, [raw_pages, ranked_pages], model)

    raw_rows, raw_embs = _make_rows(raw_pages, query_emb, emb_by_key, limit)
    ranked_rows, ranked_embs = _make_rows(ranked_pages, query_emb, emb_by_key, limit)

    # Relevance of every distinct candidate = the pool the ideal ordering uses.
    ideal_rels = [
        float(np.clip(e @ query_emb, 0.0, 1.0)) for e in emb_by_key.values()
    ]

    # Position in the ordinary list (first occurrence wins).
    raw_positions = {}
    for i, page in enumerate(raw_pages, start=1):
        key = _url_key(page)
        if key:
            raw_positions.setdefault(key, i)

    ranked_keys = {_page_key(r["page"]) for r in ranked_rows}

    for i, row in enumerate(ranked_rows, start=1):
        row["pos"] = i
        row["orig"] = raw_positions.get(_url_key(row["page"]))
        if row["orig"] is None:
            row["move"] = "NEW"
        elif row["orig"] > i:
            row["move"] = f"UP {row['orig'] - i}"
        elif row["orig"] < i:
            row["move"] = f"DOWN {i - row['orig']}"
        else:
            row["move"] = "SAME"

    for i, row in enumerate(raw_rows, start=1):
        row["pos"] = i

    dropped = []
    for row in raw_rows:
        if _page_key(row["page"]) in ranked_keys:
            continue
        reasons = []
        if row["rel"] < min_relevance:
            reasons.append("below relevance threshold")
        if row["qual"] < min_quality:
            reasons.append("below metadata-quality threshold")
        if not reasons:
            reasons.append("outranked by more relevant or more varied results")
        row["reason"] = "; ".join(reasons)
        dropped.append(row)

    web_m = _metrics(raw_rows, raw_embs, ideal_rels)
    enh_m = _metrics(ranked_rows, ranked_embs, ideal_rels)
    card, verdict, enh_wins, web_wins = _scorecard(web_m, enh_m)

    raw_keys = {_page_key(r["page"]) for r in raw_rows}

    return {
        "query": query,
        "fetched": len(raw_pages),
        "ranked_fetched": len(ranked_pages),
        "limit": limit,
        "min_relevance": min_relevance,
        "min_quality": min_quality,
        "info": info,
        "raw": raw_rows,
        "ranked": ranked_rows,
        "dropped": dropped,
        "pulled_in": [r for r in ranked_rows if r["move"] == "NEW"],
        "overlap": len(raw_keys & ranked_keys),
        "web_metrics": web_m,
        "enh_metrics": enh_m,
        "scorecard": card,
        "verdict": verdict,
        "enh_wins": enh_wins,
        "web_wins": web_wins,
        "separate_eval_model": eval_model is not None,
    }


# --------------------------------------------------------------------------
# Terminal report
# --------------------------------------------------------------------------
CAVEATS = [
    "Metadata quality is a completeness heuristic, not a credibility or fact-check score.",
    "Text similarity is a rough proxy for variety, not for viewpoint diversity or accuracy.",
]


def print_report(data):
    line = "=" * 90
    print(f"\n{line}\nSEARCH COMPARISON\nQuery: {data['query']}")
    print(f"Plain web results fetched:   {data['fetched']}")
    print(f"Enhanced results fetched:    {data['ranked_fetched']}")
    print(f"Top {data['limit']} compared. Results shared by both lists: {data['overlap']}")
    if data["info"]["warning"]:
        print(f"Warning: {data['info']['warning']}")

    print(f"\nA) PLAIN SEARCH API (original order)\n{'-' * 90}")
    for row in data["raw"]:
        print(f"{row['pos']:>2}. [{row['rel']:.2f} rel, {row['qual']:.2f} qual] "
              f"{_short(row['title'], 62)}")
        print(f"    {row['url']}")

    print(f"\nB) SEARCH API + YOUR ALGORITHM\n{'-' * 90}")
    for row in data["ranked"]:
        tag = "[SCHOLAR]" if row["source"] == "scholar" else "[WEB]"
        origin = f"was #{row['orig']}" if row["orig"] else "NEW"
        print(f"{row['pos']:>2}. {tag:<9} {origin:<8} "
              f"[{row['rel']:.2f} rel, {row['qual']:.2f} qual] "
              f"{_short(row['title'], 48)}")
        print(f"    {row['url']}")
        if row["page"].get("cited_by") is not None:
            print(f"    Citations: {row['page']['cited_by']}")
        if row["forced"]:
            print("    Note: added to meet the Scholar-source minimum")

    if data["dropped"]:
        print(f"\nREMOVED FROM THE PLAIN TOP {data['limit']}\n{'-' * 90}")
        for row in data["dropped"]:
            print(f"- {_short(row['title'])}: {row['reason']}")

    if data["pulled_in"]:
        print(f"\nPULLED IN BY THE ALGORITHM\n{'-' * 90}")
        for row in data["pulled_in"]:
            print(f"+ [{row['source']}] {_short(row['title'])}")

    print(f"\nSCORECARD\n{'-' * 90}")
    print(f"{'Metric':<38}{'Web only':>10}{'Enhanced':>10}{'Diff':>9}   Winner")
    for r in data["scorecard"]:
        print(f"{r['label']:<38}{r['web']:>10}{r['enh']:>10}{r['diff']:>9}   {r['winner']}")

    print(f"\nVERDICT: {data['verdict']}")

    print("\nHow to read this:")
    print("  nDCG = how well the ORDER matches the most-relevant-first ideal (1.0 is perfect).")
    print("  Similarity: lower means the results are more varied.")
    for caveat in CAVEATS:
        print(f"  {caveat}")
    if not data["separate_eval_model"]:
        print("  Relevance is scored with the same model your algorithm ranks with, which")
        print("  favours the enhanced list. Use --eval-model for a fairer test.")


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
def write_html(data, path):
    def esc(value):
        return html.escape(str(value if value is not None else ""))

    def move_html(row):
        move = row["move"]
        if move == "NEW":
            return "<span class='new'>NEW</span>"
        if move.startswith("UP"):
            return f"<span class='up'>▲{esc(move.split()[1])}</span>"
        if move.startswith("DOWN"):
            return f"<span class='down'>▼{esc(move.split()[1])}</span>"
        return "<span class='same'>–</span>"

    def result_rows(rows, enhanced):
        cols = 6 if enhanced else 4
        out = []
        for row in rows:
            title = esc(_short(row["title"], 100))
            href = _safe_link(row["url"])
            link = (f'<a href="{esc(href)}" target="_blank" rel="noopener noreferrer">'
                    f"{title}</a>") if href else title

            tag = ""
            detail = ""
            extra = ""
            if enhanced:
                if row["source"] == "scholar":
                    tag = "<span class='tag scholar'>Scholar</span> "
                    page = row["page"]
                    bits = []
                    if page.get("publication_info"):
                        bits.append(esc(_short(page["publication_info"], 80)))
                    if page.get("cited_by") is not None:
                        bits.append(f"cited by {esc(page['cited_by'])}")
                    if row["forced"]:
                        bits.append("added to guarantee a Scholar source")
                    if bits:
                        detail = f"<div class='domain'>{' · '.join(bits)}</div>"
                was = f"#{row['orig']}" if row["orig"] is not None else "–"
                extra = f"<td class='muted'>{was}</td><td>{move_html(row)}</td>"

            cells = (f"<td class='muted'>{row['pos']}</td>"
                     f"<td>{tag}{link}<div class='domain'>{esc(row['domain'])}</div>{detail}</td>"
                     f"{extra}<td>{row['rel']:.2f}</td><td>{row['qual']:.2f}</td>")
            out.append(f"<tr>{cells}</tr>")
        return "".join(out) or f"<tr><td colspan='{cols}'>No results</td></tr>"

    score_rows = "".join(
        f"<tr class='w-{esc(r['winner'].split()[0].lower())}'>"
        f"<td>{esc(r['label'])}<div class='domain'>{esc(r['better'])} is better</div></td>"
        f"<td>{esc(r['web'])}</td><td>{esc(r['enh'])}</td>"
        f"<td>{esc(r['diff'])}</td><td><strong>{esc(r['winner'])}</strong></td></tr>"
        for r in data["scorecard"]
    )

    dropped = "".join(
        f"<li>{esc(r['title'])} <span class='domain'>— {esc(r['reason'])}</span></li>"
        for r in data["dropped"]) or "<li>None</li>"
    pulled = "".join(
        f"<li>{esc(r['title'])} <span class='domain'>({esc(r['source'])})</span></li>"
        for r in data["pulled_in"]) or "<li>None</li>"

    warning = data["info"]["warning"]
    warning_html = f"<p class='warn'>{esc(warning)}</p>" if warning else ""
    bias_note = "" if data["separate_eval_model"] else (
        "Relevance is scored with the same model the algorithm ranks with, "
        "which favours the enhanced list. Re-run with <code>--eval-model</code> "
        "for a fairer test.")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search comparison</title>
<style>
:root{{color-scheme:light}}
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:1240px;margin:0 auto;padding:32px 20px 60px;
background:#ffffff;color:#0b1f4d}}
h1{{font-size:28px;margin:0 0 6px}}
h2{{font-size:20px;margin:0 0 14px}}
h3{{font-size:18px;margin:34px 0 12px}}
.sub{{color:#3d5282;margin:0 0 22px}}.sub strong{{color:#0b1f4d}}
.verdict{{background:#eef3fb;border:1px solid #b9c8e6;border-radius:12px;padding:14px 18px;font-size:17px;margin:0 0 22px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
section,.card{{background:#ffffff;border:1px solid #c9d4ea;border-radius:12px;padding:18px;overflow:auto}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px 8px;border-bottom:1px solid #dde4f2;text-align:left;vertical-align:top}}
tr:last-child td{{border-bottom:0}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#3d5282;font-weight:600}}
a{{color:#0b1f4d;text-decoration:none}}a:hover{{text-decoration:underline}}
.domain{{font-size:13px;color:#3d5282}}.muted{{color:#3d5282}}
.tag{{font-size:11px;text-transform:uppercase;background:#e4eaf6;padding:2px 6px;border-radius:4px;margin-right:4px}}
.tag.scholar{{background:#dff1e5;color:#14532d}}
.up{{color:#15803d;font-weight:600}}.down{{color:#b91c1c;font-weight:600}}
.new{{color:#b45309;font-weight:600;font-size:13px}}.same{{color:#3d5282}}
.w-enhanced td:last-child{{color:#15803d}}.w-web td:last-child{{color:#b91c1c}}.w-tie td:last-child{{color:#3d5282}}
.warn{{color:#b45309}}
ul{{padding-left:20px}}li{{margin-bottom:6px}}
.note{{color:#3d5282;font-size:14px}}
code{{background:#e4eaf6;padding:1px 5px;border-radius:4px}}
@media(max-width:860px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Search comparison</h1>
<p class="sub">Query: <strong>{esc(data['query'])}</strong> · {data['fetched']} web results fetched
· comparing top {data['limit']} · filters: relevance ≥ {data['min_relevance']:.2f}, quality ≥ {data['min_quality']:.2f}
· {data['info']['scholar_in_results']} Google Scholar source(s) in the enhanced list</p>
{warning_html}
<div class="verdict"><strong>Verdict:</strong> {esc(data['verdict'])}</div>
<div class="grid">
<section><h2>A) Search API only</h2>
<table><tr><th>#</th><th>Result</th><th>Rel</th><th>Qual</th></tr>
{result_rows(data['raw'], False)}</table></section>
<section><h2>B) With your algorithm</h2>
<table><tr><th>#</th><th>Result</th><th>Was</th><th>Move</th><th>Rel</th><th>Qual</th></tr>
{result_rows(data['ranked'], True)}</table></section>
</div>
<h3>Scorecard</h3>
<div class="card"><table><tr><th>Metric</th><th>Web only</th><th>Enhanced</th><th>Diff</th><th>Winner</th></tr>
{score_rows}</table></div>
<h3>Removed from the plain top {data['limit']}</h3><ul>{dropped}</ul>
<h3>Pulled in by the algorithm</h3><ul>{pulled}</ul>
<h3>How to read this</h3>
<p class="note">nDCG measures how well the order matches a most-relevant-first ideal built from all
candidate pages (1.0 is perfect). Lower similarity means results are more varied.
{esc(CAVEATS[0])} {esc(CAVEATS[1])}</p>
<p class="note">{bias_note}</p>
</body></html>"""

    with open(path, "w", encoding="utf-8") as file:
        file.write(doc)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare plain search API results with search API + your algorithm."
    )
    parser.add_argument("query", nargs="*", help="search query")
    parser.add_argument("--limit", type=int, default=10, help="results compared (default 10)")
    parser.add_argument("--count", type=int, default=20, help="web results to fetch (default 20)")
    parser.add_argument("--min-scholar", type=int, default=2)
    parser.add_argument("--min-relevance", type=float, default=0.30)
    parser.add_argument("--min-quality", type=float, default=0.60)
    parser.add_argument("--eval-model", default=None,
                        help="separate SentenceTransformer model used only for scoring")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--html", default="comparison_report.html")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    limit = max(1, args.limit)
    min_scholar = max(0, min(args.min_scholar, limit))
    if args.min_scholar > limit:
        print(f"Note: --min-scholar {args.min_scholar} is larger than --limit; using {limit}.")

    query = " ".join(args.query).strip() or input("What would you like to search for? ").strip()
    if not query:
        print("Please enter a search query.")
        return 1

    try:
        eval_model = None
        if args.eval_model:
            from sentence_transformers import SentenceTransformer
            eval_model = SentenceTransformer(args.eval_model)

        # A) plain search API results, in the API's own order.
        raw_pages = algorithm.search_web(query, count=args.count)

        # B) search API + your algorithm (+ Scholar). Reuse the plain results
        #    if search_with_scholar accepts them, to save a Serper query.
        kwargs = dict(
            count=args.count,
            limit=limit,
            min_scholar=min_scholar,
            min_relevance=args.min_relevance,
            min_quality=args.min_quality,
        )
        if "web_pages" in inspect.signature(scholar.search_with_scholar).parameters:
            kwargs["web_pages"] = raw_pages
        enhanced_pages, info = scholar.search_with_scholar(query, **kwargs)

        data = build_comparison(
            query, raw_pages, enhanced_pages, info,
            limit=limit,
            min_relevance=args.min_relevance,
            min_quality=args.min_quality,
            eval_model=eval_model,
        )

    except requests.RequestException as error:
        print(f"Search request failed: {error}")
        return 1
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    print_report(data)

    if not args.no_html:
        write_html(data, args.html)
        path = Path(args.html).resolve()
        print(f"\nHTML report saved to: {path}")
        if args.open:
            webbrowser.open(path.as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())