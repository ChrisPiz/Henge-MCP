"""Persistence for TenthAI reports — HTML + JSON per run + browseable index.

Layout (under TENTHAI_REPORTS_DIR or ~/.tenthai/reports):

    20260430-143559_should-i-leave-pm-job/
        report.html       # full editorial visualization
        report.json       # raw data (question, context, 10 responses, distances, summary)
    index.html            # auto-regenerated browseable ledger of all reports

The JSON is the canonical record. The HTML is a rendered view of it. The index
is rebuilt from scratch every time write_index() runs, so deleting a report
directory removes it from the ledger on the next invocation.
"""
import html as html_mod
import json
import os
import re
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(os.environ.get("TENTHAI_REPORTS_DIR", "~/.tenthai/reports")).expanduser()


def slugify(question: str, max_len: int = 60) -> str:
    """Lowercase ASCII slug for filesystem use. Strips accents to plain letters."""
    text = question.lower()
    # Strip common Spanish accents to keep filesystem-friendly slugs
    accent_map = str.maketrans("áéíóúñü", "aeiounu")
    text = text.translate(accent_map)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "report"


def make_report_id(question: str) -> str:
    """Returns ``{YYYYMMDD-HHMMSS}_{slug}`` — collision-resistant within one second."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{slugify(question)}"


def make_report_dir(report_id: str) -> Path:
    path = REPORTS_DIR / report_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_record(report_dir: Path, html: str, payload: dict) -> tuple[Path, Path]:
    """Writes ``report.html`` + ``report.json`` into report_dir. Returns both paths."""
    html_path = report_dir / "report.html"
    json_path = report_dir / "report.json"
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return html_path, json_path


def list_records() -> list:
    """Scan REPORTS_DIR for ``report.json`` files. Returns desc by timestamp.

    Malformed JSON is skipped silently — the ledger should not break because
    of a corrupted run.
    """
    if not REPORTS_DIR.exists():
        return []
    records = []
    for json_path in sorted(REPORTS_DIR.glob("*/report.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Stamp the on-disk dir name so we can build relative links even
        # if the in-file id drifts from the directory name.
        data["_dir"] = json_path.parent.name
        records.append(data)
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


def _format_row(record: dict) -> str:
    rid = record.get("_dir") or record.get("id", "")
    ts = record.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%Y·%m·%d")
        time_str = dt.strftime("%H:%M")
    except Exception:
        date_str = ts[:10] if ts else "—"
        time_str = ts[11:16] if len(ts) > 11 else ""

    question = record.get("question", "")
    if len(question) > 110:
        question = question[:108].rsplit(" ", 1)[0] + "…"

    summary = record.get("summary", {})
    state = summary.get("consensus_state")
    fragility = summary.get("consensus_fragility", "—")
    state_labels = {
        "aligned-stable": "aligned",
        "aligned-fragile": "fragile consensus",
        "divided": "divided",
    }
    if state in state_labels:
        verdict = state_labels[state]
    else:
        # Fallback for legacy records without consensus_state
        lowered = fragility.lower()
        if "frágil" in lowered or "fragile" in lowered:
            verdict = "fragile consensus"
        elif "alineados" in lowered or "aligned" in lowered:
            verdict = "aligned"
        elif "divididos" in lowered or "divided" in lowered or "moderate" in lowered:
            verdict = "divided"
        else:
            verdict = fragility[:32]

    tenth_d = summary.get("tenth_man_distance")
    tenth_str = f"{tenth_d:.3f}" if isinstance(tenth_d, (int, float)) else "—"

    q_safe = html_mod.escape(question)
    rid_safe = html_mod.escape(rid)
    return (
        f"<tr>"
        f"<td class=\"date\"><div class=\"d\">{date_str}</div>"
        f"<div class=\"t\">{time_str}</div></td>"
        f"<td class=\"q\">{q_safe}</td>"
        f"<td class=\"v\">{verdict}</td>"
        f"<td class=\"td\">{tenth_str}</td>"
        f"<td class=\"link\"><a href=\"./{rid_safe}/report.html\">open ↗</a></td>"
        f"</tr>"
    )


def _index_html(records: list) -> str:
    if records:
        rows_html = "\n".join(_format_row(r) for r in records)
        body = (
            "<table class=\"ledger\">"
            "<thead><tr>"
            "<th class=\"date\">Date</th>"
            "<th class=\"q\">Question</th>"
            "<th class=\"v\">Verdict</th>"
            "<th class=\"td\">d 10</th>"
            "<th class=\"link\"></th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
    else:
        body = (
            "<p class=\"empty\">No decisions recorded yet. "
            "Run <code>/decidir</code> in Claude Code to create your first report.</p>"
        )

    count = len(records)
    count_label = f"{count} report{'s' if count != 1 else ''}"
    updated = datetime.now().strftime("%Y·%m·%d · %H:%M")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TenthAI · Decision history</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,0..100;1,9..144,300..700,0..100&family=Inter+Tight:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: oklch(96.5% 0.012 78);
    --paper: oklch(98.5% 0.008 78);
    --ink: oklch(20% 0.012 60);
    --ink-2: oklch(38% 0.012 60);
    --ink-3: oklch(56% 0.012 60);
    --ink-4: oklch(72% 0.010 70);
    --rule: oklch(86% 0.012 70);
    --rule-2: oklch(91% 0.012 70);
    --accent: oklch(56% 0.165 32);
    --accent-soft: oklch(92% 0.045 32);
    --serif: 'Fraunces', 'Iowan Old Style', Georgia, serif;
    --sans: 'Inter Tight', system-ui, sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: var(--sans);
    background: var(--bg);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    line-height: 1.55;
  }}
  ::selection {{ background: var(--accent-soft); color: var(--ink); }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 56px 56px 96px; }}
  .mast {{
    display: flex; align-items: center; gap: 14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 28px;
  }}
  .mast .rule {{ width: 28px; height: 1px; background: var(--ink-3); }}
  .mast .dot {{
    width: 7px; height: 7px; border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }}
  .mast b {{ color: var(--ink); font-weight: 500; }}
  .mast .sep {{ color: var(--ink-4); }}
  h1 {{
    font-family: var(--serif);
    font-weight: 380;
    font-size: clamp(36px, 5vw, 56px);
    line-height: 1.05;
    letter-spacing: -0.025em;
    margin: 0 0 8px;
    font-variation-settings: "SOFT" 30, "opsz" 144;
  }}
  h1 em {{ font-style: italic; color: var(--accent); font-weight: 360; }}
  .sub {{
    font-family: var(--serif);
    font-style: italic;
    font-weight: 360;
    font-size: 18px;
    color: var(--ink-2);
    margin: 0 0 56px;
    max-width: 60ch;
  }}
  .empty {{
    margin: 80px 0;
    text-align: center;
    color: var(--ink-3);
    font-family: var(--serif);
    font-style: italic;
    font-size: 18px;
  }}
  .empty code {{
    font-family: var(--mono);
    font-size: 14px;
    background: var(--paper);
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid var(--rule);
    color: var(--ink);
  }}
  table.ledger {{
    width: 100%;
    border-collapse: collapse;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--rule);
  }}
  table.ledger thead th {{
    text-align: left;
    padding: 14px 16px 14px 0;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
    font-weight: 500;
    border-bottom: 1px solid var(--rule);
    background: var(--paper);
  }}
  table.ledger tbody tr {{
    border-bottom: 1px solid var(--rule-2);
    transition: background .12s ease;
  }}
  table.ledger tbody tr:hover {{ background: var(--paper); }}
  table.ledger td {{
    padding: 16px 16px 16px 0;
    vertical-align: top;
  }}
  td.date {{ width: 110px; white-space: nowrap; }}
  td.date .d {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
  }}
  td.date .t {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-3);
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }}
  td.q {{
    font-family: var(--serif);
    font-weight: 400;
    font-size: 16px;
    line-height: 1.4;
    color: var(--ink);
    font-variation-settings: "opsz" 30;
    padding-right: 24px;
  }}
  td.v {{
    width: 160px;
    font-family: var(--serif);
    font-style: italic;
    font-size: 14px;
    color: var(--ink-2);
    white-space: nowrap;
  }}
  td.td {{
    width: 70px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-2);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  td.link {{ width: 86px; text-align: right; }}
  td.link a {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .08em;
    color: var(--accent);
    text-decoration: none;
    white-space: nowrap;
  }}
  td.link a:hover {{ text-decoration: underline; }}
  footer {{
    margin-top: 48px;
    padding-top: 22px;
    border-top: 1px solid var(--rule);
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .08em;
    color: var(--ink-3);
    display: flex;
    justify-content: space-between;
    text-transform: uppercase;
  }}
  footer b {{ color: var(--ink-2); font-weight: 500; }}
  @media (max-width: 720px) {{
    .page {{ padding: 32px 22px 64px; }}
    td.v, td.td {{ display: none; }}
    table.ledger thead th.v, table.ledger thead th.td {{ display: none; }}
  }}
</style>
</head>
<body>
<main class="page">
  <div class="mast">
    <span class="rule"></span>
    <span class="dot"></span>
    <b>TenthAI</b>
    <span class="sep">·</span>
    <span>Decision history</span>
  </div>
  <h1>Past <em>decisions</em>.</h1>
  <p class="sub">A ledger of every disagreement map you've generated. Newest first.</p>
  {body}
  <footer>
    <span><b>{count_label}</b></span>
    <span>updated {updated}</span>
  </footer>
</main>
</body>
</html>
"""


def write_index() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = list_records()
    index_path = REPORTS_DIR / "index.html"
    index_path.write_text(_index_html(records), encoding="utf-8")
    return index_path
