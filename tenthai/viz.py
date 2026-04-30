"""HTML viz: editorial-style disagreement map.

Layout: hero with question, polished plotly chart with guide rings,
tenth-man response as pull-quote, 9 frame cards with markdown rendering.
"""
import html as html_mod
import math
import re
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go


FRAME_GLYPHS = {
    "empirical": "01",
    "historical": "02",
    "first-principles": "03",
    "analogical": "04",
    "systemic": "05",
    "ethical": "06",
    "soft-contrarian": "07",
    "radical-optimist": "08",
    "pre-mortem": "09",
}


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML. Handles headers, bold, italic, hr, paragraphs."""
    text = html_mod.escape(text)
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if b.startswith("## "):
            out.append(f"<h3>{b[3:].strip()}</h3>")
        elif b.startswith("### "):
            out.append(f"<h4>{b[4:].strip()}</h4>")
        elif b.startswith("# "):
            out.append(f"<h2>{b[2:].strip()}</h2>")
        elif re.match(r"^-{3,}$", b):
            out.append("<hr>")
        else:
            b = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", b)
            b = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", b)
            b = b.replace("\n", "<br>")
            out.append(f"<p>{b}</p>")
    return "\n".join(out)


def _make_chart(question, frames, statuses, coords_2d, distances):
    """Editorial chart: guide rings, ink-on-paper aesthetic, halo on tenth-man."""
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]

    INK = "#1a1a1a"
    FRAME_COLOR = "#1e3a5f"
    TENTH_COLOR = "#c0392b"
    PAPER = "#faf7f2"
    GUIDE = "rgba(26,26,26,0.08)"
    GUIDE_TEXT = "rgba(26,26,26,0.35)"

    colors = [FRAME_COLOR] * 9 + [TENTH_COLOR]
    sizes = [14 + min(d, 1.5) * 10 for d in distances[:9]] + [22]

    fig = go.Figure()

    all_dist = [math.hypot(x, y) for x, y in zip(xs, ys)]
    max_r = max(all_dist) if all_dist else 1.0
    ring_radii = [max_r * 0.33, max_r * 0.66, max_r]

    for r in ring_radii:
        theta = [i / 64 * 2 * math.pi for i in range(65)]
        fig.add_trace(go.Scatter(
            x=[r * math.cos(t) for t in theta],
            y=[r * math.sin(t) for t in theta],
            mode="lines",
            line=dict(color=GUIDE, width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[max_r * 1.02], y=[0],
        mode="text",
        text=[f"d≈{max_r:.2f}"],
        textfont=dict(size=9, color=GUIDE_TEXT, family="JetBrains Mono, monospace"),
        textposition="middle right",
        hoverinfo="skip",
        showlegend=False,
    ))

    for i in range(10):
        fig.add_trace(go.Scatter(
            x=[0, xs[i]],
            y=[0, ys[i]],
            mode="lines",
            line=dict(color="rgba(26,26,26,0.06)", width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(symbol="diamond-open", size=10, color=INK, line=dict(width=1.2)),
        hovertext="centroide de los 9",
        hoverinfo="text",
        showlegend=False,
    ))

    # tenth-man halo
    fig.add_trace(go.Scatter(
        x=[xs[9]], y=[ys[9]],
        mode="markers",
        marker=dict(size=42, color=TENTH_COLOR, opacity=0.12),
        hoverinfo="skip",
        showlegend=False,
    ))

    hover_texts = [
        f"<b>{frame}</b><br>distance: {dist:.3f}<br><i>{status}</i>"
        for frame, status, dist in zip(frames, statuses, distances)
    ]

    label_text = [f"{FRAME_GLYPHS.get(frames[i], '·')} {frames[i]}" for i in range(9)] + ["10 · tenth-man"]

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=2, color=PAPER),
            opacity=0.95,
        ),
        text=label_text,
        textposition="top center",
        textfont=dict(size=11, color=INK, family="JetBrains Mono, ui-monospace, monospace"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        cliponaxis=False,
    ))

    pad = max_r * 0.35
    x_range = [-max_r - pad, max_r + pad]
    y_range = [-max_r - pad, max_r + pad]

    fig.update_layout(
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=dict(family="Inter, system-ui, sans-serif"),
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=INK,
            font=dict(family="Inter, sans-serif", size=12),
        ),
    )

    return fig


def render(question, results, coords_2d, distances, provider, model, cost_estimate_clp):
    """Render the disagreement map to HTML. Returns absolute path. Auto-opens."""
    frames = [r[0] for r in results]
    responses = [r[1] for r in results]
    statuses = [r[2] for r in results]

    fig = _make_chart(question, frames, statuses, coords_2d, distances)
    chart_html = fig.to_html(
        include_plotlyjs="inline",
        full_html=False,
        div_id="tenthai-chart",
        config={"displaylogo": False, "responsive": True, "displayModeBar": False},
    )

    tenth_response_html = _md_to_html(responses[9])
    tenth_distance = distances[9]
    max_frame_distance = max(distances[:9])
    fragility = (
        "consenso frágil — el disidente vive en otro mundo"
        if tenth_distance > 2 * max_frame_distance
        else "marcos ya dispersos — no había consenso fuerte"
    )

    frame_cards = []
    for i in range(9):
        frame = frames[i]
        glyph = FRAME_GLYPHS.get(frame, "·")
        resp = _md_to_html(responses[i])
        status = statuses[i]
        dist = distances[i]
        status_badge = "" if status == "ok" else '<span class="badge-failed">FAILED</span>'
        frame_cards.append(f"""
        <details class="frame-card">
          <summary>
            <span class="frame-glyph">{glyph}</span>
            <span class="frame-name">{html_mod.escape(frame)}</span>
            {status_badge}
            <span class="frame-distance">d {dist:.3f}</span>
          </summary>
          <div class="frame-body">{resp}</div>
        </details>
        """)
    frame_cards_html = "\n".join(frame_cards)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    question_safe = html_mod.escape(question)

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TenthAI · {question_safe[:80]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #1a1a1a;
    --ink-soft: #2c2c2c;
    --muted: #6b6660;
    --muted-soft: #999591;
    --paper: #faf7f2;
    --paper-warm: #f5f0e6;
    --surface: #ffffff;
    --border: rgba(26, 26, 26, 0.10);
    --border-soft: rgba(26, 26, 26, 0.06);
    --tenth: #c0392b;
    --tenth-soft: #fdf2f0;
    --frame: #1e3a5f;
    --serif: "Fraunces", "Source Serif 4", Georgia, serif;
    --serif-text: "Source Serif 4", Georgia, serif;
    --sans: "Inter", system-ui, -apple-system, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ background: var(--paper); }}
  body {{
    font-family: var(--sans);
    color: var(--ink);
    margin: 0;
    padding: 0;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }}
  .container {{
    max-width: 880px;
    margin: 0 auto;
    padding: 64px 32px 96px;
  }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .hero {{
    margin-bottom: 56px;
    animation: fadeUp 0.5s ease-out;
  }}
  .eyebrow {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .eyebrow::before {{
    content: "";
    width: 24px; height: 1px;
    background: var(--ink);
    display: inline-block;
  }}
  .eyebrow .brand-dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--tenth);
    display: inline-block;
  }}
  h1.brand-line {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: clamp(28px, 4vw, 40px);
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin: 0 0 24px;
    color: var(--ink);
  }}
  h1.brand-line em {{
    font-style: italic;
    color: var(--tenth);
    font-weight: 500;
  }}
  blockquote.question {{
    font-family: var(--serif-text);
    font-size: 19px;
    line-height: 1.5;
    color: var(--ink-soft);
    border-left: 3px solid var(--ink);
    padding: 4px 0 4px 20px;
    margin: 0 0 28px;
    font-style: italic;
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 24px;
  }}
  .meta-chip {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
  }}
  .meta-chip .label {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 4px;
  }}
  .meta-chip .value {{
    font-family: var(--mono);
    font-size: 14px;
    color: var(--ink);
    font-weight: 500;
  }}
  .meta-chip.verdict .value {{
    font-family: var(--serif-text);
    font-size: 13px;
    font-style: italic;
    line-height: 1.35;
  }}
  .chart-wrap {{
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin: 0 0 64px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(26,26,26,0.03), 0 8px 24px -12px rgba(26,26,26,0.08);
    animation: fadeUp 0.6s ease-out 0.1s both;
  }}
  .chart-caption {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    padding: 14px 16px;
    border-top: 1px solid var(--border-soft);
    background: rgba(255,255,255,0.4);
  }}
  section {{ margin-bottom: 56px; }}
  .section-header {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .section-num {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 0.10em;
  }}
  .section-title {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 22px;
    margin: 0;
    letter-spacing: -0.01em;
    flex: 1;
  }}
  .section-title em {{
    font-style: italic;
    font-weight: 500;
    color: var(--tenth);
  }}
  .tenth-man {{
    background: var(--tenth-soft);
    border-left: 4px solid var(--tenth);
    padding: 32px 36px;
    border-radius: 4px;
    font-family: var(--serif-text);
    font-size: 16.5px;
    line-height: 1.7;
    color: var(--ink-soft);
    animation: fadeUp 0.6s ease-out 0.2s both;
  }}
  .tenth-man h3 {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 22px;
    margin: 28px 0 14px;
    color: var(--ink);
    letter-spacing: -0.01em;
  }}
  .tenth-man h3:first-child {{ margin-top: 0; }}
  .tenth-man h4 {{
    font-family: var(--sans);
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin: 24px 0 10px;
  }}
  .tenth-man p {{ margin: 0 0 14px; }}
  .tenth-man strong {{ color: var(--ink); font-weight: 600; }}
  .tenth-man em {{ color: var(--tenth); font-style: italic; }}
  .tenth-man hr {{
    border: none;
    border-top: 1px dashed rgba(192, 57, 43, 0.3);
    margin: 24px 0;
  }}
  .tenth-man .distance-stamp {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px dashed rgba(192, 57, 43, 0.3);
    text-transform: uppercase;
    letter-spacing: 0.10em;
  }}
  .frame-card {{
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 10px;
    background: var(--surface);
    transition: border-color 0.15s, box-shadow 0.15s;
  }}
  .frame-card:hover {{ border-color: rgba(26,26,26,0.20); }}
  .frame-card[open] {{
    border-color: var(--frame);
    box-shadow: 0 1px 2px rgba(30,58,95,0.04), 0 4px 12px -4px rgba(30,58,95,0.08);
  }}
  .frame-card summary {{
    padding: 14px 18px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 14px;
    user-select: none;
    list-style: none;
  }}
  .frame-card summary::-webkit-details-marker {{ display: none; }}
  .frame-glyph {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    background: var(--paper-warm);
    padding: 4px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }}
  .frame-card[open] .frame-glyph {{
    background: var(--frame);
    color: var(--paper);
  }}
  .frame-name {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 16px;
    flex: 1;
    letter-spacing: -0.005em;
  }}
  .frame-distance {{
    font-family: var(--mono);
    color: var(--muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.05em;
  }}
  .frame-card summary::after {{
    content: "+";
    font-family: var(--mono);
    color: var(--muted);
    font-size: 16px;
    transition: transform 0.2s;
    display: inline-block;
  }}
  .frame-card[open] summary::after {{ transform: rotate(45deg); }}
  .badge-failed {{
    background: #fee2e2;
    color: #991b1b;
    font-family: var(--mono);
    font-size: 10px;
    padding: 3px 7px;
    border-radius: 3px;
    font-weight: 500;
    letter-spacing: 0.08em;
  }}
  .frame-body {{
    padding: 4px 22px 22px 22px;
    font-family: var(--serif-text);
    font-size: 15px;
    line-height: 1.65;
    color: var(--ink-soft);
    border-top: 1px solid var(--border-soft);
    margin-top: 0;
  }}
  .frame-body h3 {{
    font-family: var(--serif);
    font-size: 17px;
    margin: 18px 0 8px;
    color: var(--ink);
  }}
  .frame-body h4 {{
    font-family: var(--sans);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin: 16px 0 6px;
  }}
  .frame-body p {{ margin: 0 0 10px; }}
  .frame-body strong {{ color: var(--ink); }}
  .frame-body hr {{
    border: none;
    border-top: 1px dashed var(--border);
    margin: 16px 0;
  }}
  footer {{
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 64px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-transform: uppercase;
    line-height: 1.8;
  }}
  footer .signature {{ color: var(--ink); font-weight: 500; }}
</style>
</head>
<body>
<div class="container">
  <header class="hero">
    <p class="eyebrow"><span class="brand-dot"></span>TenthAI · disagreement map</p>
    <h1 class="brand-line">Nine advisors aligned. <em>The tenth must dissent.</em></h1>
    <blockquote class="question">{question_safe}</blockquote>
    <div class="meta-grid">
      <div class="meta-chip">
        <p class="label">Tenth-man dist</p>
        <p class="value">{tenth_distance:.3f}</p>
      </div>
      <div class="meta-chip">
        <p class="label">Max frame dist</p>
        <p class="value">{max_frame_distance:.3f}</p>
      </div>
      <div class="meta-chip verdict">
        <p class="label">Veredicto</p>
        <p class="value">{fragility}</p>
      </div>
    </div>
  </header>

  <div class="chart-wrap">
    {chart_html}
    <p class="chart-caption">10 voces · proyección MDS · distancia al centroide de los 9</p>
  </div>

  <section>
    <div class="section-header">
      <span class="section-num">10</span>
      <h2 class="section-title">El <em>décimo hombre</em></h2>
      <span class="frame-distance">d {tenth_distance:.3f}</span>
    </div>
    <div class="tenth-man">
      {tenth_response_html}
      <div class="distance-stamp">disenso steel-man · d {tenth_distance:.3f} vs centroide</div>
    </div>
  </section>

  <section>
    <div class="section-header">
      <span class="section-num">01—09</span>
      <h2 class="section-title">Los nueve marcos</h2>
    </div>
    {frame_cards_html}
  </section>

  <footer>
    <span class="signature">TenthAI</span> · classical MDS over cosine distance · preserves pairwise distances<br>
    {timestamp} · embed {provider}/{model} · ~CLP {cost_estimate_clp:.0f}
  </footer>
</div>
</body>
</html>"""

    out_dir = Path(tempfile.gettempdir())
    timestamp_file = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_q = "".join(c if c.isalnum() else "_" for c in question[:40])
    path = out_dir / f"tenthai_{timestamp_file}_{safe_q}.html"
    path.write_text(page, encoding="utf-8")

    try:
        webbrowser.open(f"file://{path.absolute()}")
    except Exception:
        pass

    return str(path)
