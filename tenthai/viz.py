"""HTML viz: plotly scatter (markers + short labels) + static response panel + browser auto-open.

Layout: chart on top (just dots + frame names), tenth-man response prominent below,
9 frames as collapsible cards. No long text inside plotly — that broke rendering.
"""
import html as html_mod
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go


def _make_chart(question, frames, statuses, coords_2d, distances):
    """Build plotly figure. Markers + short frame labels only. No response text inside."""
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]

    colors = ["#3b82f6"] * 9 + ["#ef4444"]
    sizes = [16 + min(d, 1.5) * 12 for d in distances[:9]] + [24]

    hover_texts = [
        f"<b>{frame}</b> ({status})<br>distance: {dist:.3f}"
        for frame, status, dist in zip(frames, statuses, distances)
    ]

    fig = go.Figure()

    for i in range(10):
        fig.add_trace(go.Scatter(
            x=[0, xs[i]],
            y=[0, ys[i]],
            mode="lines",
            line=dict(color="#e5e7eb", width=0.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1.5, color="#1f2937"),
        ),
        text=frames,
        textposition="top center",
        textfont=dict(size=11, color="#374151"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        cliponaxis=False,
    ))

    all_x = xs + [0]
    all_y = ys + [0]
    pad = 0.3
    x_range = [min(all_x) - pad, max(all_x) + pad]
    y_range = [min(all_y) - pad, max(all_y) + pad]

    fig.update_layout(
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=True,
            zerolinecolor="#e5e7eb",
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=True,
            zerolinecolor="#e5e7eb",
            showticklabels=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, sans-serif"),
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


def render(question, results, coords_2d, distances, provider, model, cost_estimate_clp):
    """Render the disagreement map to HTML. Returns absolute path. Auto-opens.

    results: list of (frame, response, status) tuples, length 10. Last is tenth-man.
    """
    frames = [r[0] for r in results]
    responses = [r[1] for r in results]
    statuses = [r[2] for r in results]

    fig = _make_chart(question, frames, statuses, coords_2d, distances)
    chart_html = fig.to_html(
        include_plotlyjs="inline",
        full_html=False,
        div_id="tenthai-chart",
        config={"displaylogo": False, "responsive": True},
    )

    tenth_response_html = html_mod.escape(responses[9]).replace("\n", "<br>")
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
        resp = html_mod.escape(responses[i]).replace("\n", "<br>")
        status = statuses[i]
        dist = distances[i]
        status_badge = "" if status == "ok" else f'<span class="badge-failed">FAILED</span>'
        frame_cards.append(f"""
        <details class="frame-card">
          <summary>
            <span class="frame-name">{html_mod.escape(frame)}</span>
            {status_badge}
            <span class="frame-distance">d = {dist:.3f}</span>
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
<style>
  :root {{
    --fg: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --bg: #ffffff;
    --bg-soft: #f9fafb;
    --tenth: #ef4444;
    --frame: #3b82f6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 32px 24px 80px;
    max-width: 980px;
    margin: 0 auto;
    line-height: 1.55;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ margin: 0 0 8px; font-size: 22px; }}
  h1 .brand {{ color: var(--tenth); }}
  .question {{ color: var(--muted); font-size: 15px; margin: 0; }}
  .meta {{
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: var(--muted);
    margin-top: 12px;
    flex-wrap: wrap;
  }}
  .meta strong {{ color: var(--fg); }}
  .chart-wrap {{
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 32px;
    overflow: hidden;
  }}
  section {{ margin-bottom: 32px; }}
  h2 {{
    font-size: 16px;
    margin: 0 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  h2 .dot {{
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
  }}
  .tenth-man {{
    background: #fef2f2;
    border-left: 4px solid var(--tenth);
    padding: 16px 20px;
    border-radius: 4px;
    font-size: 15px;
  }}
  .tenth-man .distance {{
    font-size: 12px;
    color: var(--muted);
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px dashed #fca5a5;
  }}
  .frame-card {{
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 8px;
    background: var(--bg-soft);
  }}
  .frame-card[open] {{ background: var(--bg); }}
  .frame-card summary {{
    padding: 10px 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    user-select: none;
    list-style: none;
  }}
  .frame-card summary::-webkit-details-marker {{ display: none; }}
  .frame-card summary::before {{
    content: "▸";
    color: var(--muted);
    transition: transform 0.15s;
  }}
  .frame-card[open] summary::before {{ content: "▾"; }}
  .frame-name {{ font-weight: 600; flex: 1; }}
  .frame-distance {{ color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
  .badge-failed {{
    background: #fee2e2;
    color: #991b1b;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }}
  .frame-body {{
    padding: 0 14px 14px;
    font-size: 14px;
    color: #374151;
  }}
  footer {{
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-top: 40px;
    font-size: 11px;
    color: var(--muted);
    font-style: italic;
  }}
</style>
</head>
<body>
  <header>
    <h1><span class="brand">●</span> TenthAI</h1>
    <p class="question">{question_safe}</p>
    <div class="meta">
      <span><strong>Tenth-man distance:</strong> {tenth_distance:.3f}</span>
      <span><strong>Max frame distance:</strong> {max_frame_distance:.3f}</span>
      <span><strong>Veredicto:</strong> {fragility}</span>
    </div>
  </header>

  <div class="chart-wrap">{chart_html}</div>

  <section>
    <h2><span class="dot" style="background:{('var(--tenth)')}"></span>Décimo hombre — disenso steel-man</h2>
    <div class="tenth-man">
      {tenth_response_html}
      <div class="distance">d = {tenth_distance:.3f} (vs centroide de los 9)</div>
    </div>
  </section>

  <section>
    <h2><span class="dot" style="background:var(--frame)"></span>9 marcos cognitivos</h2>
    {frame_cards_html}
  </section>

  <footer>
    Mapa basado en classical MDS sobre cosine distance · preserva distancias entre pares fielmente.<br>
    {timestamp} · embed: {provider}/{model} · costo estimado: ~CLP {cost_estimate_clp:.0f}
  </footer>
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
