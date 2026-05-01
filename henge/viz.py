"""HTML viz: editorial-modernist disagreement report.

Design language:
- Fraunces (serif display) + Inter Tight (UI) + JetBrains Mono (data)
- Warm paper background (oklch), carbon ink, single terracotta accent for tenth-man
- Sober ink-blue for the consensus nine
- Generous whitespace, fine rules, data as first-class citizen

Layout: masthead → headline → question → stats strip → MDS map → consensus →
9 frames list (sorted by distance, closest open by default) → tenth-man feature →
colophon. Order honors the user's reading flow: see what the 9 agree on first,
then individual conclusions, then the dissent as punchline at the end.
"""
import html as html_mod
import math
import re
from datetime import datetime


# Calibration for consensus_verdict() — empirical defaults for voyage-3-large /
# text-embedding-3-small over Spanish reasoning text. Tighten or loosen if you
# observe systematic miscalibration over many runs.
TIGHT_THRESHOLD = 0.15  # max_frame below this = the 9 are clustered
DISSENT_RATIO = 1.6     # tenth_distance / max_frame above this = tenth meaningfully separated


def consensus_verdict(tenth_distance: float, max_frame_distance: float) -> dict:
    """Three-state classification of the consensus shape.

    - aligned-stable:  9 advisors tight, tenth's dissent is moderate. The consensus holds.
    - aligned-fragile: 9 tight, but the tenth is far enough to coherently break it.
    - divided:         the 9 themselves are spread — there was no strong consensus to break.

    The previous binary model conflated "aligned + moderate dissent" with "divided",
    which mislabeled tightly-clustered 9s as divided whenever the tenth wasn't
    extreme. This helper is the single source of truth for the verdict text.
    """
    tight_nine = max_frame_distance < TIGHT_THRESHOLD
    if not tight_nine:
        return {
            "state": "divided",
            "label_short": "divided",
            "verdict": "Consejeros divididos — no había consenso fuerte para empezar.",
        }
    if tenth_distance > DISSENT_RATIO * max_frame_distance:
        return {
            "state": "aligned-fragile",
            "label_short": "fragile consensus",
            "verdict": "Consenso fuerte pero frágil — el disidente lo rompe coherentemente.",
        }
    return {
        "state": "aligned-stable",
        "label_short": "aligned",
        "verdict": "Consejeros alineados — el disenso suena pero el consenso aguanta.",
    }


FRAME_INDEX = {
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
    """Minimal markdown → HTML. Headers, bold, italic, hr, paragraphs, blockquote."""
    text = html_mod.escape(text)
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if b.startswith("> "):
            inner = b[2:].strip()
            inner = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", inner)
            inner = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", inner)
            out.append(f"<p class=\"pull\">{inner}</p>")
        elif b.startswith("## "):
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


def _split_consensus_title(text: str):
    """Pull a leading ``# Title`` line out of the consensus output.

    Returns (title_or_none, body). If the first non-empty line starts with
    ``# `` (markdown h1), use it as the title and strip from the body.
    Otherwise return (None, text) so the caller renders a fallback title.
    """
    if not text:
        return None, ""
    stripped = text.lstrip()
    lines = stripped.split("\n", 1)
    first = lines[0].strip()
    if first.startswith("# ") and not first.startswith("## "):
        title = first[2:].strip()
        body = lines[1].lstrip() if len(lines) > 1 else ""
        return title, body
    return None, text


def _split_failure_modes(text: str):
    """Pull a [FAILURE_MODES]...[/FAILURE_MODES] block out of the tenth-man response.

    Returns (main_body, modes) where modes is a list of (title, body) tuples
    parsed from ``### Title\\nbody`` items. If the block is absent or malformed,
    returns (text, []) so the caller falls back to plain prose rendering.
    """
    match = re.search(
        r"\[FAILURE_MODES\](.+?)\[/FAILURE_MODES\]",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return text.strip(), []
    inner = match.group(1).strip()
    main = (text[:match.start()] + text[match.end():]).strip()
    modes = []
    for chunk in re.split(r"\n###\s+", "\n" + inner):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.split("\n", 1)
        title = lines[0].lstrip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if title and body:
            modes.append((title, body))
    if len(modes) != 3:
        # Don't render a partial grid — fall back to plain prose.
        return text.strip(), []
    return main, modes


def _style_section_markers(html: str) -> str:
    """Wrap ``§N`` prefix in tenth-body h3 headings with a terracotta accent span.

    Pattern in the rendered HTML: ``<h3>§1 Hechos que acepto</h3>``. We wrap
    the §N portion in a dedicated span the CSS picks up.
    """
    return re.sub(
        r"<h3>(§\s*\d+)\s+(.+?)</h3>",
        r'<h3><span class="ord">\1</span>\2</h3>',
        html,
    )


def _extract_conclusion(text: str, max_chars: int = 360) -> str:
    """Last paragraph, assumed conclusion per prompt structure."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return text[:max_chars]
    last = paragraphs[-1]
    if len(last) < 100 and len(paragraphs) >= 2:
        last = paragraphs[-2] + "\n\n" + last
    if len(last) > max_chars:
        truncated = last[:max_chars]
        last_period = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
        if last_period > max_chars * 0.5:
            return truncated[: last_period + 1]
        return truncated.rsplit(" ", 1)[0] + "..."
    return last


def _map_to_svg(coords_2d):
    """Map 10 MDS coords into SVG viewBox 1000×700, centered at (500, 350).

    Scale so the max extent fits within radius ~240 (matches design's outer ring).
    Returns list of (x_svg, y_svg) tuples, length 10.
    """
    cx, cy = 500, 350
    max_radius = 230

    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]
    max_extent = max(
        max(abs(x) for x in xs) if xs else 0.001,
        max(abs(y) for y in ys) if ys else 0.001,
        0.001,
    )
    scale = max_radius / max_extent

    return [(cx + x * scale, cy + y * scale) for x, y in zip(xs, ys)]


def _build_map_svg(coords_2d, frames, distances, statuses, max_frame_dist, min_frame_dist):
    """Generate the full MDS map SVG with rings, crosshair, centroid, frames, tenth-man."""
    points = _map_to_svg(coords_2d)
    cx, cy = 500, 350

    parts = [
        '<svg class="map" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid meet" aria-hidden="true">',
        '<defs>',
        '<radialGradient id="halo" cx="50%" cy="50%" r="50%">',
        '<stop offset="0%" stop-color="oklch(56% 0.165 32 / .22)"/>',
        '<stop offset="60%" stop-color="oklch(56% 0.165 32 / .04)"/>',
        '<stop offset="100%" stop-color="oklch(56% 0.165 32 / 0)"/>',
        '</radialGradient>',
        '</defs>',
        # Concentric rings (4 levels)
        '<g stroke="oklch(86% 0.012 70)" stroke-width="0.6" fill="none" opacity=".7">',
        f'<circle cx="{cx}" cy="{cy}" r="60"/>',
        f'<circle cx="{cx}" cy="{cy}" r="120"/>',
        f'<circle cx="{cx}" cy="{cy}" r="180" stroke-dasharray="2 4"/>',
        f'<circle cx="{cx}" cy="{cy}" r="240" stroke-dasharray="2 4"/>',
        '</g>',
        # Crosshair
        '<g stroke="oklch(82% 0.012 70)" stroke-width="0.6">',
        '<line x1="100" y1="350" x2="900" y2="350"/>',
        '<line x1="500" y1="60" x2="500" y2="640"/>',
        '</g>',
        # Centroid label + diamond marker
        '<g font-family="JetBrains Mono, monospace" font-size="10" fill="oklch(56% 0.012 60)" letter-spacing="1">',
        '<text x="510" y="346">CENTROIDE</text>',
        f'<rect x="{cx - 4}" y="{cy - 4}" width="8" height="8" fill="none" stroke="oklch(56% 0.012 60)" stroke-width="1" transform="rotate(45 {cx} {cy})"/>',
        '</g>',
    ]

    # Distance lines (9 frames in blue, tenth in accent dashed)
    parts.append('<g stroke="oklch(34% 0.085 250 / .35)" stroke-width="0.7">')
    for i in range(9):
        x, y = points[i]
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>')
    parts.append('</g>')
    tx, ty = points[9]
    parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" '
        f'stroke="oklch(56% 0.165 32 / .55)" stroke-width="1.1" stroke-dasharray="3 3"/>'
    )

    # Nine frame nodes
    parts.append('<g class="nodes" font-family="JetBrains Mono, monospace" font-size="11" fill="oklch(20% 0.012 60)">')
    for i in range(9):
        x, y = points[i]
        frame = frames[i]
        idx = FRAME_INDEX.get(frame, f"{i+1:02d}")
        dist = distances[i]
        # Label position: anchor right when point is in right half, left otherwise
        if x >= cx:
            anchor = "start"
            lx = x + 17
        else:
            anchor = "end"
            lx = x - 17
        # Highlight closest + farthest with thicker ring
        is_closest = abs(dist - min_frame_dist) < 1e-6
        is_farthest = abs(dist - max_frame_dist) < 1e-6
        ring_opacity = ".35" if (is_closest or is_farthest) else ".25"
        ring_radius = 15 if is_closest else 13
        marker_radius = 9 if is_closest else 8
        # Tag suffix for closest/farthest
        suffix = ""
        if is_closest:
            suffix = " · más cercano"
        elif is_farthest:
            suffix = " · más lejano"
        parts.append('<g>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{marker_radius}" fill="oklch(34% 0.085 250)"/>')
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{ring_radius}" fill="none" '
            f'stroke="oklch(34% 0.085 250 / {ring_opacity})" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y - 4:.1f}" font-size="11" text-anchor="{anchor}">{idx}</text>'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y + 10:.1f}" font-family="Inter Tight, sans-serif" '
            f'font-weight="500" font-size="13" text-anchor="{anchor}">{html_mod.escape(frame)}</text>'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y + 25:.1f}" font-size="9.5" fill="oklch(56% 0.012 60)" '
            f'text-anchor="{anchor}">d&#160;{dist:.3f}{suffix}</text>'
        )
        parts.append('</g>')
    parts.append('</g>')

    # Tenth-man with halo
    tx, ty = points[9]
    tenth_dist = distances[9]
    parts.append('<g>')
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="42" fill="url(#halo)"/>')
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="11" fill="oklch(56% 0.165 32)"/>')
    parts.append(
        f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="17" fill="none" '
        f'stroke="oklch(56% 0.165 32 / .35)" stroke-width="3"/>'
    )
    label_y_offset = -37 if ty > cy - 50 else 50
    parts.append(
        f'<text x="{tx:.1f}" y="{ty + label_y_offset:.1f}" text-anchor="middle" '
        f'font-family="Inter Tight, sans-serif" font-weight="600" font-size="13" '
        f'fill="oklch(56% 0.165 32)" letter-spacing="2">10 · TENTH-MAN</text>'
    )
    parts.append(
        f'<text x="{tx:.1f}" y="{ty + label_y_offset + 15:.1f}" text-anchor="middle" '
        f'font-size="10" fill="oklch(56% 0.165 32)">d&#160;{tenth_dist:.3f} · disenso steel-man</text>'
    )
    parts.append('</g>')

    # Axis labels
    parts.append(
        '<g font-family="JetBrains Mono, monospace" font-size="9.5" '
        'fill="oklch(56% 0.012 60)" letter-spacing="2">'
    )
    parts.append('<text x="100" y="345">MDS-1</text>')
    parts.append('<text x="892" y="345" text-anchor="end">→</text>')
    parts.append('<text x="500" y="55" text-anchor="middle">MDS-2</text>')
    parts.append('</g>')

    parts.append('</svg>')
    return "\n".join(parts)


def _build_frame_card(frame, response, status, distance, max_dist, idx_str, is_open=False):
    """Build a frame-list article. Open by default for the closest frame."""
    body = _md_to_html(response)
    bar_pct = min(100, (distance / max_dist) * 100) if max_dist > 0 else 0
    status_tag = "OK" if status == "ok" else "FAILED"
    open_class = " open" if is_open else ""
    bar_color_var = "var(--accent)" if is_open else "var(--nine)"
    return f"""
    <article class="frame{open_class}" data-frame="{html_mod.escape(frame)}">
      <div class="frame-row">
        <span class="f-idx">{idx_str}</span>
        <span class="f-name">{html_mod.escape(frame)}</span>
        <span class="f-tag"><b>{status_tag}</b></span>
        <span class="f-bar"><i style="width:{bar_pct:.0f}%; background:{bar_color_var};"></i></span>
        <span class="f-d">d&#160;<b>{distance:.3f}</b></span>
        <span class="f-caret">›</span>
      </div>
      <div class="f-body">
        {body}
      </div>
    </article>
    """


def render(question, results, coords_2d, distances, provider, model, cost_estimate_clp, consensus=None):
    """Render the editorial disagreement report. Returns the full HTML as a string.

    Persistence and browser-open are handled by the caller (server.py orchestrates
    storage.write_record + webbrowser.open). Keeping render() pure makes it easy
    to test, embed, and post-process (the demo pipes through an English-chrome
    rewrite before writing to disk).

    Layout order: masthead → headline → question → stats → map → consensus →
    9 frames (sorted by distance, closest open) → tenth-man feature → colophon.
    """
    frames = [r[0] for r in results]
    responses = [r[1] for r in results]
    statuses = [r[2] for r in results]

    tenth_distance = distances[9]
    frame_distances = distances[:9]
    max_frame_distance = max(frame_distances)
    min_frame_distance = min(frame_distances)
    most_divergent_idx = frame_distances.index(max_frame_distance)
    closest_frame_idx = frame_distances.index(min_frame_distance)
    most_divergent_name = frames[most_divergent_idx]

    fragility = consensus_verdict(tenth_distance, max_frame_distance)["verdict"]

    # Map SVG (uses real MDS coords)
    map_svg = _build_map_svg(
        coords_2d=coords_2d,
        frames=frames,
        distances=distances,
        statuses=statuses,
        max_frame_dist=max_frame_distance,
        min_frame_dist=min_frame_distance,
    )

    # Frames sorted by distance ASC; closest is default open
    frame_order = sorted(range(9), key=lambda i: frame_distances[i])
    frame_cards_html = "\n".join(
        _build_frame_card(
            frame=frames[i],
            response=responses[i],
            status=statuses[i],
            distance=frame_distances[i],
            max_dist=max_frame_distance,
            idx_str=FRAME_INDEX.get(frames[i], f"{i+1:02d}"),
            is_open=(i == closest_frame_idx),
        )
        for i in frame_order
    )

    tenth_main, tenth_modes = _split_failure_modes(responses[9])
    tenth_response_html = _style_section_markers(_md_to_html(tenth_main))
    if tenth_modes:
        mode_cards = []
        for i, (title, body) in enumerate(tenth_modes, start=1):
            mode_cards.append(
                f'<div class="mode">'
                f'<div class="ord">Modo de fallo · {i}</div>'
                f'<h4>{html_mod.escape(title)}</h4>'
                f'<p>{html_mod.escape(body)}</p>'
                f'</div>'
            )
        tenth_modes_html = '<div class="modes">' + "".join(mode_cards) + '</div>'
    else:
        tenth_modes_html = ""

    # Consensus block (optional). Editorial 2-column essay treatment:
    # extracted h1 title + verdict-aware metric in the header card.
    consensus_block_html = ""
    if consensus:
        consensus_title, consensus_body_md = _split_consensus_title(consensus)
        if not consensus_title:
            consensus_title = "Lo que los nueve coinciden"
        consensus_html = _md_to_html(consensus_body_md)
        verdict = consensus_verdict(tenth_distance, max_frame_distance)
        verdict_label_upper = verdict["label_short"].upper()
        consensus_block_html = f"""
  <div class="sec">
    <span class="glyph">∑</span>
    <h2>El <em>consenso</em> de los nueve</h2>
    <span class="rule"></span>
    <span class="aside">qué creen <b>en común</b> los nueve</span>
  </div>

  <article class="consensus">
    <header class="consensus-top">
      <div>
        <div class="lbl">Σ · 9 consejeros · {html_mod.escape(verdict_label_upper)}</div>
        <h3>{html_mod.escape(consensus_title)}</h3>
      </div>
      <div class="d"><b>{max_frame_distance:.3f}</b>max distancia · vs centroide</div>
    </header>
    <div class="consensus-body">
      {consensus_html}
    </div>
  </article>
"""

    timestamp = datetime.now().strftime("%Y·%m·%d · %H:%M CLT")
    timestamp_short = datetime.now().strftime("%Y·%m·%d")
    report_id = datetime.now().strftime("%H%M")
    question_safe = html_mod.escape(question)

    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Henge · {question_safe[:60]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,0..100;1,9..144,300..700,0..100&family=Inter+Tight:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --bg:        oklch(96.5% 0.012 78);
    --paper:     oklch(98.5% 0.008 78);
    --paper-2:   oklch(94.5% 0.014 78);
    --ink:       oklch(20% 0.012 60);
    --ink-2:     oklch(38% 0.012 60);
    --ink-3:     oklch(56% 0.012 60);
    --ink-4:     oklch(72% 0.010 70);
    --rule:      oklch(86% 0.012 70);
    --rule-2:    oklch(91% 0.012 70);

    --accent:      oklch(56% 0.165 32);
    --accent-2:    oklch(48% 0.150 30);
    --accent-soft: oklch(92% 0.045 32);

    --nine:        oklch(34% 0.085 250);
    --nine-2:      oklch(42% 0.080 250);
    --nine-soft:   oklch(93% 0.025 250);

    --serif: 'Fraunces', 'Iowan Old Style', Georgia, serif;
    --sans:  'Inter Tight', system-ui, sans-serif;
    --mono:  'JetBrains Mono', ui-monospace, monospace;
  }}
  *{{ box-sizing: border-box; }}
  html, body{{ margin:0; padding:0; }}
  body{{
    font-family: var(--sans);
    background: var(--bg);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    line-height: 1.55;
    font-feature-settings: "ss01","cv11";
  }}
  ::selection{{ background: var(--accent-soft); color: var(--ink); }}

  .page{{ max-width: 1180px; margin: 0 auto; padding: 56px 56px 96px; }}

  .mast{{
    display:flex; align-items:center; gap:14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .mast .rule{{ width: 28px; height:1px; background: var(--ink-3); }}
  .mast .dot{{ width:7px; height:7px; border-radius:999px; background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);}}
  .mast b{{ color: var(--ink); font-weight: 500; letter-spacing: .14em;}}
  .mast .sep{{ color: var(--ink-4); }}

  .head{{
    margin: 28px 0 0;
    font-family: var(--serif);
    font-weight: 380;
    font-size: clamp(40px, 5.4vw, 72px);
    line-height: 1.02;
    letter-spacing: -0.025em;
    color: var(--ink);
    font-variation-settings: "SOFT" 30, "opsz" 144;
    text-wrap: balance;
    max-width: 16ch;
  }}
  .head em{{
    font-style: italic;
    color: var(--accent);
    font-weight: 360;
    font-variation-settings: "SOFT" 60, "opsz" 144;
  }}

  .question{{
    margin: 28px 0 0;
    padding: 4px 0 4px 18px;
    border-left: 2px solid var(--ink-4);
    font-family: var(--serif);
    font-style: italic;
    font-weight: 360;
    font-size: 20px;
    line-height: 1.4;
    color: var(--ink-2);
    max-width: 64ch;
    font-variation-settings: "opsz" 60;
    text-wrap: pretty;
  }}

  .strip{{
    display:grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    margin: 40px 0 56px;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--rule);
  }}
  .stat{{
    padding: 18px 24px 18px 0;
    border-right: 1px solid var(--rule);
    display:flex; flex-direction:column; gap:8px;
  }}
  .stat:nth-child(2){{ padding-left: 24px; }}
  .stat:nth-child(3){{ padding-left: 24px; border-right: none;}}
  .stat .lbl{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .stat .val{{
    font-family: var(--serif);
    font-weight: 400;
    font-size: 26px;
    letter-spacing: -0.018em;
    color: var(--ink);
    line-height: 1.1;
  }}
  .stat .val .num{{
    font-family: var(--mono);
    font-weight: 500;
    font-size: 13px;
    color: var(--ink-3);
    margin-left: 8px;
    letter-spacing: 0;
  }}
  .stat.accent .val{{ color: var(--accent); }}
  .stat .verdict{{
    font-family: var(--serif);
    font-style: italic;
    font-weight: 380;
    font-size: 16px;
    line-height: 1.4;
    color: var(--ink-2);
  }}

  .map-card{{
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 2px;
    overflow: hidden;
    position: relative;
  }}
  .map-head{{
    display:flex; justify-content: space-between; align-items: baseline;
    padding: 18px 24px;
    border-bottom: 1px solid var(--rule-2);
  }}
  .map-head h3{{
    margin:0;
    font-family: var(--serif);
    font-style: italic;
    font-weight: 400;
    font-size: 17px;
    letter-spacing: -0.005em;
  }}
  .map-head .meta{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .map-head .meta b{{ color: var(--ink-2); font-weight:500;}}
  .map-svg-wrap{{
    position: relative;
    aspect-ratio: 16 / 11;
    background:
      radial-gradient(circle at 50% 56%, oklch(99% 0.005 78) 0%, var(--paper) 70%);
  }}
  svg.map{{ position:absolute; inset:0; width:100%; height:100%;}}

  .map-help{{
    position: absolute;
    top: 14px;
    right: 14px;
    z-index: 10;
  }}
  .map-help summary{{
    list-style: none;
    cursor: pointer;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--paper);
    border: 1px solid var(--rule);
    color: var(--ink-3);
    font-family: var(--serif);
    font-style: italic;
    font-size: 16px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    transition: all .15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  .map-help summary::-webkit-details-marker{{ display: none; }}
  .map-help summary:hover{{
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
  }}
  .map-help[open] summary{{
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
  }}
  .map-help-popover{{
    position: absolute;
    top: 36px;
    right: 0;
    width: 320px;
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px -4px rgba(0,0,0,.12), 0 1px 3px rgba(0,0,0,.04);
  }}
  .map-help-popover p{{
    margin: 0 0 10px;
    font-family: var(--serif);
    font-weight: 380;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--ink-2);
    font-variation-settings: "opsz" 24;
  }}
  .map-help-popover p:last-child{{ margin-bottom: 0; }}
  .map-help-popover strong{{ font-weight: 600; color: var(--ink); }}
  .map-help-popover .map-help-title{{
    font-family: var(--mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: var(--ink-3);
    font-weight: 500;
    margin-bottom: 12px;
  }}
  .map-foot{{
    display:flex; justify-content: space-between; align-items: center;
    padding: 14px 24px;
    border-top: 1px solid var(--rule-2);
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .map-foot .legend{{ display:flex; gap:18px; align-items:center;}}
  .map-foot .legend i{{ display:inline-block; width:8px; height:8px; border-radius:999px; margin-right:8px; vertical-align:middle; transform: translateY(-1px);}}
  .map-foot .legend i.t{{ background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);}}
  .map-foot .legend i.n{{ background: var(--nine); box-shadow: 0 0 0 3px var(--nine-soft);}}

  .sec{{
    display:flex; align-items: baseline; gap: 14px;
    margin: 64px 0 22px;
  }}
  .sec .glyph{{
    font-family: var(--serif);
    font-style: italic;
    font-weight: 400;
    font-size: 22px;
    color: var(--ink-3);
  }}
  .sec h2{{
    margin:0;
    font-family: var(--serif);
    font-weight: 420;
    font-size: 28px;
    letter-spacing: -0.02em;
    color: var(--ink);
  }}
  .sec h2 em{{ font-style: italic; color: var(--accent); font-weight: 380;}}
  .sec .rule{{ flex: 1; height: 1px; background: var(--rule); transform: translateY(-4px);}}
  .sec .aside{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .sec .aside b{{ color: var(--ink-2); font-weight:500;}}

  .consensus{{
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 2px;
    overflow: hidden;
  }}
  .consensus-top{{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 24px;
    padding: 22px 32px;
    border-bottom: 1px solid var(--rule);
    background: linear-gradient(180deg, var(--nine-soft), var(--paper) 100%);
  }}
  .consensus-top .lbl{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--nine);
    font-weight: 500;
  }}
  .consensus-top h3{{
    margin: 4px 0 0;
    font-family: var(--serif);
    font-weight: 420;
    font-size: 24px;
    letter-spacing: -0.018em;
    color: var(--ink);
    text-wrap: balance;
  }}
  .consensus-top .d{{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-3);
    text-align: right;
    letter-spacing: .04em;
  }}
  .consensus-top .d b{{
    display: block;
    font-family: var(--serif);
    font-weight: 400;
    font-size: 26px;
    color: var(--nine);
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
    font-variant-numeric: tabular-nums;
  }}
  .consensus-body{{
    padding: 28px 32px 36px;
    column-count: 2;
    column-gap: 40px;
    column-rule: 1px solid var(--rule-2);
    font-family: var(--serif);
    font-weight: 380;
    font-size: 15px;
    line-height: 1.62;
    color: var(--ink);
    font-variation-settings: "opsz" 24;
  }}
  .consensus-body h3{{
    margin: 0 0 10px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
    break-after: avoid;
  }}
  .consensus-body h3:not(:first-child){{ margin-top: 22px; }}
  .consensus-body p{{ margin: 0 0 12px; break-inside: avoid-column; }}
  .consensus-body p:last-child{{ margin-bottom: 0; }}
  .consensus-body strong{{ font-weight: 600; color: var(--ink); }}
  .consensus-body em{{ font-style: italic; color: var(--ink); }}

  .tenth{{
    background: var(--paper);
    border: 1px solid var(--accent-soft);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }}
  .tenth::before{{
    content:"";
    position:absolute; left:0; top:0; bottom:0; width: 3px;
    background: var(--accent);
  }}
  .tenth-top{{
    display:grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 24px;
    padding: 24px 28px 22px 32px;
    border-bottom: 1px solid var(--rule-2);
  }}
  .tenth-eyebrow{{
    display:flex; align-items:center; gap:10px;
    font-family: var(--mono); font-size: 10.5px;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent);
  }}
  .tenth-eyebrow i{{ width:6px; height:6px; border-radius:999px; background: var(--accent);}}
  .tenth-top h2{{
    margin: 8px 0 0;
    font-family: var(--serif);
    font-weight: 420;
    font-style: italic;
    font-size: 28px;
    letter-spacing: -0.012em;
    color: var(--ink);
  }}
  .tenth-top .d{{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .04em;
    color: var(--ink-3);
    text-align: right;
  }}
  .tenth-top .d b{{
    display:block;
    font-family: var(--serif); font-style: italic;
    font-weight: 400; font-size: 32px;
    color: var(--accent);
    letter-spacing: -0.015em;
    line-height: 1;
    margin-bottom: 4px;
  }}
  .tenth-body{{
    padding: 26px 32px;
    font-family: var(--serif);
    font-weight: 380;
    font-size: 16.5px;
    line-height: 1.62;
    color: var(--ink);
    font-variation-settings: "opsz" 30;
    max-width: 78ch;
  }}
  .tenth-body h3{{
    margin: 22px 0 10px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .tenth-body h3:first-child{{ margin-top: 0; }}
  .tenth-body h3 .ord{{ color: var(--accent); margin-right: 10px; letter-spacing: .08em; }}
  .tenth-body h4{{
    margin: 18px 0 8px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .tenth-body p{{ margin: 0 0 12px; }}
  .tenth-body strong{{ font-weight: 600; color: var(--ink); }}
  .tenth-body em{{ font-style: italic; color: var(--ink); }}
  .tenth-body .pull{{
    margin: 18px 0;
    padding: 4px 0 4px 18px;
    border-left: 2px solid var(--accent);
    font-family: var(--serif); font-style: italic;
    font-size: 19px; line-height: 1.4;
    letter-spacing: -0.005em;
    color: var(--ink);
  }}
  .tenth-body .modes{{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border-top: 1px solid var(--rule-2);
    margin-top: 28px;
    padding-top: 8px;
  }}
  .tenth-body .mode{{
    padding: 18px 22px 4px 0;
    border-right: 1px dashed var(--rule);
  }}
  .tenth-body .mode:nth-child(2){{ padding: 18px 22px 4px 22px; }}
  .tenth-body .mode:last-child{{ padding: 18px 0 4px 22px; border-right: none; }}
  .tenth-body .mode .ord{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent);
  }}
  .tenth-body .mode h4{{
    margin: 6px 0 8px;
    font-family: var(--serif);
    font-weight: 460;
    font-size: 16px;
    letter-spacing: -0.005em;
    text-transform: none;
    color: var(--ink);
  }}
  .tenth-body .mode p{{
    margin: 0;
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.55;
    color: var(--ink-2);
  }}
  .tenth-foot{{
    display:flex; justify-content: space-between; gap: 18px;
    padding: 14px 32px;
    border-top: 1px solid var(--rule-2);
    background: oklch(99% 0.005 78);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .04em;
    color: var(--ink-3);
  }}
  .tenth-foot b{{ color: var(--ink-2); font-weight: 500; }}

  .frames{{
    border-top: 1px solid var(--ink);
  }}
  .frame{{
    border-bottom: 1px solid var(--rule);
    padding: 20px 4px;
    cursor: pointer;
    transition: background .15s ease;
  }}
  .frame:hover{{ background: var(--paper); }}
  .frame-row{{
    display: grid;
    grid-template-columns: 36px 200px 1fr 160px 80px 18px;
    align-items: center;
    gap: 16px;
  }}
  .f-idx{{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .04em;
    color: var(--ink-3);
  }}
  .f-name{{
    font-family: var(--serif);
    font-weight: 440;
    font-size: 20px;
    letter-spacing: -0.012em;
    color: var(--ink);
  }}
  .f-tag{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .f-tag b{{ color: var(--ink-2); font-weight: 500;}}
  .f-bar{{
    height: 2px; background: var(--rule); position:relative; border-radius:2px;
    margin-right: 10px;
  }}
  .f-bar > i{{
    position:absolute; left:0; top:0; bottom:0; border-radius: 2px;
    background: var(--nine);
  }}
  .f-d{{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-2);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .f-d small{{ color: var(--ink-3); }}
  .f-caret{{
    font-family: var(--mono);
    color: var(--ink-3);
    transition: transform .2s ease;
    font-size: 12px;
  }}
  .frame.open .f-caret{{ transform: rotate(90deg); }}
  .frame.open .f-name{{ color: var(--accent); }}

  .f-body{{
    display: none;
    padding: 18px 0 6px 36px;
    max-width: 78ch;
    font-family: var(--serif);
    font-weight: 380;
    font-size: 15.5px;
    line-height: 1.62;
    color: var(--ink-2);
    font-variation-settings: "opsz" 24;
  }}
  .frame.open .f-body{{ display:block; }}
  .f-body h3, .f-body h4{{
    margin: 18px 0 8px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .f-body h3:first-child, .f-body h4:first-child{{ margin-top: 0; }}
  .f-body p{{ margin: 0 0 12px; }}
  .f-body strong{{ font-weight: 600; color: var(--ink); }}
  .f-body em{{ font-style: italic; color: var(--ink); }}

  .colophon{{
    margin-top: 64px;
    padding-top: 22px;
    border-top: 1px solid var(--rule);
    display:grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .04em;
    color: var(--ink-3);
  }}
  .colophon b{{ color: var(--ink-2); font-weight: 500; }}
  .colophon .center{{
    text-align:center;
    font-family: var(--serif);
    font-style: italic;
    letter-spacing: 0;
    font-size: 13px;
    color: var(--ink-2);
  }}
  .colophon .right{{ text-align: right; }}

  .guide{{
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 50;
    font-family: var(--sans);
  }}
  .guide-toggle{{
    appearance: none;
    -webkit-appearance: none;
    border: 1px solid var(--ink);
    background: var(--ink);
    color: var(--paper);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: 12px 16px;
    border-radius: 999px;
    cursor: pointer;
    box-shadow: 0 6px 24px -8px oklch(20% 0.012 60 / .45);
    display: inline-flex;
    align-items: center;
    gap: 10px;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
    outline: none;
  }}
  .guide-toggle:hover{{
    transform: translateY(-1px);
    box-shadow: 0 10px 28px -8px oklch(20% 0.012 60 / .55);
  }}
  .guide-toggle:focus-visible{{
    box-shadow: 0 0 0 3px var(--accent-soft), 0 6px 24px -8px oklch(20% 0.012 60 / .45);
  }}
  .guide-toggle .marker{{
    width:7px; height:7px; border-radius:999px;
    background: var(--accent);
    box-shadow: 0 0 0 3px oklch(56% 0.165 32 / .25);
  }}
  .guide-panel{{
    display:none;
    width: min(380px, calc(100vw - 48px));
    max-height: min(72vh, 640px);
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 14px;
    box-shadow: 0 24px 60px -16px oklch(20% 0.012 60 / .35);
    margin-bottom: 12px;
    overflow: hidden;
    flex-direction: column;
  }}
  .guide.open .guide-panel{{ display: flex; }}
  .guide-panel-body{{
    overflow-y: auto;
    padding: 26px 24px 4px 26px;
    scrollbar-width: thin;
    scrollbar-color: var(--rule) transparent;
  }}
  .guide-panel-body::-webkit-scrollbar{{ width: 8px; }}
  .guide-panel-body::-webkit-scrollbar-track{{ background: transparent; }}
  .guide-panel-body::-webkit-scrollbar-thumb{{
    background: var(--rule);
    border-radius: 999px;
    border: 2px solid var(--paper);
  }}
  .guide-panel-body::-webkit-scrollbar-thumb:hover{{ background: var(--ink-4); }}
  .guide-panel-foot{{
    flex-shrink: 0;
    padding: 14px 26px 18px;
    border-top: 1px solid var(--rule);
    background: var(--paper);
  }}
  .guide-panel h3{{
    margin: 0 0 4px;
    font-family: var(--serif);
    font-weight: 400;
    font-size: 22px;
    line-height: 1.15;
    letter-spacing: -0.015em;
    color: var(--ink);
    font-variation-settings: "SOFT" 30, "opsz" 60;
  }}
  .guide-panel h3 em{{ font-style: italic; color: var(--accent); }}
  .guide-panel .kicker{{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 0 0 14px;
  }}
  .guide-panel ol{{
    margin: 0;
    padding: 0;
    list-style: none;
    counter-reset: g;
  }}
  .guide-panel ol li{{
    counter-increment: g;
    position: relative;
    padding: 12px 0 12px 30px;
    border-top: 1px solid var(--rule-2);
    font-size: 13.5px;
    line-height: 1.5;
    color: var(--ink-2);
  }}
  .guide-panel ol li:first-child{{ border-top: none; }}
  .guide-panel ol li::before{{
    content: counter(g, decimal-leading-zero);
    position: absolute;
    left: 0; top: 12px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--ink-3);
    letter-spacing: .08em;
  }}
  .guide-panel ol li b{{ color: var(--ink); font-weight: 600; }}
  .guide-panel .foot{{
    margin: 0 0 12px;
    font-family: var(--serif);
    font-style: italic;
    font-size: 13px;
    line-height: 1.45;
    color: var(--ink-2);
    font-variation-settings: "opsz" 60;
  }}
  .guide-close{{
    appearance: none;
    -webkit-appearance: none;
    background: none;
    border: 0;
    cursor: pointer;
    color: var(--ink-3);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: 0;
    outline: none;
  }}
  .guide-close:hover{{ color: var(--ink); }}
  .guide-close:focus-visible{{ color: var(--accent); }}

  @media (max-width: 920px){{
    .page{{ padding: 32px 22px 64px; }}
    .strip{{ grid-template-columns: 1fr 1fr; }}
    .strip .stat:nth-child(3){{ grid-column: 1 / -1; padding: 16px 0 0; border-right:none; padding-left: 0; border-top: 1px solid var(--rule); margin-top: 8px;}}
    .frame-row{{ grid-template-columns: 26px 1fr 80px 16px; }}
    .frame-row .f-tag, .frame-row .f-bar{{ display:none; }}
    .guide{{ right: 16px; bottom: 16px; }}
    .guide-panel-body{{ padding: 22px 18px 4px 20px; }}
    .guide-panel-foot{{ padding: 12px 20px 16px; }}
    .consensus-body{{ column-count: 1; }}
    .consensus-top{{ grid-template-columns: 1fr; }}
    .consensus-top .d{{ text-align: left; margin-top: 8px; }}
    .tenth-body .modes{{ grid-template-columns: 1fr; }}
    .tenth-body .mode,
    .tenth-body .mode:nth-child(2),
    .tenth-body .mode:last-child{{
      padding: 16px 0;
      border-right: none;
      border-bottom: 1px dashed var(--rule);
    }}
    .tenth-body .mode:last-child{{ border-bottom: none; }}
  }}
</style>
</head>
<body>
<main class="page" data-screen-label="Henge Report">

  <header>
    <div class="mast">
      <span class="rule"></span>
      <span class="dot"></span>
      <b>Henge</b>
      <span class="sep">·</span>
      <span>Disagreement Map</span>
      <span class="sep">·</span>
      <span>Report&nbsp;<b style="font-weight:500;">#{report_id}</b></span>
      <span class="sep">·</span>
      <span>{timestamp_short}</span>
    </div>

    <h1 class="head">Nueve consejeros alineados.<br/><em>El décimo debe disentir.</em></h1>

    <p class="question">{question_safe}</p>
  </header>

  <section class="strip" aria-label="Métricas globales">
    <div class="stat accent">
      <div class="lbl">Distancia décimo hombre</div>
      <div class="val">{tenth_distance:.3f} <span class="num">vs centroide de los 9</span></div>
    </div>
    <div class="stat">
      <div class="lbl">Consejero más en contra</div>
      <div class="val">{html_mod.escape(most_divergent_name)} <span class="num">d {max_frame_distance:.3f}</span></div>
    </div>
    <div class="stat">
      <div class="lbl">Veredicto</div>
      <div class="verdict">{fragility}</div>
    </div>
  </section>

  <section class="map-card" aria-label="Mapa MDS">
    <div class="map-head">
      <h3>Geografía del desacuerdo</h3>
      <div class="meta">10 voces · MDS sobre <b>cosine distance</b> · centroide de los 9</div>
    </div>
    <div class="map-svg-wrap">
      <details class="map-help">
        <summary aria-label="Cómo leer este mapa">?</summary>
        <div class="map-help-popover">
          <p class="map-help-title">Cómo leer este mapa</p>
          <p>Cada punto es un consejero.</p>
          <p>El <strong>centroide</strong> al centro es la zona de consenso — donde el grupo coincide.</p>
          <p>Más <strong>cerca</strong> del centro = más alineado con el resto.<br/>Más <strong>lejos</strong> = piensa distinto.</p>
          <p>El punto <strong style="color: var(--accent)">rojo (10 · tenth-man)</strong> es el disidente obligado.</p>
          <p>Los anillos concéntricos marcan distancias iguales al centroide.</p>
        </div>
      </details>
      {map_svg}
    </div>
    <div class="map-foot">
      <div class="legend">
        <span><i class="t"></i>Décimo hombre</span>
        <span><i class="n"></i>Los nueve consejeros</span>
      </div>
      <div>Mapa MDS clásico · preserva distancias por pares</div>
    </div>
  </section>

  {consensus_block_html}

  <div class="sec">
    <span class="glyph">¶</span>
    <h2>Los nueve consejeros</h2>
    <span class="rule"></span>
    <span class="aside">orden por <b>distancia al centroide</b> · click para expandir</span>
  </div>

  <section class="frames" id="frames">
    {frame_cards_html}
  </section>

  <div class="sec">
    <span class="glyph">¶</span>
    <h2>El décimo hombre — <em>disenso steel-man</em></h2>
    <span class="rule"></span>
    <span class="aside">d&nbsp;<b>{tenth_distance:.3f}</b> · obligación de discrepar</span>
  </div>

  <article class="tenth">
    <header class="tenth-top">
      <div>
        <div class="tenth-eyebrow"><i></i>Frame 10 · Tenth-man</div>
        <h2>Por qué los nueve podrían estar equivocados</h2>
      </div>
      <div class="d"><b>{tenth_distance:.3f}</b>distancia al consenso</div>
    </header>

    <div class="tenth-body">
      {tenth_response_html}
      {tenth_modes_html}
    </div>

    <footer class="tenth-foot">
      <span>Generado bajo restricción · <b>steel-man</b> obligatorio</span>
      <span>embed <b>{html_mod.escape(provider)}/{html_mod.escape(model)}</b></span>
    </footer>
  </article>

  <footer class="colophon">
    <div>
      <b>Henge</b><br/>
      classical MDS · cosine distance<br/>
      embed: {html_mod.escape(provider)}/{html_mod.escape(model)}
    </div>
    <div class="center">
      «Nueve voces alineadas no son señal — son sólo ruido coherente.»
    </div>
    <div class="right">
      <b>Costo estimado</b> ~CLP {cost_estimate_clp:.0f}<br/>
      {timestamp}<br/>
      report&nbsp;#{report_id}
    </div>
  </footer>

</main>

<aside class="guide" id="guide" aria-label="Cómo leer este reporte">
  <div class="guide-panel" role="dialog" aria-labelledby="guide-title">
    <div class="guide-panel-body">
      <p class="kicker">Guía de lectura</p>
      <h3 id="guide-title">Cómo abordar <em>este reporte</em></h3>
      <ol>
        <li><b>Empieza por el consenso, no por el décimo.</b> Es lo que los 9 consejeros creen en común — el ancla de la decisión.</li>
        <li><b>Los 9 consejeros no son votos.</b> Son lentes distintos sobre el mismo problema. Lee las diferencias entre ellos, no la mayoría.</li>
        <li><b>El décimo es auditoría, no recomendación.</b> Su rol es atacar el consenso para probar si aguanta. Sonar convincente es su trabajo, no señal de que tenga razón.</li>
        <li><b>Aísla los ataques del décimo, no su veredicto.</b> Quédate con las preguntas que abre y evalúa cada una contra tu realidad — descarta su conclusión si no resiste.</li>
        <li><b>Mira las métricas.</b> Fragilidad alta + distancia 10º alta = consenso débil, el disenso pesa más. Fragilidad baja = consenso robusto, el disenso es desafío retórico.</li>
        <li><b>Aplica el test asimétrico.</b> Si el décimo describe un riesgo que reconoces como real en tu vida, súmalo aunque no cambies de bando. Si no lo reconoces, descártalo.</li>
        <li><b>Tú decides.</b> Ningún consejero conoce tu contexto completo. El reporte expone tensiones; la elección es tuya.</li>
      </ol>
    </div>
    <div class="guide-panel-foot">
      <p class="foot">El consenso protege contra el error obvio. El décimo protege contra el error compartido. Necesitas ambos lentes — y ninguno reemplaza tu juicio.</p>
      <button type="button" class="guide-close" onclick="document.getElementById('guide').classList.remove('open')">Cerrar</button>
    </div>
  </div>
  <button type="button" class="guide-toggle" onclick="document.getElementById('guide').classList.toggle('open')" aria-expanded="false">
    <span class="marker"></span>
    ¿Cómo leer esto?
  </button>
</aside>

<script>
  (function(){{
    var g = document.getElementById('guide');
    var btn = g.querySelector('.guide-toggle');
    btn.addEventListener('click', function(){{
      btn.setAttribute('aria-expanded', g.classList.contains('open') ? 'true' : 'false');
    }});
    document.addEventListener('keydown', function(e){{
      if(e.key === 'Escape' && g.classList.contains('open')){{
        g.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      }}
    }});
  }})();
</script>

<script>
  document.querySelectorAll('.frame').forEach(f => {{
    f.addEventListener('click', () => f.classList.toggle('open'));
  }});
</script>
</body>
</html>"""

    return page
