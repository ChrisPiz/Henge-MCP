"""HTML viz: TenthAI / Antimetal-style disagreement report.

Design language (v3):
- Compact masthead: small logo + report meta
- Hero: oil-painting bg behind navy gradient veil, headline + 4 stat cards (right)
- Sober ghost-canvas page surface; pure white elevated cards w/ blue-tinted
  shadow rings (no borders)
- Single chartreuse (#d0f100) accent — used only on the tenth-man and primary
  affordances; nine-advisor consensus sits in midnight-navy
- Type pairing: Fraunces (serif display + numerics) + DM Sans (UI) +
  JetBrains Mono (data, labels, meta)

Layout: masthead → hero (painting + 4 stats) → 01 Report (question + map +
optional consensus + tenth-card) → 02 Frames (frames table) → colophon.
The 9 frames are sorted by distance to centroide ascending; the closest one is
expanded by default. The tenth-man sits in its own chartreuse-accented card
with a 3-up failure-modes grid when the [FAILURE_MODES] block is present.
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
            "verdict": "Advisors divided — there was no strong consensus to begin with.",
        }
    if tenth_distance > DISSENT_RATIO * max_frame_distance:
        return {
            "state": "aligned-fragile",
            "label_short": "fragile consensus",
            "verdict": "Strong but fragile consensus — the dissenter breaks it coherently.",
        }
    return {
        "state": "aligned-stable",
        "label_short": "aligned",
        "verdict": "Advisors aligned — dissent sounds reasonable but consensus holds.",
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
    """Wrap ``§N`` prefix in tenth-body h3 headings with an accent span.

    Pattern in the rendered HTML: ``<h3>§1 Hechos que acepto</h3>``. We wrap
    the §N portion in a dedicated span the CSS picks up. We also rewrite to
    h4 to match the v3 template's smaller ord/title eyebrow treatment.
    """
    return re.sub(
        r"<h3>(§\s*\d+)\s+(.+?)</h3>",
        r'<h4><span class="ord">\1</span>\2</h4>',
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


def _stddev(values):
    """Population standard deviation. Returns 0.0 for empty / single-element input."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ───────── Map SVG (TenthAI v3 layout) ─────────
# viewBox 1200x638, centroid at (600, 319), three concentric distance rings,
# nine consensus nodes in midnight-navy, tenth-man with chartreuse halo.

_MAP_VB_W = 1200
_MAP_VB_H = 638
_MAP_CX = 600
_MAP_CY = 319
_MAP_MAX_R = 200  # outermost ring; consensus nodes scale to fit inside.


def _map_to_svg(coords_2d):
    """Map 10 MDS coords into the v3 SVG viewBox.

    Scales so the largest extent lands at ``_MAP_MAX_R`` from centroid. Returns
    a list of (x, y) pairs in SVG user units, length 10.
    """
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]
    max_extent = max(
        max((abs(x) for x in xs), default=0.001),
        max((abs(y) for y in ys), default=0.001),
        0.001,
    )
    scale = _MAP_MAX_R / max_extent
    return [(_MAP_CX + x * scale, _MAP_CY + y * scale) for x, y in zip(xs, ys)]


def _build_map_svg(coords_2d, frames, distances, max_frame_dist, min_frame_dist):
    """Generate the v3 disagreement-map SVG string.

    Visual structure:
      - axis crosshair + 3 distance rings (solid 60, dashed 120, dashed 200)
      - centroid marker (rotated square outline) with a 'CENTROID' label
      - 9 thin distance lines from centroid to each consensus node
      - 1 chartreuse dashed line + d-label pill from centroid to tenth-man
      - 9 navy nodes with frame index, name, distance and closest/farthest flag
      - tenth-man halo (3 stacked circles) with two-line label
    """
    points = _map_to_svg(coords_2d)
    cx, cy = _MAP_CX, _MAP_CY

    parts = [
        f'<svg viewBox="0 0 {_MAP_VB_W} {_MAP_VB_H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">',
        # Axis crosshair
        '<g stroke="rgba(0,39,80,0.06)" stroke-width="0.5">',
        f'<line x1="{cx}" y1="40" x2="{cx}" y2="598"/>',
        f'<line x1="80" y1="{cy}" x2="{_MAP_VB_W - 80}" y2="{cy}"/>',
        '</g>',
        # Distance rings
        '<g stroke="rgba(0,39,80,0.10)" stroke-width="0.7" fill="none">',
        f'<circle cx="{cx}" cy="{cy}" r="60"/>',
        f'<circle cx="{cx}" cy="{cy}" r="120" stroke-dasharray="3 5"/>',
        f'<circle cx="{cx}" cy="{cy}" r="200" stroke-dasharray="3 5"/>',
        '</g>',
        # Centroid marker + label
        '<g>',
        f'<rect x="{cx - 8}" y="{cy - 8}" width="16" height="16" fill="none" '
        f'stroke="#1b2540" stroke-width="1" transform="rotate(45 {cx} {cy})"/>',
        f'<text x="{cx + 18}" y="{cy - 3}" font-family="JetBrains Mono, monospace" '
        f'font-size="11" fill="#6b7184" letter-spacing="1">CENTROID</text>',
        '</g>',
    ]

    # Distance lines for the 9 consensus advisors
    parts.append('<g stroke="rgba(27,37,64,0.20)" stroke-width="0.7">')
    for i in range(9):
        x, y = points[i]
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>')
    parts.append('</g>')

    # Tenth-man dashed accent line
    tx, ty = points[9]
    parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" '
        f'stroke="#d0f100" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )

    # Tenth-man distance pill, drawn at the midpoint of the dashed line
    mid_x = (cx + tx) / 2
    mid_y = (cy + ty) / 2
    pill_w = 60
    pill_h = 22
    parts.append(
        f'<rect x="{mid_x - pill_w / 2:.1f}" y="{mid_y - pill_h / 2:.1f}" '
        f'width="{pill_w}" height="{pill_h}" rx="11" fill="#1b2540"/>'
    )
    parts.append(
        f'<text x="{mid_x:.1f}" y="{mid_y + 4:.1f}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="11" font-weight="500" '
        f'fill="#d0f100">d {distances[9]:.3f}</text>'
    )

    # Nine consensus nodes
    parts.append('<g font-family="DM Sans, sans-serif" font-size="13" fill="#1b2540">')
    for i in range(9):
        x, y = points[i]
        frame = frames[i]
        idx = FRAME_INDEX.get(frame, f"{i + 1:02d}")
        dist = distances[i]
        is_closest = abs(dist - min_frame_dist) < 1e-6
        is_farthest = abs(dist - max_frame_dist) < 1e-6
        # Anchor labels away from the centroid horizontally
        if x >= cx:
            anchor = "start"
            lx = x + 15
        else:
            anchor = "end"
            lx = x - 15
        marker_r = 7 if is_closest else 6
        suffix = ""
        if is_closest:
            suffix = " · closest"
        elif is_farthest:
            suffix = " · farthest"
        parts.append('<g>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{marker_r}" fill="#1b2540"/>')
        if is_closest:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="none" '
                f'stroke="#1b2540" stroke-width="1" stroke-dasharray="2 2"/>'
            )
        parts.append(
            f'<text x="{lx:.1f}" y="{y - 3:.1f}" font-weight="500" '
            f'text-anchor="{anchor}">{idx} {html_mod.escape(frame)}</text>'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y + 12:.1f}" font-family="JetBrains Mono, monospace" '
            f'font-size="10.5" fill="#6b7184" text-anchor="{anchor}">'
            f'd {dist:.3f}{suffix}</text>'
        )
        parts.append('</g>')
    parts.append('</g>')

    # Tenth-man node + chartreuse halo + label (placed away from the marker)
    parts.append('<g>')
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="48" fill="#d0f100" opacity="0.18"/>')
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="28" fill="#d0f100" opacity="0.32"/>')
    parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="12" fill="#d0f100"/>')
    parts.append(
        f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="12" fill="none" '
        f'stroke="#1b2540" stroke-width="1.5"/>'
    )
    # Push label above the marker if there's room, otherwise below
    label_offset = -52 if ty > cy - 60 else 64
    parts.append(
        f'<text x="{tx:.1f}" y="{ty + label_offset:.1f}" text-anchor="middle" '
        f'font-family="DM Sans, sans-serif" font-weight="500" font-size="13" '
        f'fill="#1b2540" letter-spacing="2">10 · TENTH-MAN</text>'
    )
    parts.append(
        f'<text x="{tx:.1f}" y="{ty + label_offset + 16:.1f}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="11" fill="#1b2540">'
        f'steel-man dissent</text>'
    )
    parts.append('</g>')

    # Axis labels
    parts.append(
        '<g font-family="JetBrains Mono, monospace" font-size="10" '
        'fill="#7c8293" letter-spacing="1">'
    )
    parts.append(f'<text x="80" y="{_MAP_VB_H - 23}">MDS-1 →</text>')
    parts.append(f'<text x="{cx}" y="34" text-anchor="middle">↑ MDS-2</text>')
    parts.append('</g>')

    parts.append('</svg>')
    return "\n".join(parts)


def _build_frame_card(frame, response, status, distance, max_dist, idx_str, is_open=False):
    """Build a frame-list article in the v3 table layout.

    Open by default for the closest frame to the centroid.
    """
    body = _md_to_html(response)
    bar_pct = min(100, (distance / max_dist) * 100) if max_dist > 0 else 0
    status_label = "OK" if status == "ok" else "FAILED"
    open_class = " open" if is_open else ""
    suffix_html = ""
    return f"""
    <article class="frame{open_class}" data-frame="{html_mod.escape(frame)}">
      <div class="frame-row">
        <span class="f-idx">#{idx_str}</span>
        <span class="f-name">{html_mod.escape(frame)}</span>
        <span class="f-tag"><span class="d"></span>{status_label}{suffix_html}</span>
        <span class="f-bar"><i style="width:{bar_pct:.0f}%"></i></span>
        <span class="f-d">d <b>{distance:.3f}</b></span>
        <span class="f-caret">›</span>
      </div>
      <div class="f-body">
        {body}
      </div>
    </article>
    """


def _build_frame_card_with_flag(frame, response, status, distance, max_dist, idx_str,
                                is_open=False, flag=None):
    """Same as _build_frame_card but injects a closest/farthest tag suffix."""
    body = _md_to_html(response)
    bar_pct = min(100, (distance / max_dist) * 100) if max_dist > 0 else 0
    status_label = "OK" if status == "ok" else "FAILED"
    open_class = " open" if is_open else ""
    flag_html = f' · <b>{html_mod.escape(flag)}</b>' if flag else ""
    return f"""
    <article class="frame{open_class}" data-frame="{html_mod.escape(frame)}">
      <div class="frame-row">
        <span class="f-idx">#{idx_str}</span>
        <span class="f-name">{html_mod.escape(frame)}</span>
        <span class="f-tag"><span class="d"></span>{status_label}{flag_html}</span>
        <span class="f-bar"><i style="width:{bar_pct:.0f}%"></i></span>
        <span class="f-d">d <b>{distance:.3f}</b></span>
        <span class="f-caret">›</span>
      </div>
      <div class="f-body">
        {body}
      </div>
    </article>
    """


def render(question, results, coords_2d, distances, provider, model, cost_estimate_usd, consensus=None):
    """Render the TenthAI/Antimetal-style disagreement report. Returns full HTML.

    Persistence and browser-open are handled by the caller (server.py orchestrates
    storage.write_record + webbrowser.open). Keeping render() pure makes it easy
    to test, embed, and post-process.

    Layout order: masthead → hero (painting + 4 stats) → 01 Report (question +
    map card + optional consensus card + tenth-man card) → 02 Frames (frames
    table) → colophon.
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
    closest_name = frames[closest_frame_idx]
    spread_sigma = _stddev(frame_distances)

    verdict = consensus_verdict(tenth_distance, max_frame_distance)
    fragility_text = verdict["verdict"]
    verdict_short = verdict["label_short"]
    verdict_state = verdict["state"]
    # Hero verdict cell — short editorial label
    hero_verdict_label = {
        "aligned-stable": "Aligned",
        "aligned-fragile": "Fragile",
        "divided": "Divided",
    }.get(verdict_state, "—")
    hero_verdict_sub = {
        "aligned-stable": "consensus holds",
        "aligned-fragile": "fragile consensus",
        "divided": "no strong consensus",
    }.get(verdict_state, "")

    # Map SVG (real MDS coords scaled into the v3 viewBox)
    map_svg = _build_map_svg(
        coords_2d=coords_2d,
        frames=frames,
        distances=distances,
        max_frame_dist=max_frame_distance,
        min_frame_dist=min_frame_distance,
    )

    # Frames sorted by distance ASC; closest is default open
    frame_order = sorted(range(9), key=lambda i: frame_distances[i])
    frame_cards_html = "\n".join(
        _build_frame_card_with_flag(
            frame=frames[i],
            response=responses[i],
            status=statuses[i],
            distance=frame_distances[i],
            max_dist=max_frame_distance,
            idx_str=FRAME_INDEX.get(frames[i], f"{i + 1:02d}"),
            is_open=(i == closest_frame_idx),
            flag=("closest" if i == closest_frame_idx
                  else "farthest" if i == most_divergent_idx
                  else None),
        )
        for i in frame_order
    )

    # Tenth-man body + optional 3-up failure modes grid
    tenth_main, tenth_modes = _split_failure_modes(responses[9])
    tenth_response_html = _style_section_markers(_md_to_html(tenth_main))
    if tenth_modes:
        mode_cards = []
        for i, (title, body) in enumerate(tenth_modes, start=1):
            mode_cards.append(
                f'<div class="mode">'
                f'<div class="ord">§ {i:02d}</div>'
                f'<h5>{html_mod.escape(title)}</h5>'
                f'<p>{html_mod.escape(body)}</p>'
                f'</div>'
            )
        tenth_modes_html = (
            '<h4><span class="ord">§ 4</span>Consensus failure modes</h4>'
            '<div class="modes">' + "".join(mode_cards) + '</div>'
        )
    else:
        tenth_modes_html = ""

    # Consensus block (optional). v3-styled card placed before the tenth-man.
    consensus_block_html = ""
    if consensus:
        consensus_title, consensus_body_md = _split_consensus_title(consensus)
        if not consensus_title:
            consensus_title = "What the nine agree on"
        consensus_html = _md_to_html(consensus_body_md)
        consensus_block_html = f"""
    <article class="consensus-card">
      <header class="consensus-top">
        <div>
          <div class="consensus-tag"><span class="d"></span>The consensus of the 9 · {html_mod.escape(verdict_short.upper())}</div>
          <p class="consensus-lead">What the nine advisors agree on, in one line:</p>
          <h3>{html_mod.escape(consensus_title)}</h3>
        </div>
        <div class="consensus-d">
          <b>{max_frame_distance:.3f}</b>
          max · vs centroid
        </div>
      </header>
      <div class="consensus-body">
        {consensus_html}
      </div>
    </article>
"""

    timestamp = datetime.now().strftime("%Y·%m·%d %H:%M CLT")
    timestamp_short = datetime.now().strftime("%Y·%m·%d")
    report_id = datetime.now().strftime("%H%M")
    question_safe = html_mod.escape(question)

    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Henge · Disagreement Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,450;9..40,500;9..40,600&family=Fraunces:opsz,wght@9..144,400;9..144,500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --midnight-navy: #1b2540;
    --deep-cosmos: #001033;
    --chartreuse: #d0f100;
    --ice-veil: #e0f6ff;
    --ghost-canvas: #f8f9fc;
    --pure: #ffffff;
    --slate-ink: #6b7184;
    --ash: #7c8293;
    --storm: #596075;
    --fog: #b1b5c0;

    --shadow-xl: rgba(0,39,80,0.03) 0 56px 72px -16px, rgba(0,39,80,0.03) 0 32px 32px -16px, rgba(0,39,80,0.04) 0 6px 12px -3px, rgba(0,39,80,0.04) 0 0 0 1px;
    --shadow-md: rgba(0,39,80,0.08) 0 6px 16px -3px, rgba(0,39,80,0.04) 0 0 0 1px;
    --shadow-subtle: rgba(255,255,255,0.72) 0 1px 1px 0 inset, rgba(4,33,80,0.02) 0 8px 16px 0, rgba(4,33,80,0.03) 0 4px 12px 0, rgba(4,33,80,0.06) 0 1px 2px 0, rgba(4,33,80,0.04) 0 0 0 1px;

    --sans:  'DM Sans', ui-sans-serif, system-ui, sans-serif;
    --serif: 'Fraunces', Georgia, serif;
    --mono:  'JetBrains Mono', ui-monospace, monospace;
  }}
  *{{ box-sizing: border-box; }}
  html, body{{ margin:0; padding:0; }}
  body{{
    background: var(--ghost-canvas);
    color: var(--midnight-navy);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.5;
    letter-spacing: -0.16px;
    -webkit-font-smoothing: antialiased;
  }}
  ::selection{{ background: var(--chartreuse); color: var(--midnight-navy); }}

  /* Compact masthead */
  .masthead{{
    max-width: 1200px;
    margin: 0 auto;
    padding: 18px 32px 14px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(0,39,80,0.08);
  }}
  .logo{{
    display:flex; align-items:center; gap: 10px;
    font-family: var(--serif);
    font-weight: 500; font-size: 18px; letter-spacing: -0.01em;
    color: var(--midnight-navy);
  }}
  .logo .mk{{
    width: 22px; height: 22px; border-radius: 6px;
    background: var(--midnight-navy);
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--chartreuse);
    font-family: var(--mono); font-size: 11px; font-weight: 500;
  }}
  .mast-meta{{
    font-family: var(--mono); font-size: 12px;
    color: var(--slate-ink); letter-spacing: 0.02em;
  }}
  .mast-meta b{{ color: var(--midnight-navy); font-weight: 500; }}
  .mast-meta .sep{{ color: var(--fog); margin: 0 8px; }}

  /* Hero */
  .hero{{
    max-width: 1200px;
    margin: 24px auto 0;
    border-radius: 20px;
    overflow: hidden;
    position: relative;
    box-shadow: var(--shadow-xl);
    color: #fafeff;
    isolation: isolate;
  }}
  .hero-bg{{
    position: absolute; inset: 0;
    background-image: url('assets/header-painting.jpg');
    background-size: cover;
    background-position: center 60%;
    z-index: -2;
  }}
  .hero::after{{
    content:"";
    position:absolute; inset:0; z-index:-1;
    background:
      linear-gradient(180deg, rgba(0,16,51,0.18) 0%, rgba(0,16,51,0.55) 60%, rgba(0,16,51,0.82) 100%),
      linear-gradient(90deg, rgba(0,16,51,0.55) 0%, rgba(0,16,51,0.10) 50%);
  }}
  .hero-inner{{
    padding: 40px;
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 40px;
    align-items: center;
  }}
  h1.hero-h{{
    margin: 0 0 14px;
    font-family: var(--serif);
    font-weight: 400;
    font-size: clamp(30px, 3.6vw, 44px);
    line-height: 1.05;
    letter-spacing: -0.4px;
    text-wrap: balance;
    max-width: 20ch;
    text-shadow: 0 1px 2px rgba(0,0,0,0.18);
  }}
  .hero-h em{{ font-style: italic; color: var(--chartreuse); }}
  .hero-dek{{
    margin: 0;
    max-width: 52ch;
    font-size: 15.5px; line-height: 1.5;
    color: rgba(224,246,255,0.92);
    letter-spacing: -0.09px;
  }}

  .hero-stats{{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .hero-stat{{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(224,246,255,0.18);
    border-radius: 14px;
    padding: 14px 16px;
  }}
  .hero-stat .lbl{{
    font-family: var(--mono); font-size: 10.5px;
    color: rgba(224,246,255,0.78);
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 6px;
  }}
  .hero-stat .val{{
    font-family: var(--serif); font-weight: 400;
    font-size: 22px; line-height: 1; letter-spacing: -0.22px;
    color: #fafeff;
    font-variant-numeric: tabular-nums;
  }}
  .hero-stat .val.acc{{ color: var(--chartreuse); }}
  .hero-stat .sub{{
    margin-top: 4px;
    font-size: 12px; color: rgba(224,246,255,0.70);
  }}

  /* Sections */
  .page{{ background: var(--ghost-canvas); }}
  .section{{
    max-width: 1200px;
    margin: 0 auto;
    padding: 80px 32px 0;
  }}
  .sec-head{{
    display: flex; align-items: end; justify-content: space-between; gap: 24px;
    margin-bottom: 28px;
  }}
  .sec-head .l{{ max-width: 64ch; }}
  .sec-eyebrow{{
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 12px;
    color: var(--slate-ink);
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 12px;
  }}
  .sec-eyebrow .n{{
    background: var(--midnight-navy); color: var(--pure);
    padding: 2px 7px; border-radius: 9999px;
    font-size: 10.5px; letter-spacing: 0.06em;
  }}
  .sec-head h2{{
    margin: 0;
    font-family: var(--serif); font-weight: 400;
    font-size: 40px; letter-spacing: -0.4px;
    line-height: 1.05; color: var(--midnight-navy);
    text-wrap: balance;
  }}
  .sec-head h2 em{{ font-style: italic; color: var(--midnight-navy); }}
  .sec-head .sub{{
    margin: 12px 0 0;
    font-size: 16px; color: var(--storm); letter-spacing: -0.16px;
  }}
  .question-pull{{
    margin: 20px 0 0;
    padding: 6px 0 6px 22px;
    border-left: 3px solid var(--chartreuse);
    font-family: var(--serif);
    font-style: italic;
    font-weight: 400;
    font-size: 24px;
    line-height: 1.35;
    letter-spacing: -0.22px;
    color: var(--midnight-navy);
    text-wrap: balance;
    max-width: 64ch;
  }}

  /* Map card */
  .map-card{{
    background: var(--pure);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    overflow: hidden;
  }}
  .map-top{{
    display: flex; justify-content: space-between; align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid rgba(0,39,80,0.06);
  }}
  .map-top .l{{ display:flex; align-items:center; gap: 14px; }}
  .map-top .pill{{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--ghost-canvas);
    border-radius: 9999px;
    padding: 5px 12px;
    font-family: var(--mono); font-size: 11px;
    color: var(--midnight-navy);
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
  }}
  .map-top .pill .d{{ width: 6px; height: 6px; border-radius: 9999px; background: var(--chartreuse); }}
  .map-top .meta{{
    font-family: var(--mono); font-size: 12px; color: var(--slate-ink);
  }}
  .legend-row{{ display:flex; gap: 16px; align-items: center; font-size: 13px; color: var(--slate-ink); }}
  .legend-row .ld{{
    display: inline-block; width: 9px; height: 9px;
    border-radius: 9999px; margin-right: 7px; transform: translateY(-1px);
  }}
  .legend-row .ld.t{{ background: var(--chartreuse); box-shadow: 0 0 0 3px rgba(208,241,0,0.18); }}
  .legend-row .ld.n{{ background: var(--midnight-navy); }}

  .map-svg{{
    aspect-ratio: 16 / 8.5;
    background: var(--pure);
    position: relative;
  }}
  .map-svg svg{{ width: 100%; height: 100%; display: block; }}

  .map-foot{{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0;
    border-top: 1px solid rgba(0,39,80,0.06);
  }}
  .map-foot .cell{{
    padding: 16px 20px;
    border-right: 1px solid rgba(0,39,80,0.06);
  }}
  .map-foot .cell:last-child{{ border-right: 0; }}
  .map-foot .cell .lbl{{
    font-family: var(--mono); font-size: 11px;
    color: var(--slate-ink); text-transform: uppercase;
    letter-spacing: 0.04em; margin-bottom: 6px;
  }}
  .map-foot .cell .val{{
    font-family: var(--serif); font-weight: 400;
    font-size: 22px; letter-spacing: -0.22px;
    color: var(--midnight-navy);
    font-variant-numeric: tabular-nums;
  }}
  .map-foot .cell .val .badge{{
    display: inline-block;
    background: var(--chartreuse); color: var(--midnight-navy);
    font-family: var(--mono); font-size: 11px; font-weight: 500;
    padding: 2px 8px; border-radius: 9999px;
    margin-left: 8px; letter-spacing: 0.04em; text-transform: uppercase;
    transform: translateY(-2px);
  }}
  .map-foot .cell .sub{{
    margin-top: 4px;
    font-size: 13px; color: var(--slate-ink);
  }}

  /* Consensus card (navy-themed) */
  .consensus-card{{
    background: linear-gradient(180deg, var(--ice-veil) 0%, var(--pure) 100%);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    overflow: hidden;
    margin-top: 24px;
    position: relative;
  }}
  .consensus-top{{
    display: grid; grid-template-columns: 1fr auto;
    align-items: start; gap: 24px;
    padding: 24px 28px;
    border-bottom: 1px solid rgba(0,39,80,0.06);
  }}
  .consensus-tag{{
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--midnight-navy); color: var(--pure);
    border-radius: 9999px; padding: 4px 10px;
    font-family: var(--mono); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 12px;
  }}
  .consensus-tag .d{{ width: 5px; height: 5px; border-radius: 9999px; background: var(--chartreuse); }}
  .consensus-lead{{
    margin: 0 0 8px;
    font-size: 14px;
    color: var(--storm);
    letter-spacing: -0.09px;
  }}
  .consensus-top h3{{
    margin: 0;
    font-family: var(--serif); font-weight: 400;
    font-size: 28px; letter-spacing: -0.4px; line-height: 1.08;
    color: var(--midnight-navy);
    text-wrap: balance;
    max-width: 28ch;
  }}
  .consensus-d{{
    text-align: right;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(4px);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: rgba(0,39,80,0.06) 0 0 0 1px;
    font-family: var(--mono); font-size: 11px;
    color: var(--slate-ink);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .consensus-d b{{
    display: block;
    font-family: var(--serif); font-weight: 400;
    font-size: 28px; letter-spacing: -0.4px; line-height: 1;
    color: var(--midnight-navy);
    margin-bottom: 4px;
    text-transform: none;
  }}
  .consensus-body{{
    padding: 28px 32px;
    column-count: 2;
    column-gap: 40px;
    column-rule: 1px solid rgba(0,39,80,0.06);
    font-size: 16px;
    line-height: 1.55;
    color: var(--midnight-navy);
    letter-spacing: -0.16px;
  }}
  .consensus-body p{{ margin: 0 0 12px; break-inside: avoid-column; }}
  .consensus-body p:last-child{{ margin-bottom: 0; }}
  .consensus-body h2,
  .consensus-body h3,
  .consensus-body h4{{
    margin: 0 0 10px;
    font-family: var(--mono); font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--slate-ink);
    break-after: avoid;
  }}
  .consensus-body h2:not(:first-child),
  .consensus-body h3:not(:first-child),
  .consensus-body h4:not(:first-child){{ margin-top: 22px; }}
  .consensus-body strong{{ font-weight: 600; color: var(--midnight-navy); }}
  .consensus-body em{{ font-style: italic; color: var(--midnight-navy); }}
  .consensus-body .pull{{
    margin: 16px 0;
    background: var(--ghost-canvas);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
    font-family: var(--serif); font-weight: 400;
    font-size: 18px; line-height: 1.35; letter-spacing: -0.18px;
    color: var(--midnight-navy);
    break-inside: avoid-column;
  }}

  /* Tenth-man feature card */
  .tenth-card{{
    background: var(--pure);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    overflow: hidden;
    margin-top: 24px;
    position: relative;
  }}
  .tenth-card::before{{
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px; background: var(--chartreuse);
  }}
  .tenth-top{{
    display: grid; grid-template-columns: 1fr auto;
    align-items: start; gap: 24px;
    padding: 24px 28px;
    border-bottom: 1px solid rgba(0,39,80,0.06);
  }}
  .tenth-tag{{
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--midnight-navy); color: var(--chartreuse);
    border-radius: 9999px; padding: 4px 10px;
    font-family: var(--mono); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 12px;
  }}
  .tenth-tag .d{{ width: 5px; height: 5px; border-radius: 9999px; background: var(--chartreuse); }}
  .tenth-lead{{
    margin: 0 0 8px;
    font-size: 14px;
    color: var(--storm);
    letter-spacing: -0.09px;
  }}
  .tenth-top h3{{
    margin: 0;
    font-family: var(--serif); font-weight: 400;
    font-size: 32px; letter-spacing: -0.4px; line-height: 1.05;
    color: var(--midnight-navy);
    text-wrap: balance;
    max-width: 24ch;
  }}
  .tenth-top h3 em{{ font-style: italic; }}

  .tenth-d-stat{{
    text-align: right;
    background: var(--ghost-canvas);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
    font-family: var(--mono); font-size: 11px;
    color: var(--slate-ink);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .tenth-d-stat b{{
    display: block;
    font-family: var(--serif); font-weight: 400;
    font-size: 32px; letter-spacing: -0.4px; line-height: 1;
    color: var(--midnight-navy);
    margin-bottom: 4px;
    text-transform: none;
  }}

  .tenth-body{{
    padding: 24px 28px;
    max-width: 78ch;
  }}
  .tenth-body h4{{
    margin: 24px 0 8px;
    font-family: var(--mono); font-weight: 500;
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--slate-ink);
  }}
  .tenth-body h4:first-child{{ margin-top: 0; }}
  .tenth-body h4 .ord{{ color: var(--midnight-navy); margin-right: 8px; }}
  .tenth-body p{{
    margin: 0 0 12px;
    font-size: 16px; line-height: 1.5;
    color: var(--midnight-navy);
    letter-spacing: -0.16px;
  }}
  .tenth-body strong{{ font-weight: 600; }}
  .tenth-body em{{ font-style: italic; color: var(--midnight-navy); }}
  .tenth-body .pull{{
    margin: 16px 0;
    background: var(--ghost-canvas);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
    font-family: var(--serif); font-weight: 400;
    font-size: 22px; line-height: 1.29; letter-spacing: -0.22px;
    color: var(--midnight-navy);
  }}
  .tenth-body .pull::before{{
    content:"\\201C";
    font-family: var(--serif);
    font-size: 36px; line-height: 0;
    margin-right: 6px;
    color: var(--chartreuse);
    vertical-align: -8px;
  }}

  .modes{{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin-top: 12px;
  }}
  .mode{{
    background: var(--ghost-canvas);
    border-radius: 16px;
    padding: 18px;
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
  }}
  .mode .ord{{
    font-family: var(--mono); font-size: 11px;
    color: var(--slate-ink);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .mode h5{{
    margin: 8px 0 6px;
    font-family: var(--sans); font-weight: 480;
    font-size: 17px; letter-spacing: -0.09px;
    color: var(--midnight-navy);
  }}
  .mode p{{
    margin: 0;
    font-size: 14px; line-height: 1.5;
    color: var(--storm); letter-spacing: -0.16px;
  }}

  .tenth-foot{{
    padding: 14px 28px;
    border-top: 1px solid rgba(0,39,80,0.06);
    background: var(--ghost-canvas);
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 12px;
    color: var(--slate-ink);
  }}
  .tenth-foot b{{ color: var(--midnight-navy); }}

  /* Frames table */
  .frames{{
    margin-top: 4px;
    background: var(--pure);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    overflow: hidden;
  }}
  .frames-head{{
    display: grid;
    grid-template-columns: 60px 220px 1fr 160px 100px 24px;
    gap: 16px;
    padding: 14px 24px;
    border-bottom: 1px solid rgba(0,39,80,0.06);
    background: var(--ghost-canvas);
    font-family: var(--mono); font-size: 11px;
    color: var(--slate-ink);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .frame{{
    border-bottom: 1px solid rgba(0,39,80,0.06);
    cursor: pointer;
    transition: background .15s ease;
  }}
  .frame:last-child{{ border-bottom: 0; }}
  .frame:hover{{ background: var(--ghost-canvas); }}
  .frame.open{{ background: var(--ghost-canvas); }}
  .frame-row{{
    display: grid;
    grid-template-columns: 60px 220px 1fr 160px 100px 24px;
    gap: 16px;
    align-items: center;
    padding: 16px 24px;
  }}
  .f-idx{{
    font-family: var(--mono); font-size: 12px; color: var(--slate-ink);
  }}
  .f-name{{
    font-family: var(--sans); font-weight: 480;
    font-size: 17px; letter-spacing: -0.09px;
    color: var(--midnight-navy);
  }}
  .f-tag{{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--pure);
    border-radius: 9999px;
    padding: 4px 10px;
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px;
    width: max-content;
  }}
  .f-tag .d{{ width: 5px; height: 5px; border-radius: 9999px; background: var(--chartreuse); }}
  .f-tag b{{ color: var(--midnight-navy); font-weight: 500; }}
  .f-bar{{
    height: 6px; background: rgba(27,37,64,0.08);
    border-radius: 9999px;
    position: relative; overflow: hidden;
  }}
  .f-bar > i{{
    display: block; height: 100%;
    background: var(--midnight-navy);
    border-radius: 9999px;
  }}
  .f-d{{
    font-family: var(--mono); font-size: 13px;
    color: var(--midnight-navy);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .f-d b{{ font-weight: 500; }}
  .f-caret{{
    color: var(--slate-ink);
    font-family: var(--mono);
    transition: transform .15s ease;
    text-align: center;
  }}
  .frame.open .f-caret{{ transform: rotate(90deg); }}

  .f-body{{
    display: none;
    padding: 4px 24px 22px 304px;
    max-width: 100%;
  }}
  .frame.open .f-body{{ display: block; }}
  .f-body p{{
    margin: 0 0 10px;
    font-size: 15px; line-height: 1.55;
    color: var(--storm); letter-spacing: -0.16px;
  }}
  .f-body strong{{ color: var(--midnight-navy); font-weight: 500; }}
  .f-body em{{ color: var(--midnight-navy); font-style: italic; }}
  .f-body .pull{{
    margin: 12px 0;
    padding: 4px 0 4px 14px;
    border-left: 2px solid var(--midnight-navy);
    font-family: var(--serif); font-style: italic;
    font-size: 16px; line-height: 1.4; letter-spacing: -0.09px;
    color: var(--midnight-navy);
  }}

  /* Map help "?" — floating inside the map card */
  .map-help{{
    position: absolute;
    top: 14px;
    right: 14px;
    z-index: 10;
  }}
  .map-help summary{{
    list-style: none;
    cursor: pointer;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--pure);
    color: var(--slate-ink);
    font-family: var(--serif);
    font-style: italic;
    font-size: 16px; font-weight: 500;
    display: flex; align-items: center; justify-content: center;
    user-select: none;
    transition: all .15s ease;
    box-shadow: rgba(0,39,80,0.04) 0 0 0 1px, rgba(0,39,80,0.10) 0 2px 6px -2px;
  }}
  .map-help summary::-webkit-details-marker{{ display: none; }}
  .map-help summary:hover,
  .map-help[open] summary{{
    background: var(--midnight-navy);
    color: var(--chartreuse);
    box-shadow: rgba(0,39,80,0.20) 0 4px 10px -2px;
  }}
  .map-help-popover{{
    position: absolute;
    top: 38px; right: 0;
    width: 320px;
    background: var(--pure);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: var(--shadow-md), rgba(0,39,80,0.06) 0 12px 24px -8px;
  }}
  .map-help-popover .map-help-title{{
    font-family: var(--mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--slate-ink);
    font-weight: 500;
    margin: 0 0 10px;
  }}
  .map-help-popover p{{
    margin: 0 0 10px;
    font-size: 14px; line-height: 1.5;
    color: var(--midnight-navy);
  }}
  .map-help-popover p:last-child{{ margin-bottom: 0; }}
  .map-help-popover strong{{ font-weight: 600; }}

  /* Reading guide — fixed floating bottom-right */
  .guide{{
    position: fixed;
    right: 24px; bottom: 24px;
    z-index: 50;
    font-family: var(--sans);
    display: flex; flex-direction: column;
    align-items: flex-end;
  }}
  .guide-toggle{{
    appearance: none; -webkit-appearance: none;
    border: 0;
    background: var(--midnight-navy);
    color: var(--chartreuse);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 12px 18px;
    border-radius: 9999px;
    cursor: pointer;
    display: inline-flex; align-items: center; gap: 10px;
    box-shadow:
      rgba(24,37,66,0.32) 0 1px 3px 0,
      rgba(24,37,66,0.44) 0 12px 24px -12px,
      rgba(219,247,255,0.48) 0 0.5px 0.5px 0 inset;
    transition: transform .15s ease, box-shadow .15s ease;
    outline: none;
  }}
  .guide-toggle:hover{{
    transform: translateY(-1px);
    box-shadow:
      rgba(24,37,66,0.40) 0 1px 3px 0,
      rgba(24,37,66,0.55) 0 16px 28px -12px,
      rgba(219,247,255,0.48) 0 0.5px 0.5px 0 inset;
  }}
  .guide-toggle:focus-visible{{
    box-shadow: 0 0 0 3px rgba(208,241,0,0.45), rgba(24,37,66,0.44) 0 12px 24px -12px;
  }}
  .guide-toggle .marker{{
    width: 7px; height: 7px; border-radius: 9999px;
    background: var(--chartreuse);
    box-shadow: 0 0 0 3px rgba(208,241,0,0.25);
  }}
  .guide-panel{{
    display: none;
    width: min(380px, calc(100vw - 48px));
    max-height: min(72vh, 640px);
    background: var(--pure);
    border-radius: 16px;
    box-shadow: var(--shadow-xl);
    margin-bottom: 12px;
    overflow: hidden;
    flex-direction: column;
  }}
  .guide.open .guide-panel{{ display: flex; }}
  .guide-panel-body{{
    overflow-y: auto;
    padding: 24px 24px 4px;
    scrollbar-width: thin;
  }}
  .guide-panel-foot{{
    flex-shrink: 0;
    padding: 14px 24px 18px;
    border-top: 1px solid rgba(0,39,80,0.08);
    background: var(--ghost-canvas);
  }}
  .guide-panel .kicker{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--slate-ink);
    margin: 0 0 8px;
  }}
  .guide-panel h3{{
    margin: 0 0 4px;
    font-family: var(--serif);
    font-weight: 400;
    font-size: 22px;
    letter-spacing: -0.22px;
    line-height: 1.15;
    color: var(--midnight-navy);
  }}
  .guide-panel h3 em{{ font-style: italic; color: var(--midnight-navy); }}
  .guide-panel ol{{
    margin: 18px 0 0;
    padding: 0;
    list-style: none;
    counter-reset: g;
  }}
  .guide-panel ol li{{
    counter-increment: g;
    position: relative;
    padding: 12px 0 12px 30px;
    border-top: 1px solid rgba(0,39,80,0.06);
    font-size: 13.5px;
    line-height: 1.5;
    color: var(--storm);
  }}
  .guide-panel ol li:first-child{{ border-top: none; }}
  .guide-panel ol li::before{{
    content: counter(g, decimal-leading-zero);
    position: absolute;
    left: 0; top: 12px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--slate-ink);
    letter-spacing: 0.06em;
  }}
  .guide-panel ol li b{{ color: var(--midnight-navy); font-weight: 600; }}
  .guide-panel .foot{{
    margin: 0 0 12px;
    font-family: var(--serif);
    font-style: italic;
    font-size: 13px;
    line-height: 1.45;
    color: var(--storm);
  }}
  .guide-close{{
    appearance: none; -webkit-appearance: none;
    background: none; border: 0; cursor: pointer;
    color: var(--slate-ink);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0;
    outline: none;
  }}
  .guide-close:hover{{ color: var(--midnight-navy); }}

  /* Colophon */
  footer.colophon{{
    max-width: 1200px;
    margin: 80px auto 0;
    padding: 32px;
    border-top: 1px solid rgba(0,39,80,0.08);
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    font-family: var(--mono); font-size: 12px;
    color: var(--slate-ink);
  }}
  footer.colophon b{{ color: var(--midnight-navy); font-weight: 500; }}
  footer.colophon .center{{ text-align: center; font-style: italic; font-family: var(--serif); }}
  footer.colophon .right{{ text-align: right; }}

  @media (max-width: 920px){{
    .hero-inner{{ grid-template-columns: 1fr; padding: 40px 24px 24px; }}
    .hero-stats{{ grid-template-columns: 1fr 1fr; }}
    .map-foot{{ grid-template-columns: 1fr 1fr; }}
    .map-foot .cell{{ border-bottom: 1px solid rgba(0,39,80,0.06); }}
    .modes{{ grid-template-columns: 1fr; }}
    .frames-head{{ display: none; }}
    .frame-row{{ grid-template-columns: 40px 1fr 80px 16px; }}
    .frame-row .f-tag, .frame-row .f-bar{{ display: none; }}
    .f-body{{ padding: 4px 24px 22px 24px; }}
    .tenth-top{{ grid-template-columns: 1fr; }}
    .tenth-d-stat{{ text-align: left; }}
    .consensus-top{{ grid-template-columns: 1fr; }}
    .consensus-d{{ text-align: left; }}
    .consensus-body{{ column-count: 1; }}
  }}
</style>
</head>
<body>
<main data-screen-label="Henge Report">

  <header class="masthead">
    <div class="logo"><span class="mk">10</span> Henge</div>
    <div class="mast-meta">
      <b>Report #{report_id}</b><span class="sep">·</span>{timestamp}<span class="sep">·</span>v0.4
    </div>
  </header>

  <section class="hero">
    <div class="hero-bg" aria-hidden="true"></div>
    <div class="hero-inner">
      <div>
        <h1 class="hero-h">Nine advisors aligned.<br><em>The tenth must dissent.</em></h1>
        <p class="hero-dek">Your question runs through nine cognitive frames. We measure the consensus and force a tenth to disagree with rigor.</p>
      </div>

      <div class="hero-stats">
        <div class="hero-stat">
          <div class="lbl">Tenth · d</div>
          <div class="val acc">{tenth_distance:.3f}</div>
          <div class="sub">vs centroid</div>
        </div>
        <div class="hero-stat">
          <div class="lbl">Closest</div>
          <div class="val">{min_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(closest_name)}</div>
        </div>
        <div class="hero-stat">
          <div class="lbl">Most divergent</div>
          <div class="val">{max_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(most_divergent_name)}</div>
        </div>
        <div class="hero-stat">
          <div class="lbl">Verdict</div>
          <div class="val">{html_mod.escape(hero_verdict_label)}</div>
          <div class="sub">{html_mod.escape(hero_verdict_sub)}</div>
        </div>
      </div>
    </div>
  </section>

  <div class="page">

  <section class="section">
    <div class="sec-head">
      <div class="l">
        <div class="sec-eyebrow"><span class="n">01</span>Report · #{report_id}</div>
        <h2>The question <em>the nine answered</em></h2>
        <blockquote class="question-pull">{question_safe}</blockquote>
      </div>
    </div>

    <section class="map-card">
      <div class="map-top">
        <div class="l">
          <span class="pill"><span class="d"></span>fig.01 · disagreement map</span>
          <span class="meta">10 voices · MDS · cosine</span>
        </div>
        <div class="legend-row">
          <span><i class="ld t"></i>Tenth man</span>
          <span><i class="ld n"></i>Consensus frames</span>
        </div>
      </div>

      <div class="map-svg">
        <details class="map-help">
          <summary aria-label="How to read this map">?</summary>
          <div class="map-help-popover">
            <p class="map-help-title">How to read this map</p>
            <p>Each point is one advisor.</p>
            <p>The <strong>centroid</strong> at the center is the consensus zone — where the group converges.</p>
            <p><strong>Closer</strong> to the center = more aligned with the rest.<br>
               <strong>Farther</strong> = thinks differently.</p>
            <p>The <strong style="color: var(--midnight-navy); background: var(--chartreuse); padding: 0 4px; border-radius: 3px;">10 · tenth-man</strong> point is the mandatory dissenter.</p>
            <p>Concentric rings mark equal distances from the centroid.</p>
          </div>
        </details>
        {map_svg}
      </div>

      <div class="map-foot">
        <div class="cell">
          <div class="lbl">Tenth · d</div>
          <div class="val">{tenth_distance:.3f}<span class="badge">tenth</span></div>
          <div class="sub">forced to dissent</div>
        </div>
        <div class="cell">
          <div class="lbl">Closest</div>
          <div class="val">{min_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(closest_name)}</div>
        </div>
        <div class="cell">
          <div class="lbl">Most divergent (of the 9)</div>
          <div class="val">{max_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(most_divergent_name)}</div>
        </div>
        <div class="cell">
          <div class="lbl">Internal spread</div>
          <div class="val">{spread_sigma:.3f}</div>
          <div class="sub">σ across the 9 frames</div>
        </div>
      </div>
    </section>

    {consensus_block_html}

    <article class="tenth-card">
      <header class="tenth-top">
        <div>
          <div class="tenth-tag"><span class="d"></span>The dissent of the tenth · STEEL-MAN</div>
          <p class="tenth-lead">The mandatory dissenter's strongest case against the nine:</p>
          <h3>Why the nine <em>might be wrong</em></h3>
        </div>
        <div class="tenth-d-stat">
          <b>{tenth_distance:.3f}</b>
          distance · centroid
        </div>
      </header>

      <div class="tenth-body">
        {tenth_response_html}
        {tenth_modes_html}
      </div>

      <footer class="tenth-foot">
        <span>Generated under constraint · <b>steel-man mandatory</b></span>
        <span>embed <b>{html_mod.escape(provider)}/{html_mod.escape(model)}</b> · ~USD {cost_estimate_usd:.2f}</span>
      </footer>
    </article>
  </section>

  <section class="section">
    <div class="sec-head">
      <div class="l">
        <div class="sec-eyebrow"><span class="n">02</span>Cognitive frames · 9 voices</div>
        <h2>The nine, ranked by <em>distance to centroid</em></h2>
        <p class="sub">Each frame brings a different lens: empirical, systemic, historical, analogical. The closer to the centroid, the more it represents the consensus. The farther, the more it reasons alone.</p>
      </div>
    </div>

    <section class="frames">
      <div class="frames-head">
        <span>Idx</span>
        <span>Frame</span>
        <span>Status</span>
        <span>Distance</span>
        <span style="text-align:right;">d</span>
        <span></span>
      </div>
      {frame_cards_html}
    </section>
  </section>

  <footer class="colophon">
    <div>
      <b>Henge v0.4</b><br>
      classical MDS · cosine<br>
      embed · {html_mod.escape(provider)}/{html_mod.escape(model)}
    </div>
    <div class="center">
      «{html_mod.escape(fragility_text)}»
    </div>
    <div class="right">
      <b>~USD {cost_estimate_usd:.2f}</b><br>
      {timestamp}<br>
      report&nbsp;<b>#{report_id}</b>
    </div>
  </footer>

  </div>
</main>

<aside class="guide" id="guide" aria-label="How to read this report">
  <div class="guide-panel" role="dialog" aria-labelledby="guide-title">
    <div class="guide-panel-body">
      <p class="kicker">Reading guide</p>
      <h3 id="guide-title">How to approach <em>this report</em></h3>
      <ol>
        <li><b>Start with the consensus, not the tenth.</b> It's what the 9 advisors agree on — the anchor of the decision.</li>
        <li><b>The 9 advisors aren't votes.</b> They're distinct lenses on the same problem. Read the differences between them, not the majority.</li>
        <li><b>The tenth is audit, not recommendation.</b> Its role is to attack the consensus to test if it holds. Sounding convincing is its job, not a signal it's right.</li>
        <li><b>Isolate the tenth's attacks, not its verdict.</b> Keep the questions it opens and evaluate each against your reality — discard its conclusion if it doesn't hold.</li>
        <li><b>Read the metrics.</b> High fragility + high tenth-man distance = weak consensus, dissent matters more. Low fragility = robust consensus, dissent is rhetorical.</li>
        <li><b>Apply the asymmetric test.</b> If the tenth names a risk you recognize as real in your life, add it even if you don't switch sides. If you don't recognize it, discard.</li>
        <li><b>You decide.</b> No advisor knows your full context. The report exposes tensions; the choice is yours.</li>
      </ol>
    </div>
    <div class="guide-panel-foot">
      <p class="foot">Consensus protects against the obvious error. The tenth protects against the shared one. You need both lenses — and neither replaces your judgment.</p>
      <button type="button" class="guide-close" onclick="document.getElementById('guide').classList.remove('open')">Close</button>
    </div>
  </div>
  <button type="button" class="guide-toggle" onclick="document.getElementById('guide').classList.toggle('open')" aria-expanded="false">
    <span class="marker"></span>
    How do I read this?
  </button>
</aside>

<script>
  document.querySelectorAll('.frame').forEach(f => {{
    f.addEventListener('click', () => f.classList.toggle('open'));
  }});
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
</body>
</html>"""

    return page
