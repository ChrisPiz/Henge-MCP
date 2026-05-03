"""HTML viz: TenthAI / Antimetal-style disagreement report.

Design language (v3):
- Compact masthead: small logo + report meta
- Hero: oil-painting bg behind navy gradient veil, headline + 4 stat cards (right)
- Sober ghost-canvas page surface; pure white elevated cards w/ blue-tinted
  shadow rings (no borders)
- Single chartreuse (#d0f100) accent — used only on the tenth-man and primary
  affordances; nine-advisor consensus sits in midnight-navy
- Type pairing: Fraunces (serif display + numerics) + Source Sans Pro (UI) +
  JetBrains Mono (data, labels, meta)

Layout: masthead → hero (painting + 4 stats) → 01 Report (question + map +
optional consensus + tenth-card) → 02 Frames (frames table) → colophon.
The 9 frames are sorted by distance to centroide ascending; the closest one is
expanded by default. The tenth-man sits in its own chartreuse-accented card
with a 3-up failure-modes grid when the [FAILURE_MODES] block is present.
"""
import html as html_mod
import math
import os
import re
from datetime import datetime

from henge.config.frame_assignment import FRAME_MODEL_MAP


# ───────── Locale ─────────
# A) auto-detect from the question's character/word frequency
# B) override with HENGE_LOCALE=en|es (env var)

_SPANISH_RE = re.compile(
    # Spanish-only characters and punctuation
    r"[¿¡áéíóúñü]|"
    # Function words (hard to confuse with English)
    r"\b(qué|cómo|cuál|cuáles|cuándo|dónde|por\s?qué|para\s?qué|si|sí|"
    r"debo|debería|debería|tengo|tiene|tienen|tenía|hacer|saber|"
    r"nuestro|nuestra|nuestros|nuestras|nosotros|nosotras|ustedes|"
    r"ellos|ellas|aquí|allí|allá|esto|eso|aquello|esta|este|estas|estos|"
    r"para|porque|pues|cuando|donde|cómo|también|además|después|antes|"
    r"entonces|ahora|todavía|aún|conmigo|contigo|"
    # Content words common in decision questions
    r"auto|autos|carro|carros|coche|coches|moto|casa|hogar|piso|"
    r"nuevo|nueva|nuevos|nuevas|usado|usada|usados|usadas|viejo|vieja|"
    r"comprar|vender|compré|vendí|alquilar|arrendar|arriendo|"
    r"trabajo|trabajar|empleo|carrera|jubilación|sueldo|salario|"
    r"hijo|hija|hijos|hijas|padre|madre|esposa|esposo|pareja|familia|"
    r"dinero|plata|ahorros|inversión|deuda|crédito|hipoteca|"
    r"vida|salud|enfermedad|muerte|futuro|pasado|presente|"
    r"bueno|buena|buenos|buenas|malo|mala|mejor|peor|grande|pequeño|"
    r"mucho|mucha|muchos|muchas|poco|poca|pocos|pocas|"
    r"todo|toda|todos|todas|nada|alguien|nadie|algo|"
    r"oportunidad|decisión|riesgo|negocio|empresa|emprender|"
    r"viaje|viajar|mudarme|mudarse|emigrar|"
    r"semana|mes|meses|año|años|día|días|hora|horas|"
    r"sin|con|hacia|desde|hasta|según|durante|"
    r"convieneme|conviene|deber|deberé)\b|"
    # Spanish-only suffixes (catch verb/adjective/noun endings)
    r"\w+(ción|cciones|mente|dad|dades|tud|tudes|isimo|ísima)\b",
    re.IGNORECASE,
)
_ENGLISH_RE = re.compile(
    r"\b(should|would|could|the|i|my|how|what|when|where|why|which|whom|whose|"
    r"do|does|did|is|am|are|was|were|will|wo|won't|can|cannot|may|might|must|"
    r"have|has|had|been|that|this|these|those|or|and|but|with|from|into|onto|"
    r"yes|no|too|very|much|many|few|some|any|all|every|each|other|another|"
    r"buy|buying|sell|selling|new|used|old|car|house|home|job|work|money|"
    r"life|health|family|future|business|invest|investment|risk|"
    r"week|month|year|day|hour|night)\b",
    re.IGNORECASE,
)


def detect_locale(question: str) -> str:
    """Return ``'en'`` or ``'es'``.

    Resolution order:
      1. ``HENGE_LOCALE`` env var, if set to ``en`` or ``es``.
      2. Heuristic on the question text — whichever language has more matches.
      3. Fallback ``'en'`` (default chrome language).
    """
    forced = os.environ.get("HENGE_LOCALE", "").strip().lower()
    if forced in ("en", "es"):
        return forced
    if not question:
        return "en"
    spanish_hits = len(_SPANISH_RE.findall(question))
    english_hits = len(_ENGLISH_RE.findall(question))
    return "es" if spanish_hits > english_hits else "en"


# Calibration for consensus_verdict() — provider-agnostic.
# Different embedding models baseline at different absolute distance ranges
# (text-embedding-3-small ≈ 0.6–0.7, voyage-3-large ≈ 0.05–0.10). What stays
# stable across providers is the *spread* across the 9 when they're clustered.
# We use σ over the 9 frame distances (not their absolute distance to centroid)
# to detect tight clustering, and a z-score-style σ-multiple to detect when
# the tenth has meaningfully separated from the cluster.
TIGHT_SIGMA = 0.03   # σ across the 9 frame distances below this → clustered
DISSENT_SIGMA = 3.0  # tenth ≥ this many σ above cluster mean → meaningful dissent

# CFI (Consensus Fragility Index) — pre-registered in docs/cfi-spec.md.
# Normalization constant: at k·σ above the cluster mean, the dissenter has
# pushed far enough that the consensus is treated as broken.
CFI_K = 6.0
CFI_FRAGILE_THRESHOLD = 0.33  # CFI < this → aligned-stable
CFI_BROKEN_THRESHOLD = 0.66   # informational only; v0.5 collapses ≥0.33 → aligned-fragile


TRANSLATIONS = {
    "en": {
        "page_title": "Henge · Disagreement Map",
        "screen_label": "Henge Report",
        "masthead_report": "Report",
        "masthead_index_btn": "Past reports",
        "masthead_index_aria": "View past reports",
        "theme_toggle_aria": "Toggle theme",
        "hero_h_a": "Nine advisors aligned.",
        "hero_h_b": "The tenth must dissent.",
        "hero_dek": "Your question runs through nine cognitive frames. We measure the consensus and force a tenth to disagree with rigor.",
        "stat_tenth_d": "Tenth · d",
        "stat_vs_centroid": "vs centroid",
        "stat_closest": "Closest",
        "stat_most_divergent": "Most divergent",
        "stat_verdict": "Verdict",
        "verdict_label_aligned": "Aligned",
        "verdict_label_fragile": "Fragile",
        "verdict_label_divided": "Divided",
        "verdict_sub_aligned": "consensus holds",
        "verdict_sub_fragile": "fragile consensus",
        "verdict_sub_divided": "no strong consensus",
        "verdict_text_aligned": "Advisors aligned — dissent sounds reasonable but consensus holds.",
        "verdict_text_fragile": "Strong but fragile consensus — the dissenter breaks it coherently.",
        "verdict_text_divided": "Advisors divided — there was no strong consensus to begin with.",
        "section01_eyebrow_prefix": "Report · #",
        "section01_h2_a": "The question ",
        "section01_h2_em": "the nine answered",
        "fig_label": "fig.01 · disagreement map",
        "fig_meta": "10 voices · MDS · cosine",
        "legend_tenth": "Tenth man",
        "legend_consensus": "Consensus frames",
        "map_help_aria": "How to read this map",
        "map_help_title": "How to read this map",
        "map_help_p1": "Each point is one advisor.",
        "map_help_p2": "The <strong>centroid</strong> at the center is the consensus zone — where the group converges.",
        "map_help_p3": "<strong>Closer</strong> to the center = more aligned with the rest.<br><strong>Farther</strong> = thinks differently.",
        "map_help_p4": "The <strong style=\"color: var(--midnight-navy); background: var(--chartreuse); padding: 0 4px; border-radius: 3px;\">10 · tenth-man</strong> point is the mandatory dissenter.",
        "map_help_p5": "Concentric rings mark equal distances from the centroid.",
        "footcell_tenth_d": "Tenth · d",
        "footcell_tenth_sub": "forced to dissent",
        "footcell_closest": "Closest",
        "footcell_divergent": "Most divergent (of the 9)",
        "footcell_spread": "Internal spread",
        "footcell_spread_sub": "σ across the 9 frames",
        "consensus_tag_prefix": "The consensus of the 9 · ",
        "consensus_lead": "What the nine advisors agree on, in one line:",
        "consensus_d_label_html": "max · vs centroid",
        "consensus_default_title": "What the nine agree on",
        "tenth_tag_label": "Tenth Man · forced dissent · method: steel-man",
        "tenth_lead": "The mandatory dissenter's strongest case against the nine:",
        "tenth_h3_a": "Why the nine ",
        "tenth_h3_em": "might be wrong",
        "tenth_d_label": "distance · centroid",
        "tenth_foot_left": "Generated under constraint · <b>steel-man mandatory</b>",
        "tenth_modes_h4": "Consensus failure modes",
        "section02_eyebrow": "Cognitive frames · 9 voices",
        "section02_h2_a": "The nine, ranked by ",
        "section02_h2_em": "distance to centroid",
        "section02_sub": "Each frame brings a different lens: empirical, systemic, historical, analogical. The closer to the centroid, the more it represents the consensus. The farther, the more it reasons alone.",
        "frames_head_idx": "Idx",
        "frames_head_frame": "Frame",
        "frames_head_status": "Status",
        "frames_head_lean": "Lean",
        "frames_head_distance": "Distance",
        "frames_head_d": "d",
        "flag_closest": "closest",
        "flag_farthest": "farthest",
        "hero_quote_nine_mark": "&#x03A3; &middot; The Nine",
        "hero_quote_tenth_mark_ord": "10",
        "hero_quote_tenth_mark_label": "The Tenth",
        "hero_quote_consensus_cite": "consensus",
        "hero_quote_tenth_cite": "forced dissent",
        "hero_verdict_consensus": "consensus",
        "svg_centroid": "CENTROID",
        "svg_steelman": "steel-man dissent",
        "guide_kicker": "Reading guide",
        "guide_title_a": "How to approach ",
        "guide_title_em": "this report",
        "guide_close": "Close",
        "guide_btn": "How do I read this?",
        "guide_aria": "How to read this report",
        "guide_rule_1": "<b>Start with the consensus, not the tenth.</b> It's what the 9 advisors agree on — the anchor of the decision.",
        "guide_rule_2": "<b>The 9 advisors aren't votes.</b> They're distinct lenses on the same problem. Read the differences between them, not the majority.",
        "guide_rule_3": "<b>The tenth is audit, not recommendation.</b> Its role is to attack the consensus to test if it holds. Sounding convincing is its job, not a signal it's right.",
        "guide_rule_4": "<b>Isolate the tenth's attacks, not its verdict.</b> Keep the questions it opens and evaluate each against your reality — discard its conclusion if it doesn't hold.",
        "guide_rule_5": "<b>Read the metrics.</b> High fragility + high tenth-man distance = weak consensus, dissent matters more. Low fragility = robust consensus, dissent is rhetorical.",
        "guide_rule_6": "<b>Apply the asymmetric test.</b> If the tenth names a risk you recognize as real in your life, add it even if you don't switch sides. If you don't recognize it, discard.",
        "guide_rule_7": "<b>You decide.</b> No advisor knows your full context. The report exposes tensions; the choice is yours.",
        "guide_rule_8": "<b>The meta-frame audits your question first.</b> If it suggests reformulating, the run short-circuits before the 9 advisors fire — read the suggested rewrite carefully, it usually sharpens the actual decision.",
        "guide_rule_9": "<b>Map points are coloured by model family.</b> Blue = Anthropic, green = OpenAI. If the dispersion comes from one family alone, the disagreement may be a model-style artefact more than a true disagreement on the substance.",
        "guide_foot": "Consensus protects against the obvious error. The tenth protects against the shared one. You need both lenses — and neither replaces your judgment.",
        "colophon_tagline": "Nine voices aligned aren't signal — just coherent noise.",
    },
    "es": {
        "page_title": "Henge · Mapa de desacuerdo",
        "screen_label": "Reporte Henge",
        "masthead_report": "Reporte",
        "masthead_index_btn": "Reportes anteriores",
        "masthead_index_aria": "Ver reportes anteriores",
        "theme_toggle_aria": "Cambiar tema",
        "hero_h_a": "Nueve consejeros alineados.",
        "hero_h_b": "El décimo debe disentir.",
        "hero_dek": "Tu pregunta corre por nueve marcos cognitivos. Medimos el consenso y obligamos a un décimo a discrepar con rigor.",
        "stat_tenth_d": "Décimo · d",
        "stat_vs_centroid": "vs centroide",
        "stat_closest": "Más cercano",
        "stat_most_divergent": "Más divergente",
        "stat_verdict": "Veredicto",
        "verdict_label_aligned": "Alineado",
        "verdict_label_fragile": "Frágil",
        "verdict_label_divided": "Disperso",
        "verdict_sub_aligned": "el consenso aguanta",
        "verdict_sub_fragile": "consenso frágil",
        "verdict_sub_divided": "sin consenso fuerte",
        "verdict_text_aligned": "Consejeros alineados — el disenso suena pero el consenso aguanta.",
        "verdict_text_fragile": "Consenso fuerte pero frágil — el disidente lo rompe coherentemente.",
        "verdict_text_divided": "Consejeros divididos — no había consenso fuerte para empezar.",
        "section01_eyebrow_prefix": "Reporte · #",
        "section01_h2_a": "La pregunta ",
        "section01_h2_em": "que los nueve respondieron",
        "fig_label": "fig.01 · mapa de desacuerdo",
        "fig_meta": "10 voces · MDS · cosine",
        "legend_tenth": "Décimo hombre",
        "legend_consensus": "Marcos del consenso",
        "map_help_aria": "Cómo leer este mapa",
        "map_help_title": "Cómo leer este mapa",
        "map_help_p1": "Cada punto es un consejero.",
        "map_help_p2": "El <strong>centroide</strong> al centro es la zona de consenso — donde el grupo converge.",
        "map_help_p3": "<strong>Más cerca</strong> del centro = más alineado con el resto.<br><strong>Más lejos</strong> = piensa distinto.",
        "map_help_p4": "El <strong style=\"color: var(--midnight-navy); background: var(--chartreuse); padding: 0 4px; border-radius: 3px;\">10 · décimo hombre</strong> es el disidente obligado.",
        "map_help_p5": "Los anillos concéntricos marcan distancias iguales al centroide.",
        "footcell_tenth_d": "Décimo · d",
        "footcell_tenth_sub": "obligado a discrepar",
        "footcell_closest": "Más cercano",
        "footcell_divergent": "Más divergente (de los 9)",
        "footcell_spread": "Spread interno",
        "footcell_spread_sub": "σ entre los 9 marcos",
        "consensus_tag_prefix": "El consenso de los 9 · ",
        "consensus_lead": "En lo que los nueve coinciden, en una línea:",
        "consensus_d_label_html": "max · vs centroide",
        "consensus_default_title": "Lo que los nueve coinciden",
        "tenth_tag_label": "Décimo Hombre · disidente obligado · método: steel-man",
        "tenth_lead": "El argumento más fuerte del disidente obligado contra los nueve:",
        "tenth_h3_a": "Por qué los nueve ",
        "tenth_h3_em": "podrían estar equivocados",
        "tenth_d_label": "distancia · centroide",
        "tenth_foot_left": "Generado bajo restricción · <b>steel-man obligatorio</b>",
        "tenth_modes_h4": "Modos de fallo del consenso",
        "section02_eyebrow": "Marcos cognitivos · 9 voces",
        "section02_h2_a": "Los nueve, ordenados por ",
        "section02_h2_em": "distancia al centroide",
        "section02_sub": "Cada marco aporta una lente: empírica, sistémica, histórica, analógica. Cuanto más cerca del centroide, más representa el consenso. Cuanto más lejos, más solo razona.",
        "frames_head_idx": "Idx",
        "frames_head_frame": "Marco",
        "frames_head_status": "Estado",
        "frames_head_lean": "Veredicto",
        "frames_head_distance": "Distancia",
        "frames_head_d": "d",
        "flag_closest": "más cercano",
        "flag_farthest": "más lejano",
        "hero_quote_nine_mark": "&#x03A3; &middot; Los Nueve",
        "hero_quote_tenth_mark_ord": "10",
        "hero_quote_tenth_mark_label": "El D&eacute;cimo",
        "hero_quote_consensus_cite": "consenso",
        "hero_quote_tenth_cite": "disenso obligado",
        "hero_verdict_consensus": "consenso",
        "svg_centroid": "CENTROIDE",
        "svg_steelman": "disenso steel-man",
        "guide_kicker": "Guía de lectura",
        "guide_title_a": "Cómo abordar ",
        "guide_title_em": "este reporte",
        "guide_close": "Cerrar",
        "guide_btn": "¿Cómo leer esto?",
        "guide_aria": "Cómo leer este reporte",
        "guide_rule_1": "<b>Empieza por el consenso, no por el décimo.</b> Es lo que los 9 consejeros creen en común — el ancla de la decisión.",
        "guide_rule_2": "<b>Los 9 consejeros no son votos.</b> Son lentes distintos sobre el mismo problema. Lee las diferencias entre ellos, no la mayoría.",
        "guide_rule_3": "<b>El décimo es auditoría, no recomendación.</b> Su rol es atacar el consenso para probar si aguanta. Sonar convincente es su trabajo, no señal de que tenga razón.",
        "guide_rule_4": "<b>Aísla los ataques del décimo, no su veredicto.</b> Quédate con las preguntas que abre y evalúa cada una contra tu realidad — descarta su conclusión si no resiste.",
        "guide_rule_5": "<b>Mira las métricas.</b> Fragilidad alta + distancia 10º alta = consenso débil, el disenso pesa más. Fragilidad baja = consenso robusto, el disenso es retórico.",
        "guide_rule_6": "<b>Aplica el test asimétrico.</b> Si el décimo nombra un riesgo que reconoces como real en tu vida, súmalo aunque no cambies de bando. Si no lo reconoces, descártalo.",
        "guide_rule_7": "<b>Tú decides.</b> Ningún consejero conoce tu contexto completo. El reporte expone tensiones; la elección es tuya.",
        "guide_rule_8": "<b>El meta-frame audita tu pregunta primero.</b> Si sugiere reformular, la corrida se corta antes de gastar en los 9 consejeros — lee la reformulación propuesta con cuidado, suele afilar la decisión real.",
        "guide_rule_9": "<b>Los puntos del mapa se colorean por familia de modelo.</b> Azul = Anthropic, verde = OpenAI. Si la dispersión viene de una sola familia, la divergencia puede ser un artefacto de estilo y no un desacuerdo de fondo.",
        "guide_foot": "El consenso protege contra el error obvio. El décimo protege contra el error compartido. Necesitas ambos lentes — y ninguno reemplaza tu juicio.",
        "colophon_tagline": "Nueve voces alineadas no son señal — son ruido coherente.",
    },
}


def t(locale: str, key: str) -> str:
    """Lookup a chrome string for the given locale, falling back to English."""
    return TRANSLATIONS.get(locale, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )


def compute_cfi(tenth_distance: float, frame_distances: list[float]) -> dict:
    """Consensus Fragility Index.

    CFI = clamp(0, 1, (d_tenth - μ) / (k · σ)) with k = ``CFI_K`` (= 6).

    σ is floored at 1e-6 to avoid division by zero on degenerate clusters.
    Bin assignment:

    - ``σ ≥ TIGHT_SIGMA``         → ``divided`` (overrides CFI; the 9 never
      reached real consensus)
    - ``CFI < CFI_FRAGILE_THRESHOLD`` (0.33)   → ``aligned-stable``
    - otherwise                                → ``aligned-fragile``

    The thresholds and constants are pre-registered in ``docs/cfi-spec.md``
    and ``WHITEPAPER.md``. Changing them requires a new pricing/spec version
    and a CHANGELOG entry — historical reports remain comparable only within
    the same spec version.
    """
    n = len(frame_distances)
    if n == 0:
        return {"cfi": None, "cfi_bin": "divided", "mu_9": None, "sigma_9": None}

    mu = sum(frame_distances) / n
    sigma = (sum((d - mu) ** 2 for d in frame_distances) / n) ** 0.5

    sigma_floor = max(sigma, 1e-6)
    cfi_raw = (tenth_distance - mu) / (CFI_K * sigma_floor)
    cfi = max(0.0, min(1.0, cfi_raw))

    if sigma >= TIGHT_SIGMA:
        cfi_bin = "divided"
    elif cfi < CFI_FRAGILE_THRESHOLD:
        cfi_bin = "aligned-stable"
    else:
        cfi_bin = "aligned-fragile"

    return {
        "cfi": round(cfi, 4),
        "cfi_bin": cfi_bin,
        "mu_9": round(mu, 4),
        "sigma_9": round(sigma, 4),
    }


def consensus_verdict(tenth_distance: float, frame_distances: list[float], locale: str = "en") -> dict:
    """Three-state classification of the consensus shape.

    Provider-agnostic: classification is driven by the *spread* across the 9
    (σ) and the tenth's z-score relative to the cluster mean, not by absolute
    distance to the centroid. This avoids the previous bug where openai's
    higher absolute-distance baseline (~0.65) made every run come back
    "divided" while voyage's tighter scale (~0.07) worked correctly.

    - aligned-stable:  9 advisors tight (σ < TIGHT_SIGMA), tenth's dissent is moderate.
    - aligned-fragile: 9 tight, but tenth is ≥ DISSENT_SIGMA σ above the cluster mean.
    - divided:         σ across the 9 is high — there was no strong consensus to break.
    """
    n = len(frame_distances)
    mean_d = sum(frame_distances) / n
    sigma = (sum((d - mean_d) ** 2 for d in frame_distances) / n) ** 0.5

    tight_nine = sigma < TIGHT_SIGMA
    if not tight_nine:
        return {
            "state": "divided",
            "label_short": "divided",
            "verdict": t(locale, "verdict_text_divided"),
        }
    sigma_floor = max(sigma, 1e-6)  # avoid div-by-zero on extremely tight clusters
    if (tenth_distance - mean_d) > DISSENT_SIGMA * sigma_floor:
        return {
            "state": "aligned-fragile",
            "label_short": "fragile consensus",
            "verdict": t(locale, "verdict_text_fragile"),
        }
    return {
        "state": "aligned-stable",
        "label_short": "aligned",
        "verdict": t(locale, "verdict_text_aligned"),
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


def _meta_card_html(meta_dict, locale: str = "es") -> str:
    """Render the meta-frame audit card. ``meta_dict`` may be None (legacy reports)."""
    if not meta_dict:
        return ""
    decision = meta_dict.get("decision_class", "unknown")
    urgency = meta_dict.get("urgency", "unknown")
    quality = meta_dict.get("question_quality", "unknown")
    rec = meta_dict.get("meta_recommendation", "proceed")
    reasoning = (meta_dict.get("reasoning") or "").strip()

    rec_label = {
        "proceed":               "Procede" if locale == "es" else "Proceed",
        "reformulate":           "Reformular" if locale == "es" else "Reformulate",
        "postpone":              "Postergar" if locale == "es" else "Postpone",
        "this-is-not-a-decision": "No es decisión" if locale == "es" else "Not a decision",
    }.get(rec, rec)

    title = "Auditoría de la pregunta" if locale == "es" else "Question audit"

    safe_decision = html_mod.escape(str(decision))
    safe_urgency = html_mod.escape(str(urgency))
    safe_quality = html_mod.escape(str(quality))
    safe_rec_label = html_mod.escape(rec_label)
    safe_rec = html_mod.escape(str(rec))
    safe_title = html_mod.escape(title)

    return (
        f'<section class="meta-card">'
        f'  <h3>{safe_title}</h3>'
        f'  <div class="meta-tags">'
        f'    <span class="tag tag-decision tag-{safe_decision}">{safe_decision}</span>'
        f'    <span class="tag tag-urgency tag-{safe_urgency}">{safe_urgency}</span>'
        f'    <span class="tag tag-quality tag-{safe_quality}">{safe_quality}</span>'
        f'    <span class="tag tag-rec tag-{safe_rec}">{safe_rec_label}</span>'
        f'  </div>'
        f'  <div class="meta-reasoning">{_md_to_html(reasoning)}</div>'
        f'</section>'
    )


def _informed_card_html(informed_dict, locale: str = "es") -> str:
    """Render the gpt-5 informed reconciliation card (Phase 5 dual tenth-man).

    Sits below the blind tenth-card. Shows what holds, what got revised, what
    was discarded as Opus bias. ``informed_dict`` may be None on legacy reports.
    """
    if not informed_dict:
        return ""
    text = (informed_dict.get("text") or "").strip()
    holds = informed_dict.get("what_holds") or []
    revised = informed_dict.get("what_revised") or []
    discarded = informed_dict.get("what_discarded") or []

    if not (text or holds or revised or discarded):
        return ""

    title = "Disenso refinado (cross-lab)" if locale == "es" else "Refined dissent (cross-lab)"
    holds_label = "Lo que se sostiene" if locale == "es" else "What holds"
    revised_label = "Lo que se revisa" if locale == "es" else "What gets revised"
    discarded_label = "Lo que se descarta" if locale == "es" else "What was lab bias"
    sub = (
        "gpt-5 audita el disenso ciego de Opus contra los 9 marcos."
        if locale == "es"
        else "gpt-5 audits Opus's blind dissent against the 9 advisors."
    )

    def _list_html(items, css_class):
        if not items:
            return ""
        lis = "".join(f"<li>{html_mod.escape(str(item))}</li>" for item in items)
        return f'<ul class="informed-list {css_class}">{lis}</ul>'

    text_html = _md_to_html(text) if text else ""

    return (
        '<article class="informed-card" id="informed">'
        f'  <header class="informed-top">'
        f'    <div class="informed-tag">cross-lab</div>'
        f'    <h3>{html_mod.escape(title)}</h3>'
        f'    <p class="informed-sub">{html_mod.escape(sub)}</p>'
        f'  </header>'
        f'  <div class="informed-body">'
        f'    {text_html}'
        f'    <div class="informed-grid">'
        f'      <div class="informed-col">'
        f'        <div class="ord">§ 01</div>'
        f'        <h5>{html_mod.escape(holds_label)}</h5>'
        f'        {_list_html(holds, "holds")}'
        f'      </div>'
        f'      <div class="informed-col">'
        f'        <div class="ord">§ 02</div>'
        f'        <h5>{html_mod.escape(revised_label)}</h5>'
        f'        {_list_html(revised, "revised")}'
        f'      </div>'
        f'      <div class="informed-col">'
        f'        <div class="ord">§ 03</div>'
        f'        <h5>{html_mod.escape(discarded_label)}</h5>'
        f'        {_list_html(discarded, "discarded")}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</article>'
    )


def _claims_panel_html(claims_list, locale: str = "es") -> str:
    """Render the Claims+support panel after the consensus.

    ``claims_list`` is a list of dicts with claim_text, claim_type,
    supporting_frames, contesting_frames, support_strength. May be None
    (legacy reports) — returns "" in that case.
    """
    if not claims_list:
        return ""

    title = "Claims del consensus + soporte" if locale == "es" else "Consensus claims + support"
    sub = (
        "Sonnet extrae proposiciones falsables del consensus; gpt-5 cross-lab verifica cada una contra los 9."
        if locale == "es"
        else "Sonnet extracts falsifiable claims from the consensus; gpt-5 cross-lab verifies each against the 9."
    )

    sup_label = "Sostienen" if locale == "es" else "Support"
    con_label = "Contradicen" if locale == "es" else "Contest"
    none_label = "—"

    strength_label = {
        "strong":      ("alto", "strong") if locale == "es" else ("strong", "strong"),
        "moderate":    ("moderado", "moderate") if locale == "es" else ("moderate", "moderate"),
        "weak":        ("débil", "weak") if locale == "es" else ("weak", "weak"),
        "unsupported": ("sin soporte", "unsupported") if locale == "es" else ("unsupported", "unsupported"),
    }
    type_label = {
        "factual":      "factual",
        "prescriptive": "prescriptiva" if locale == "es" else "prescriptive",
        "causal":       "causal",
        "unknown":      "?",
    }

    items_html = []
    for c in claims_list:
        text = html_mod.escape(str(c.get("claim_text", "")).strip())
        ctype = c.get("claim_type", "unknown")
        strength = c.get("support_strength", "unsupported")
        s_text, s_class = strength_label.get(strength, ("?", "unsupported"))
        sup = c.get("supporting_frames") or []
        con = c.get("contesting_frames") or []
        sup_n = len(sup)
        sup_names = ", ".join(html_mod.escape(s) for s in sup) if sup else none_label
        con_names = ", ".join(html_mod.escape(s) for s in con) if con else none_label

        items_html.append(
            f'<li class="claim-item claim-{html_mod.escape(s_class)}">'
            f'<div class="claim-row1">'
            f'<span class="claim-type">{html_mod.escape(type_label.get(ctype, "?"))}</span>'
            f'<span class="claim-strength claim-strength-{html_mod.escape(s_class)}">{html_mod.escape(s_text)}</span>'
            f'</div>'
            f'<p class="claim-text">{text}</p>'
            f'<div class="claim-meta">'
            f'<span class="claim-support"><b>{html_mod.escape(sup_label)}</b> ({sup_n}/9): {sup_names}</span>'
            f'<span class="claim-contest"><b>{html_mod.escape(con_label)}:</b> {con_names}</span>'
            f'</div>'
            f'</li>'
        )

    return (
        '<section class="claims-panel">'
        f'  <header class="claims-head">'
        f'    <h3>{html_mod.escape(title)}</h3>'
        f'    <p class="claims-sub">{html_mod.escape(sub)}</p>'
        f'  </header>'
        f'  <ul class="claims-list">{"".join(items_html)}</ul>'
        '</section>'
    )


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


_LEAN_MARKERS = re.compile(
    r"\*\*\s*(?:Conclusion|Conclusi[oó]n|Net lean|Lean|Bottom line|Veredicto|Net)\s*[:\.]?\s*\*\*\s*[:\.]?\s*",
    re.IGNORECASE,
)


def _extract_lean(text: str, max_chars: int = 90) -> str:
    """One-liner verdict for the frames-table Lean column.

    Heuristic: prefer the sentence after a ``**Conclusion:**`` / ``**Net lean:**``
    marker (the prompts close with these). Fall back to the first sentence of
    the last paragraph. Strip markdown emphasis, collapse whitespace, truncate at
    a sentence boundary close to ``max_chars`` so the row stays one line.
    """
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return ""

    candidate = ""
    for para in reversed(paragraphs):
        m = _LEAN_MARKERS.search(para)
        if m:
            candidate = para[m.end():].strip()
            break
    if not candidate:
        candidate = paragraphs[-1]

    # Take the first sentence
    sentence_end = re.search(r"[.!?](?:\s|$)", candidate)
    if sentence_end:
        candidate = candidate[: sentence_end.end()].strip()

    # Strip markdown emphasis ** / __ / * / _
    candidate = re.sub(r"\*\*(.+?)\*\*", r"\1", candidate)
    candidate = re.sub(r"__(.+?)__", r"\1", candidate)
    candidate = re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", r"\1", candidate)
    candidate = re.sub(r"(?<!_)_(?!_)([^_]+)_(?!_)", r"\1", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    if len(candidate) > max_chars:
        truncated = candidate[:max_chars]
        # Prefer cutting at the last sentence-end or comma in the truncation window
        cut = max(truncated.rfind("."), truncated.rfind(","), truncated.rfind(";"))
        if cut > max_chars * 0.6:
            candidate = truncated[: cut + 1].rstrip(",;") + ("" if truncated[cut] == "." else "&hellip;")
        else:
            candidate = truncated.rsplit(" ", 1)[0] + "&hellip;"
    return candidate


def _stddev(values):
    """Population standard deviation. Returns 0.0 for empty / single-element input."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ───────── Phase 9: family colors ─────────
# Anthropic stays in the existing midnight-navy palette (matches consensus card).
# OpenAI gets a distinct green that reads as "different lab" without screaming.
_FAMILY_COLOR = {
    "anthropic": "var(--midnight-navy)",
    "openai":    "#1b8c5a",  # forest green, accessible on both light + dark
}


def _family_for_model(model_id: str) -> str:
    if not model_id:
        return "anthropic"
    if model_id.startswith("openai/"):
        return "openai"
    return "anthropic"


def _color_for_frame(frame_name: str) -> str:
    """Return CSS color for a frame's assigned canonical model family."""
    model_id = FRAME_MODEL_MAP.get(frame_name, "")
    family = _family_for_model(model_id)
    return _FAMILY_COLOR.get(family, "var(--midnight-navy)")


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


def _build_map_svg(coords_2d, frames, distances, max_frame_dist, min_frame_dist, locale="en"):
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
        f'font-size="11" fill="#6b7184" letter-spacing="1">{t(locale, "svg_centroid")}</text>',
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
    parts.append('<g font-family="Source Sans 3, Source Sans Pro, sans-serif" font-size="13" fill="#1b2540">')
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
            suffix = " · " + t(locale, "flag_closest")
        elif is_farthest:
            suffix = " · " + t(locale, "flag_farthest")
        node_color = _color_for_frame(frame)
        parts.append('<g>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{marker_r}" fill="{node_color}"/>')
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
        f'font-family="Source Sans 3, Source Sans Pro, sans-serif" font-weight="500" font-size="13" '
        f'fill="#1b2540" letter-spacing="2">10 · TENTH-MAN</text>'
    )
    parts.append(
        f'<text x="{tx:.1f}" y="{ty + label_offset + 16:.1f}" text-anchor="middle" '
        f'font-family="JetBrains Mono, monospace" font-size="11" fill="#1b2540">'
        f'{t(locale, "svg_steelman")}</text>'
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


def _build_frame_card(frame, response, status, distance, max_dist, idx_str,
                      is_open=False, lean=""):
    """Build a frame-list article in the v3 table layout.

    Open by default for the closest frame to the centroid. ``lean`` is the
    one-line verdict shown in the Lean column (extracted upstream from the
    frame response). Failed frames render an italic ``frame failed`` line so
    the row keeps a meaningful slot.
    """
    body = _md_to_html(response)
    bar_pct = min(100, (distance / max_dist) * 100) if max_dist > 0 else 0
    open_class = " open" if is_open else ""
    if status != "ok":
        lean_html = "<em>frame failed</em>"
    elif lean:
        lean_html = html_mod.escape(lean).replace("&amp;hellip;", "&hellip;")
    else:
        lean_html = ""
    return f"""
    <article class="frame{open_class}" data-frame="{html_mod.escape(frame)}">
      <div class="frame-row">
        <span class="f-idx">#{idx_str}</span>
        <span class="f-name">{html_mod.escape(frame)}</span>
        <span class="f-lean">{lean_html}</span>
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
                                is_open=False, flag=None, flag_kind=None, lean=""):
    """Same as _build_frame_card but appends a closest/farthest chip to the name.

    ``flag`` is the localized label ("closest" / "más cercano" / ...).
    ``flag_kind`` is the canonical key ("closest" / "farthest") used to pick
    the chip variant in CSS — solid chartreuse for closest, outlined for farthest.
    """
    body = _md_to_html(response)
    bar_pct = min(100, (distance / max_dist) * 100) if max_dist > 0 else 0
    open_class = " open" if is_open else ""
    if flag:
        kind_class = f" {html_mod.escape(flag_kind)}" if flag_kind else ""
        flag_html = f' <span class="f-mark{kind_class}">{html_mod.escape(flag)}</span>'
    else:
        flag_html = ""
    # Phase 9: canonical model-id chip
    chip_html = ""
    cid = FRAME_MODEL_MAP.get(frame, "")
    if cid:
        fam = _family_for_model(cid)
        chip_html = f'<span class="frame-model-chip frame-model-{fam}">{html_mod.escape(cid)}</span>'
    if status != "ok":
        lean_html = "<em>frame failed</em>"
    elif lean:
        lean_html = html_mod.escape(lean).replace("&amp;hellip;", "&hellip;")
    else:
        lean_html = ""
    return f"""
    <article class="frame{open_class}" data-frame="{html_mod.escape(frame)}">
      <div class="frame-row">
        <span class="f-idx">#{idx_str}</span>
        <span class="f-name">{html_mod.escape(frame)}{flag_html}{chip_html}</span>
        <span class="f-lean">{lean_html}</span>
        <span class="f-bar"><i style="width:{bar_pct:.0f}%"></i></span>
        <span class="f-d">d <b>{distance:.3f}</b></span>
        <span class="f-caret">›</span>
      </div>
      <div class="f-body">
        {body}
      </div>
    </article>
    """


def render(question, results, coords_2d, distances, provider, model, cost_estimate_usd, consensus=None, cfi_data=None, meta_frame=None, informed=None, claims=None):
    """Render the TenthAI/Antimetal-style disagreement report. Returns full HTML.

    Persistence and browser-open are handled by the caller (server.py orchestrates
    storage.write_record + webbrowser.open). Keeping render() pure makes it easy
    to test, embed, and post-process.

    Layout order: masthead → hero (painting + 4 stats) → 01 Report (question +
    map card + optional consensus card + tenth-man card) → 02 Frames (frames
    table) → colophon.
    """
    locale = detect_locale(question)
    meta_html = _meta_card_html(meta_frame, locale=locale)
    informed_card_html = _informed_card_html(informed, locale=locale) if informed else ""
    claims_panel_html = _claims_panel_html(claims, locale=locale) if claims else ""

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

    verdict = consensus_verdict(tenth_distance, frame_distances, locale=locale)
    fragility_text = verdict["verdict"]
    verdict_short = verdict["label_short"]
    verdict_state = verdict["state"]
    # CFI surfaced on the hero card. Caller (server) may pre-compute it from
    # the un-substituted distance list for accuracy when some frames failed;
    # fall back to local computation otherwise.
    if cfi_data is None:
        cfi_data = compute_cfi(tenth_distance, frame_distances)
    cfi_value = cfi_data.get("cfi")
    cfi_sigma = cfi_data.get("sigma_9")
    if cfi_value is not None and cfi_sigma is not None:
        hero_cfi_line = f"CFI {cfi_value:.2f} · σ₉ {cfi_sigma:.3f}"
    else:
        hero_cfi_line = ""
    # Hero verdict cell — short editorial label, localized
    hero_verdict_label = {
        "aligned-stable": t(locale, "verdict_label_aligned"),
        "aligned-fragile": t(locale, "verdict_label_fragile"),
        "divided": t(locale, "verdict_label_divided"),
    }.get(verdict_state, "—")
    hero_verdict_sub = {
        "aligned-stable": t(locale, "verdict_sub_aligned"),
        "aligned-fragile": t(locale, "verdict_sub_fragile"),
        "divided": t(locale, "verdict_sub_divided"),
    }.get(verdict_state, "")

    # Map SVG (real MDS coords scaled into the v3 viewBox)
    map_svg = _build_map_svg(
        coords_2d=coords_2d,
        frames=frames,
        distances=distances,
        max_frame_dist=max_frame_distance,
        min_frame_dist=min_frame_distance,
        locale=locale,
    )

    # Phase 9: family-color legend below the SVG
    map_legend_html = (
        '<div class="map-legend">'
        f'  <span class="legend-dot" style="background: {_FAMILY_COLOR["anthropic"]};"></span>Anthropic'
        f'  <span class="legend-dot" style="background: {_FAMILY_COLOR["openai"]};"></span>OpenAI'
        '  <span class="legend-dot legend-tenth"></span>Tenth-man (blind)'
        '</div>'
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
            flag=(t(locale, "flag_closest") if i == closest_frame_idx
                  else t(locale, "flag_farthest") if i == most_divergent_idx
                  else None),
            flag_kind=("closest" if i == closest_frame_idx
                       else "farthest" if i == most_divergent_idx
                       else None),
            lean=_extract_lean(responses[i]) if statuses[i] == "ok" else "",
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
            f'<h4><span class="ord">§ 4</span>{t(locale, "tenth_modes_h4")}</h4>'
            '<div class="modes">' + "".join(mode_cards) + '</div>'
        )
    else:
        tenth_modes_html = ""

    # Consensus block (optional). v3-styled card placed before the tenth-man.
    consensus_block_html = ""
    consensus_title = ""
    if consensus:
        consensus_title, consensus_body_md = _split_consensus_title(consensus)
        if not consensus_title:
            consensus_title = t(locale, "consensus_default_title")
        consensus_html = _md_to_html(consensus_body_md)
        consensus_block_html = f"""
    <article class="consensus-card" id="consensus">
      <header class="consensus-top">
        <div>
          <div class="consensus-tag"><span class="d"></span>{t(locale, "consensus_tag_prefix")}{html_mod.escape(verdict_short.upper())}</div>
          <p class="consensus-lead">{t(locale, "consensus_lead")}</p>
          <h3>{html_mod.escape(consensus_title)}</h3>
        </div>
        <div class="consensus-d">
          <b>{max_frame_distance:.3f}</b>
          {t(locale, "consensus_d_label_html")}
        </div>
      </header>
      <div class="consensus-body">
        {consensus_html}
      </div>
    </article>
"""

    # Hero quote teasers — derived from the consensus title and the tenth response.
    # Both are shown on the hero panels and link to their full sections below.
    nine_lean = consensus_title or t(locale, "consensus_default_title")
    tenth_lean_raw = _extract_lean(responses[9], max_chars=110)
    tenth_lean_html = html_mod.escape(tenth_lean_raw).replace("&amp;hellip;", "&hellip;")

    timestamp = datetime.now().strftime("%Y·%m·%d %H:%M CLT")
    timestamp_short = datetime.now().strftime("%Y·%m·%d")
    report_id = datetime.now().strftime("%H%M")
    question_safe = html_mod.escape(question)

    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{t(locale, "page_title")}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
  /* Resolve theme before first paint to avoid FOUC.
     Stored preference wins; otherwise follow system. */
  (function() {{
    var stored = null;
    try {{ stored = localStorage.getItem('henge-theme'); }} catch (e) {{}}
    var theme = (stored === 'light' || stored === 'dark')
      ? stored
      : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  }})();
</script>
<style>
  :root{{
    --midnight-navy: #1b2540;
    --deep-cosmos: #001033;
    --chartreuse: #d0f100;
    --ice-veil: #e0f6ff;
    --ghost-canvas: #f8f9fc;
    --pure: #ffffff;
    --ash: #7c8293;
    --storm: #596075;
    --fog: #b1b5c0;

    /* Tinted whites for use on dark surfaces */
    --on-dark:        #fafeff;
    --on-dark-92:     rgba(224,246,255,0.92);
    --on-dark-78:     rgba(224,246,255,0.78);
    --on-dark-62:     rgba(224,246,255,0.62);
    --on-dark-32:     rgba(224,246,255,0.32);
    --on-dark-border-strong: rgba(224,246,255,0.18);

    /* Translucent surfaces */
    --surface-glass-08:   rgba(255,255,255,0.08);
    --surface-glass-soft: rgba(255,255,255,0.55);

    /* Border/rule semantic aliases (for new components) */
    --border-subtle: rgba(0,39,80,0.06);
    --border-rule:   rgba(0,39,80,0.08);

    /* Chartreuse alpha ramp */
    --chartreuse-04: rgba(208,241,0,0.04);
    --chartreuse-10: rgba(208,241,0,0.10);
    --chartreuse-14: rgba(208,241,0,0.14);
    --chartreuse-18: rgba(208,241,0,0.18);
    --chartreuse-20: rgba(208,241,0,0.20);
    --chartreuse-25: rgba(208,241,0,0.25);
    --chartreuse-28: rgba(208,241,0,0.28);
    --chartreuse-32: rgba(208,241,0,0.32);
    --chartreuse-55: rgba(208,241,0,0.55);

    /* Glow — used on tenth-quote panel */
    --glow-chartreuse:       0 0 0 1px var(--chartreuse-20), 0 0 24px var(--chartreuse-18), 0 0 60px var(--chartreuse-10);
    --glow-chartreuse-hover: 0 0 0 1px var(--chartreuse-32), 0 0 28px var(--chartreuse-28), 0 0 70px var(--chartreuse-14);

    /* Map line tint (midnight-navy alpha) */
    --navy-08: rgba(27,37,64,0.08);

    --shadow-xl: rgba(0,39,80,0.03) 0 56px 72px -16px, rgba(0,39,80,0.03) 0 32px 32px -16px, rgba(0,39,80,0.04) 0 6px 12px -3px, rgba(0,39,80,0.04) 0 0 0 1px;
    --shadow-md: rgba(0,39,80,0.08) 0 6px 16px -3px, rgba(0,39,80,0.04) 0 0 0 1px;
    --shadow-subtle: rgba(255,255,255,0.72) 0 1px 1px 0 inset, rgba(4,33,80,0.02) 0 8px 16px 0, rgba(4,33,80,0.03) 0 4px 12px 0, rgba(4,33,80,0.06) 0 1px 2px 0, rgba(4,33,80,0.04) 0 0 0 1px;
    --ring-subtle: rgba(0,39,80,0.04) 0 0 0 1px;
    --ring-rule:   rgba(0,39,80,0.06) 0 0 0 1px;

    --sans:  'Source Sans 3', 'Source Sans Pro', ui-sans-serif, system-ui, sans-serif;
    --serif: 'Fraunces', Georgia, serif;
    --mono:  'JetBrains Mono', ui-monospace, monospace;

    color-scheme: light dark;
  }}

  /* Dark mode — token flips live alongside :root above; component overrides
     are at the bottom of this stylesheet so they win over base declarations. */

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
    gap: 16px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border-rule);
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
    color: var(--storm); letter-spacing: 0.02em;
  }}
  .mast-meta b{{ color: var(--midnight-navy); font-weight: 500; }}
  .mast-meta .sep{{ color: var(--fog); margin: 0 8px; }}
  .mast-index-btn{{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 999px;
    border: 1px solid rgba(0,39,80,0.15);
    background: white;
    color: var(--midnight-navy);
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em;
    text-transform: uppercase;
    text-decoration: none;
    transition: background 120ms ease, border-color 120ms ease;
  }}
  .mast-index-btn:hover{{
    background: var(--midnight-navy);
    color: var(--chartreuse);
    border-color: var(--midnight-navy);
  }}
  .mast-index-btn svg{{ flex-shrink: 0; }}
  .mast-actions{{ display: inline-flex; align-items: center; gap: 8px; }}

  /* Hero */
  .hero{{
    max-width: 1200px;
    margin: 24px auto 0;
    border-radius: 20px;
    overflow: hidden;
    position: relative;
    box-shadow: var(--shadow-md);
    color: var(--on-dark);
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
    color: var(--on-dark-92);
    letter-spacing: -0.09px;
  }}

  /* Hero verdict eyebrow */
  .hero-verdict{{
    display: inline-flex; gap: 8px; align-items: baseline;
    font-family: var(--mono); font-size: 11px;
    color: var(--on-dark-78);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 14px;
  }}
  .hero-verdict b{{ color: var(--chartreuse); font-weight: 500; }}
  .hero-verdict .sep{{ color: var(--on-dark-32); }}

  /* Hero quote panels — translucent glass over the painting */
  .hero-quotes{{
    display: grid;
    gap: 12px;
  }}
  .hero-quote{{
    display: block;
    background: var(--surface-glass-08);
    backdrop-filter: blur(14px) saturate(120%);
    -webkit-backdrop-filter: blur(14px) saturate(120%);
    border: 1px solid var(--on-dark-border-strong);
    border-radius: 16px;
    padding: 14px 18px;
    text-decoration: none;
    color: inherit;
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
  }}
  .hero-quote:hover{{ transform: translateY(-1px); }}
  .hero-quote:focus-visible{{ outline: 2px solid var(--chartreuse); outline-offset: 3px; }}
  .hero-quote.consensus:hover{{ border-color: var(--on-dark-32); }}
  .hero-quote.tenth{{
    background: linear-gradient(180deg, var(--chartreuse-04), var(--surface-glass-08));
    border-color: var(--chartreuse-55);
    box-shadow: var(--glow-chartreuse);
  }}
  .hero-quote.tenth:hover{{
    border-color: var(--chartreuse);
    box-shadow: var(--glow-chartreuse-hover);
  }}
  .hero-quote .mark{{
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 10px;
    color: var(--on-dark-78);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 8px;
  }}
  .hero-quote.consensus .mark .d{{
    width: 6px; height: 6px; border-radius: 9999px;
    background: var(--on-dark-32);
  }}
  .hero-quote.tenth .mark .d{{
    width: 7px; height: 7px; border-radius: 9999px;
    background: var(--chartreuse);
    box-shadow: 0 0 0 2px var(--chartreuse-25);
  }}
  .hero-quote.tenth .mark .ord{{ color: var(--chartreuse); font-weight: 500; }}
  .hero-q{{
    margin: 0 0 8px;
    font-family: var(--serif); font-weight: 400;
    font-size: 16px; line-height: 1.3;
    letter-spacing: -0.16px;
    color: var(--on-dark);
    text-wrap: balance;
    quotes: none;
  }}
  .hero-q em{{ font-style: italic; color: var(--chartreuse); }}
  .hero-quote .cite{{
    font-family: var(--mono); font-size: 9.5px;
    color: var(--on-dark-62);
    letter-spacing: 0.02em;
    text-transform: uppercase;
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
    color: var(--storm);
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
    border-bottom: 1px solid var(--border-subtle);
  }}
  .map-top .l{{ display:flex; align-items:center; gap: 14px; }}
  .map-top .pill{{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--ghost-canvas);
    border-radius: 9999px;
    padding: 5px 12px;
    font-family: var(--mono); font-size: 11px;
    color: var(--midnight-navy);
    box-shadow: var(--ring-subtle);
  }}
  .map-top .pill .d{{ width: 6px; height: 6px; border-radius: 9999px; background: var(--chartreuse); }}
  .map-top .meta{{
    font-family: var(--mono); font-size: 12px; color: var(--storm);
  }}
  .legend-row{{ display:flex; gap: 16px; align-items: center; font-size: 13px; color: var(--storm); }}
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
    border-top: 1px solid var(--border-subtle);
  }}
  .map-foot .cell{{
    padding: 16px 20px;
    border-right: 1px solid var(--border-subtle);
  }}
  .map-foot .cell:last-child{{ border-right: 0; }}
  .map-foot .cell .lbl{{
    font-family: var(--mono); font-size: 11px;
    color: var(--storm); text-transform: uppercase;
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
    font-size: 13px; color: var(--storm);
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
    border-bottom: 1px solid var(--border-subtle);
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
    background: var(--surface-glass-soft);
    backdrop-filter: blur(4px);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: var(--ring-rule);
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
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
    column-rule: 1px solid var(--border-subtle);
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
    color: var(--storm);
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
    box-shadow: var(--ring-subtle);
    font-family: var(--serif); font-weight: 400;
    font-size: 18px; line-height: 1.35; letter-spacing: -0.18px;
    color: var(--midnight-navy);
    break-inside: avoid-column;
  }}

  /* Meta-frame audit card */
  .meta-card{{
    margin: 16px auto 24px;
    max-width: 880px;
    padding: 18px 22px;
    border: 1px solid var(--border-rule);
    border-radius: 12px;
    background: var(--surface-glass-soft);
    box-shadow: var(--ring-rule);
  }}
  .meta-card h3{{ margin: 0 0 10px; font-family: var(--sans); font-size: 12px; font-weight: 600; color: var(--storm); text-transform: uppercase; letter-spacing: 0.06em; }}
  .meta-tags{{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }}
  .meta-card .tag{{ display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 500; background: var(--chartreuse-10); color: var(--midnight-navy); border: 1px solid var(--chartreuse-25); font-family: var(--sans); }}
  .meta-card .tag-rec{{ background: var(--chartreuse-25); border-color: var(--chartreuse-55); }}
  .meta-reasoning{{ font-size: 14px; line-height: 1.55; color: var(--storm); font-family: var(--sans); }}
  .meta-reasoning p{{ margin: 0 0 10px; }}
  [data-theme="dark"] .meta-card{{ background: var(--surface-glass-08); border-color: var(--on-dark-border-strong); }}
  [data-theme="dark"] .meta-card h3{{ color: var(--on-dark-78); }}
  [data-theme="dark"] .meta-card .tag{{ background: var(--chartreuse-14); color: var(--on-dark); border-color: var(--chartreuse-32); }}
  [data-theme="dark"] .meta-reasoning{{ color: var(--on-dark-92); }}

  /* Tenth-man feature card */
  .tenth-card{{
    background: var(--pure);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    overflow: hidden;
    margin-top: 24px;
    position: relative;
  }}
  .tenth-top{{
    display: grid; grid-template-columns: 1fr auto;
    align-items: start; gap: 24px;
    padding: 24px 28px;
    border-bottom: 1px solid var(--border-subtle);
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
    box-shadow: var(--ring-subtle);
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
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
  }}
  .tenth-body h4{{
    margin: 24px 0 8px;
    font-family: var(--mono); font-weight: 500;
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--storm);
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
    box-shadow: var(--ring-subtle);
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
    box-shadow: var(--ring-subtle);
  }}
  .mode .ord{{
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
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
    border-top: 1px solid var(--border-subtle);
    background: var(--ghost-canvas);
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 12px;
    color: var(--storm);
  }}
  .tenth-foot b{{ color: var(--midnight-navy); }}

  /* Informed dissent card (Phase 5 cross-lab reconciliation) */
  .informed-card{{
    margin: 16px 0 0;
    padding: 20px 22px;
    border: 1px solid var(--border-rule);
    border-radius: 14px;
    background: var(--surface-glass-soft);
    box-shadow: var(--ring-rule);
  }}
  .informed-top{{ margin-bottom: 14px; }}
  .informed-tag{{
    display: inline-block; padding: 2px 8px; margin-bottom: 6px;
    font-size: 10px; font-family: var(--mono); letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--midnight-navy);
    background: var(--chartreuse); border-radius: 999px;
  }}
  .informed-top h3{{ margin: 0 0 4px; font-family: var(--serif); font-size: 22px; font-weight: 500; color: var(--midnight-navy); }}
  .informed-sub{{ margin: 0; font-size: 13px; color: var(--storm); font-family: var(--sans); }}
  .informed-body{{ font-family: var(--sans); }}
  .informed-body p{{ margin: 0 0 12px; line-height: 1.55; color: var(--midnight-navy); }}
  .informed-grid{{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px;
  }}
  .informed-col{{
    background: var(--ghost-canvas);
    border-radius: 16px;
    padding: 18px;
    box-shadow: var(--ring-subtle);
  }}
  .informed-col .ord{{
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .informed-col h5{{
    margin: 8px 0 10px;
    font-family: var(--sans); font-weight: 480;
    font-size: 17px; letter-spacing: -0.09px;
    color: var(--midnight-navy);
  }}
  .informed-list{{ margin: 0; padding-left: 16px; font-size: 13px; line-height: 1.45; color: var(--midnight-navy); }}
  .informed-list li{{ margin-bottom: 6px; }}
  .informed-list.discarded li{{ color: var(--storm); text-decoration: line-through; opacity: 0.75; }}
  @media (max-width: 720px){{ .informed-grid{{ grid-template-columns: 1fr; }} }}
  [data-theme="dark"] .informed-card{{ background: var(--surface-glass-08); border-color: var(--on-dark-border-strong); }}
  [data-theme="dark"] .informed-top h3{{ color: var(--on-dark); }}
  [data-theme="dark"] .informed-sub{{ color: var(--on-dark-78); }}
  [data-theme="dark"] .informed-body p{{ color: var(--on-dark-92); }}
  [data-theme="dark"] .informed-col h5{{ color: var(--on-dark); }}
  [data-theme="dark"] .informed-list{{ color: var(--on-dark-92); }}
  [data-theme="dark"] .informed-list.discarded li{{ color: var(--on-dark-62); }}

  /* Claims panel */
  .claims-panel{{
    margin: 16px 0 0;
    padding: 18px 22px;
    border: 1px solid var(--border-rule);
    border-radius: 14px;
    background: var(--surface-glass-soft);
    box-shadow: var(--ring-rule);
    font-family: var(--sans);
  }}
  .claims-head h3{{ margin: 0 0 4px; font-family: var(--serif); font-size: 20px; font-weight: 500; color: var(--midnight-navy); }}
  .claims-sub{{ margin: 0 0 12px; font-size: 13px; color: var(--storm); }}
  .claims-list{{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }}
  .claim-item{{
    padding: 10px 14px; border: 1px solid var(--border-subtle); border-radius: 10px;
    background: white;
  }}
  .claim-row1{{ display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }}
  .claim-type{{
    font-size: 10px; font-family: var(--mono); text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--storm);
    padding: 2px 6px; border: 1px solid var(--border-subtle); border-radius: 999px;
  }}
  .claim-strength{{
    font-size: 10px; font-family: var(--mono); text-transform: uppercase;
    letter-spacing: 0.06em; padding: 2px 8px; border-radius: 999px;
  }}
  .claim-strength-strong{{      background: rgba(0,150,80,0.12);   color: #047a47;  border: 1px solid rgba(0,150,80,0.30); }}
  .claim-strength-moderate{{    background: var(--chartreuse-14);  color: var(--midnight-navy); border: 1px solid var(--chartreuse-32); }}
  .claim-strength-weak{{        background: rgba(220,140,0,0.10);  color: #a36800;  border: 1px solid rgba(220,140,0,0.30); }}
  .claim-strength-unsupported{{ background: rgba(200,40,40,0.10);  color: #b32424;  border: 1px solid rgba(200,40,40,0.32); }}
  .claim-text{{ margin: 0 0 8px; font-size: 14px; line-height: 1.5; color: var(--midnight-navy); }}
  .claim-meta{{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--storm); }}
  [data-theme="dark"] .claims-panel{{ background: var(--surface-glass-08); border-color: var(--on-dark-border-strong); }}
  [data-theme="dark"] .claims-head h3{{ color: var(--on-dark); }}
  [data-theme="dark"] .claims-sub{{ color: var(--on-dark-78); }}
  [data-theme="dark"] .claim-item{{ background: rgba(0,0,0,0.20); border-color: var(--on-dark-border-strong); }}
  [data-theme="dark"] .claim-text{{ color: var(--on-dark-92); }}
  [data-theme="dark"] .claim-meta{{ color: var(--on-dark-78); }}
  [data-theme="dark"] .claim-type{{ color: var(--on-dark-78); border-color: var(--on-dark-border-strong); }}

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
    border-bottom: 1px solid var(--border-subtle);
    background: var(--ghost-canvas);
    font-family: var(--mono); font-size: 11px;
    color: var(--storm);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .frame{{
    border-bottom: 1px solid var(--border-subtle);
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
    font-family: var(--mono); font-size: 12px; color: var(--storm);
  }}
  .f-name{{
    display: inline-flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
    font-family: var(--sans); font-weight: 480;
    font-size: 17px; letter-spacing: -0.09px;
    color: var(--midnight-navy);
  }}
  .f-name .f-mark{{
    font-family: var(--mono); font-size: 9.5px; font-weight: 500;
    color: var(--midnight-navy);
    background: var(--chartreuse);
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 2px 7px; border-radius: 9999px;
    transform: translateY(-1px);
  }}
  .f-name .f-mark.farthest{{
    background: transparent; color: var(--storm);
    box-shadow: var(--ring-rule);
  }}
  .f-lean{{
    font-family: var(--serif); font-style: italic;
    font-weight: 400; font-size: 14.5px;
    line-height: 1.35; letter-spacing: -0.09px;
    color: var(--storm);
    max-width: 38ch;
    text-wrap: balance;
  }}
  .f-lean strong{{ color: var(--midnight-navy); font-weight: 500; font-style: normal; }}
  .f-bar{{
    height: 6px; background: var(--navy-08);
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
    color: var(--storm);
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
    color: var(--storm);
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
    color: var(--storm);
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

  /* Phase 9: family-color legend below the SVG */
  .map-legend{{
    display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
    font-size: 12px; font-family: var(--mono); color: var(--storm);
    margin: 8px 0 0; padding: 8px 28px 0; letter-spacing: 0.04em;
  }}
  .map-legend .legend-dot{{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
  }}
  .map-legend .legend-tenth{{ background: var(--chartreuse); border: 1px solid var(--midnight-navy); }}
  [data-theme="dark"] .map-legend{{ color: var(--on-dark-78); }}
  [data-theme="dark"] .map-legend .legend-tenth{{ border-color: var(--on-dark); }}

  /* Phase 9: model-family chip on frame cards */
  .frame-model-chip{{
    display: inline-block; padding: 2px 8px; margin-left: 8px;
    font-size: 10px; font-family: var(--mono); letter-spacing: 0.04em;
    border-radius: 999px; vertical-align: middle;
  }}
  .frame-model-anthropic{{ background: rgba(27,37,64,0.08); color: var(--midnight-navy); }}
  .frame-model-openai{{    background: rgba(27,140,90,0.10); color: #1b8c5a; }}
  [data-theme="dark"] .frame-model-anthropic{{ background: var(--surface-glass-08); color: var(--on-dark-92); }}
  [data-theme="dark"] .frame-model-openai{{    background: rgba(27,140,90,0.18); color: #4dd49a; }}

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
    border-top: 1px solid var(--border-rule);
    background: var(--ghost-canvas);
  }}
  .guide-panel .kicker{{
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--storm);
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
    border-top: 1px solid var(--border-subtle);
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
    color: var(--storm);
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
    color: var(--storm);
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
    border-top: 1px solid var(--border-rule);
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    font-family: var(--mono); font-size: 12px;
    color: var(--storm);
  }}
  footer.colophon b{{ color: var(--midnight-navy); font-weight: 500; }}
  footer.colophon .center{{ text-align: center; font-style: italic; font-family: var(--serif); }}
  footer.colophon .right{{ text-align: right; }}

  /* === Dark mode ===
     Activated by JS setting [data-theme="dark"] on <html>. The inline script
     in <head> resolves the effective theme (stored preference > system) before
     first paint to avoid FOUC. Fallback when JS is disabled: page renders in
     light always (acceptable trade-off for static reports). */
  :root[data-theme="dark"] {{
    --ghost-canvas:  #0b1124;
    --pure:          #14192a;
    --ice-veil:      #0d1734;
    --midnight-navy: #e6eefb;     /* primary text only */
    --ash:           #6b7184;
    --storm:         #b1b5c0;
    --fog:           #4a5168;

    --on-dark-border-strong: rgba(255,255,255,0.14);
    --surface-glass-soft:    rgba(20,25,42,0.55);

    --border-faint:  rgba(255,255,255,0.04);
    --border-subtle: rgba(255,255,255,0.06);
    --border-rule:   rgba(255,255,255,0.10);
    --border-strong: rgba(255,255,255,0.14);

    --navy-08:     rgba(255,255,255,0.10);
    --ring-subtle: rgba(255,255,255,0.06) 0 0 0 1px;
    --ring-rule:   rgba(255,255,255,0.08) 0 0 0 1px;

    --shadow-xl:     rgba(0,0,0,0.45) 0 56px 72px -16px, rgba(0,0,0,0.40) 0 32px 32px -16px, rgba(0,0,0,0.35) 0 6px 12px -3px, rgba(255,255,255,0.05) 0 0 0 1px;
    --shadow-md:     rgba(0,0,0,0.35) 0 6px 16px -3px, rgba(255,255,255,0.05) 0 0 0 1px;
    --shadow-subtle: rgba(255,255,255,0.04) 0 1px 1px 0 inset, rgba(0,0,0,0.30) 0 8px 16px 0, rgba(0,0,0,0.25) 0 4px 12px 0, rgba(0,0,0,0.20) 0 1px 2px 0, rgba(255,255,255,0.05) 0 0 0 1px;
  }}

  /* Components that use --midnight-navy AS A BACKGROUND need a fixed
     dark navy in dark mode — the flipped token would invert them. */
  [data-theme="dark"] .logo .mk         {{ background: #1b2540; color: var(--chartreuse); }}
  [data-theme="dark"] .sec-eyebrow .n   {{ background: #1b2540; color: var(--on-dark); }}
  [data-theme="dark"] .consensus-tag    {{ background: #1b2540; color: var(--on-dark); }}
  [data-theme="dark"] .tenth-tag        {{ background: #1b2540; color: var(--chartreuse); }}
  [data-theme="dark"] .guide-toggle     {{ background: #1b2540; color: var(--chartreuse);
                                           box-shadow: rgba(0,0,0,0.55) 0 1px 3px 0,
                                                       rgba(0,0,0,0.65) 0 12px 24px -12px,
                                                       rgba(255,255,255,0.06) 0 0.5px 0.5px 0 inset; }}
  [data-theme="dark"] .guide-toggle:hover{{ background: #243054;
                                            box-shadow: rgba(0,0,0,0.65) 0 1px 3px 0,
                                                        rgba(0,0,0,0.75) 0 16px 28px -12px,
                                                        rgba(255,255,255,0.08) 0 0.5px 0.5px 0 inset; }}
  /* Closest chip — chartreuse bg, force dark text */
  [data-theme="dark"] .f-name .f-mark           {{ color: #1b2540; background: var(--chartreuse); }}
  [data-theme="dark"] .f-name .f-mark.farthest  {{ background: transparent; color: var(--storm);
                                                   box-shadow: 0 0 0 1px rgba(255,255,255,0.20); }}

  /* Past-reports button: solid white in light → translucent in dark */
  [data-theme="dark"] .mast-index-btn       {{ background: rgba(255,255,255,0.04);
                                               border: 1px solid rgba(255,255,255,0.14);
                                               color: var(--midnight-navy); }}
  [data-theme="dark"] .mast-index-btn:hover {{ background: var(--chartreuse);
                                               border-color: var(--chartreuse);
                                               color: #1b2540; }}

  /* Hard-coded borders that aren't on tokens yet */
  [data-theme="dark"] .masthead         {{ border-bottom-color: rgba(255,255,255,0.10); }}
  [data-theme="dark"] footer.colophon   {{ border-top-color: rgba(255,255,255,0.10); }}

  /* Map card surfaces */
  [data-theme="dark"] .map-svg          {{ background: var(--pure); }}
  [data-theme="dark"] .map-help summary {{ background: var(--pure); color: var(--storm); }}
  [data-theme="dark"] .map-help summary:hover,
  [data-theme="dark"] .map-help[open] summary {{ background: #1b2540; color: var(--chartreuse); }}
  [data-theme="dark"] .map-help-popover           {{ background: var(--pure); }}
  [data-theme="dark"] .map-help-popover p         {{ color: var(--midnight-navy); }}
  [data-theme="dark"] .map-help-popover .map-help-title {{ color: var(--storm); }}

  /* SVG ink overrides — split by element + attribute so fill="none"
     circles don't accidentally fill in. Order: more specific first. */
  [data-theme="dark"] .map-svg svg circle[fill="#1b2540"]            {{ fill: var(--midnight-navy); }}
  [data-theme="dark"] .map-svg svg circle[fill="none"][stroke="#1b2540"] {{ stroke: rgba(255,255,255,0.28); }}
  [data-theme="dark"] .map-svg svg rect[fill="#1b2540"]              {{ fill: #1b2540; }}
  [data-theme="dark"] .map-svg svg rect[fill="none"][stroke="#1b2540"]   {{ stroke: rgba(255,255,255,0.36); }}
  [data-theme="dark"] .map-svg svg g[fill="#1b2540"]                 {{ fill: var(--midnight-navy); }}
  [data-theme="dark"] .map-svg svg g[fill="#7c8293"]                 {{ fill: var(--storm); }}
  [data-theme="dark"] .map-svg svg text[fill="#1b2540"]              {{ fill: var(--midnight-navy); }}
  [data-theme="dark"] .map-svg svg text[fill="#d0f100"]              {{ fill: var(--chartreuse); }}
  [data-theme="dark"] .map-svg svg text[fill="#6b7184"]              {{ fill: var(--storm); }}
  [data-theme="dark"] .map-svg svg text[fill="#7c8293"]              {{ fill: var(--storm); }}
  [data-theme="dark"] .map-svg svg g[stroke="rgba(0,39,80,0.06)"]    {{ stroke: rgba(255,255,255,0.06); }}
  [data-theme="dark"] .map-svg svg g[stroke="rgba(0,39,80,0.10)"]    {{ stroke: rgba(255,255,255,0.10); }}
  [data-theme="dark"] .map-svg svg g[stroke="rgba(27,37,64,0.20)"]   {{ stroke: rgba(255,255,255,0.16); }}

  /* Subtle pill/header backgrounds layered on dark surface */
  [data-theme="dark"] .map-top .pill                    {{ background: rgba(255,255,255,0.04); }}
  [data-theme="dark"] .frames-head                      {{ background: rgba(255,255,255,0.02); }}
  [data-theme="dark"] .tenth-foot                       {{ background: rgba(255,255,255,0.02); }}
  [data-theme="dark"] .mode                             {{ background: rgba(255,255,255,0.03); }}
  [data-theme="dark"] .tenth-body .pull,
  [data-theme="dark"] .consensus-body .pull,
  [data-theme="dark"] .f-body .pull                     {{ background: rgba(255,255,255,0.04); }}
  [data-theme="dark"] .frame:hover                      {{ background: rgba(255,255,255,0.02); }}
  [data-theme="dark"] .frame.open                       {{ background: rgba(255,255,255,0.03); }}
  [data-theme="dark"] .guide-panel-foot                 {{ background: rgba(255,255,255,0.03); }}

  /* Painting on hero — slight dim for visual continuity */
  [data-theme="dark"] .hero-bg                          {{ filter: brightness(0.75); }}

  /* === Theme toggle button === */
  .theme-toggle{{
    appearance: none; -webkit-appearance: none;
    background: transparent;
    border: 1px solid var(--border-strong);
    color: var(--midnight-navy);
    width: 32px; height: 32px;
    border-radius: 999px;
    cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    flex-shrink: 0;
  }}
  .theme-toggle:hover{{ background: var(--border-subtle); }}
  .theme-toggle:focus-visible{{ outline: 2px solid var(--chartreuse); outline-offset: 2px; }}
  .theme-toggle svg{{ width: 16px; height: 16px; display: block; }}
  .theme-toggle .icon-moon{{ display: none; }}
  .theme-toggle .icon-sun{{ display: block; }}
  [data-theme="dark"] .theme-toggle{{
    border-color: rgba(255,255,255,0.18);
    color: var(--chartreuse);
  }}
  [data-theme="dark"] .theme-toggle:hover{{ background: rgba(255,255,255,0.06); }}
  [data-theme="dark"] .theme-toggle .icon-sun{{ display: none; }}
  [data-theme="dark"] .theme-toggle .icon-moon{{ display: block; }}

  @media (max-width: 920px){{
    .hero-inner{{ grid-template-columns: 1fr; padding: 40px 24px 24px; }}
    .hero-quote.consensus{{ display: none; }}
    .hero-q{{ font-size: 15px; }}
    .map-foot{{ grid-template-columns: 1fr 1fr; }}
    .map-foot .cell{{ border-bottom: 1px solid var(--border-subtle); }}
    .modes{{ grid-template-columns: 1fr; }}
    .frames-head{{ display: none; }}
    .frame-row{{ grid-template-columns: 40px 1fr 80px 16px; row-gap: 4px; }}
    .frame-row .f-bar{{ display: none; }}
    .frame-row .f-lean{{ grid-column: 2; grid-row: 2; font-size: 13px; max-width: none; }}
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
<main data-screen-label="{t(locale, "screen_label")}">

  <header class="masthead">
    <div class="logo"><span class="mk">10</span> Henge</div>
    <div class="mast-meta">
      <b>{t(locale, "masthead_report")} #{report_id}</b><span class="sep">·</span>{timestamp}<span class="sep">·</span>v0.4
    </div>
    <div class="mast-actions">
      <a class="mast-index-btn" href="../index.html" aria-label="{t(locale, "masthead_index_aria")}">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M2 3.5h12M2 8h12M2 12.5h12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        <span>{t(locale, "masthead_index_btn")}</span>
      </a>
      <button type="button" class="theme-toggle" aria-label="{t(locale, "theme_toggle_aria")}" title="{t(locale, "theme_toggle_aria")}">
        <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
        <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      </button>
    </div>
  </header>

  {meta_html}

  <section class="hero">
    <div class="hero-bg" aria-hidden="true"></div>
    <div class="hero-inner">
      <div>
        <div class="hero-verdict">
          <span>{t(locale, "hero_verdict_consensus")}</span><span class="sep">&middot;</span><b>{html_mod.escape(hero_verdict_label)}</b><span class="sep">&middot;</span><span>&sigma; {spread_sigma:.3f}</span>
        </div>
        <h1 class="hero-h">{t(locale, "hero_h_a")}<br><em>{t(locale, "hero_h_b")}</em></h1>
        <p class="hero-dek">{t(locale, "hero_dek")}</p>
      </div>

      <div class="hero-quotes">
        <a class="hero-quote consensus" href="#consensus">
          <div class="mark"><span class="d"></span>{t(locale, "hero_quote_nine_mark")}</div>
          <blockquote class="hero-q">{html_mod.escape(nine_lean)}</blockquote>
          <div class="cite">{t(locale, "hero_quote_consensus_cite")} &middot; &sigma; {spread_sigma:.3f}</div>
        </a>
        <a class="hero-quote tenth" href="#tenth">
          <div class="mark"><span class="ord">{t(locale, "hero_quote_tenth_mark_ord")}</span> &middot; {t(locale, "hero_quote_tenth_mark_label")}</div>
          <blockquote class="hero-q">{tenth_lean_html}</blockquote>
          <div class="cite">{t(locale, "hero_quote_tenth_cite")} &middot; d {tenth_distance:.3f}</div>
        </a>
      </div>
    </div>
  </section>

  <div class="page">

  <section class="section">
    <div class="sec-head">
      <div class="l">
        <div class="sec-eyebrow"><span class="n">01</span>{t(locale, "section01_eyebrow_prefix")}{report_id}</div>
        <h2>{t(locale, "section01_h2_a")}<em>{t(locale, "section01_h2_em")}</em></h2>
        <blockquote class="question-pull">{question_safe}</blockquote>
      </div>
    </div>

    <section class="map-card">
      <div class="map-top">
        <div class="l">
          <span class="pill"><span class="d"></span>{t(locale, "fig_label")}</span>
          <span class="meta">{t(locale, "fig_meta")}</span>
        </div>
        <div class="legend-row">
          <span><i class="ld t"></i>{t(locale, "legend_tenth")}</span>
          <span><i class="ld n"></i>{t(locale, "legend_consensus")}</span>
        </div>
      </div>

      <div class="map-svg">
        <details class="map-help">
          <summary aria-label="{t(locale, "map_help_aria")}">?</summary>
          <div class="map-help-popover">
            <p class="map-help-title">{t(locale, "map_help_title")}</p>
            <p>{t(locale, "map_help_p1")}</p>
            <p>{t(locale, "map_help_p2")}</p>
            <p>{t(locale, "map_help_p3")}</p>
            <p>{t(locale, "map_help_p4")}</p>
            <p>{t(locale, "map_help_p5")}</p>
          </div>
        </details>
        {map_svg}
        {map_legend_html}
      </div>

      <div class="map-foot">
        <div class="cell">
          <div class="lbl">{t(locale, "footcell_tenth_d")}</div>
          <div class="val">{tenth_distance:.3f}<span class="badge">tenth</span></div>
          <div class="sub">{t(locale, "footcell_tenth_sub")}</div>
        </div>
        <div class="cell">
          <div class="lbl">{t(locale, "footcell_closest")}</div>
          <div class="val">{min_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(closest_name)}</div>
        </div>
        <div class="cell">
          <div class="lbl">{t(locale, "footcell_divergent")}</div>
          <div class="val">{max_frame_distance:.3f}</div>
          <div class="sub">{html_mod.escape(most_divergent_name)}</div>
        </div>
        <div class="cell">
          <div class="lbl">{t(locale, "footcell_spread")}</div>
          <div class="val">{spread_sigma:.3f}</div>
          <div class="sub">{t(locale, "footcell_spread_sub")}</div>
        </div>
      </div>
    </section>

    {consensus_block_html}

    {claims_panel_html}

    <article class="tenth-card" id="tenth">
      <header class="tenth-top">
        <div>
          <div class="tenth-tag"><span class="d"></span>{t(locale, "tenth_tag_label")}</div>
          <p class="tenth-lead">{t(locale, "tenth_lead")}</p>
          <h3>{t(locale, "tenth_h3_a")}<em>{t(locale, "tenth_h3_em")}</em></h3>
        </div>
        <div class="tenth-d-stat">
          <b>{tenth_distance:.3f}</b>
          {t(locale, "tenth_d_label")}
        </div>
      </header>

      <div class="tenth-body">
        {tenth_response_html}
        {tenth_modes_html}
      </div>

      <footer class="tenth-foot">
        <span>{t(locale, "tenth_foot_left")}</span>
        <span>embed <b>{html_mod.escape(provider)}/{html_mod.escape(model)}</b> · ~USD {cost_estimate_usd:.2f}</span>
      </footer>
    </article>

    {informed_card_html}
  </section>

  <section class="section">
    <div class="sec-head">
      <div class="l">
        <div class="sec-eyebrow"><span class="n">02</span>{t(locale, "section02_eyebrow")}</div>
        <h2>{t(locale, "section02_h2_a")}<em>{t(locale, "section02_h2_em")}</em></h2>
        <p class="sub">{t(locale, "section02_sub")}</p>
      </div>
    </div>

    <section class="frames">
      <div class="frames-head">
        <span>{t(locale, "frames_head_idx")}</span>
        <span>{t(locale, "frames_head_frame")}</span>
        <span>{t(locale, "frames_head_lean")}</span>
        <span>{t(locale, "frames_head_distance")}</span>
        <span style="text-align:right;">{t(locale, "frames_head_d")}</span>
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
      {t(locale, "masthead_report").lower()}&nbsp;<b>#{report_id}</b>
    </div>
  </footer>

  </div>
</main>

<aside class="guide" id="guide" aria-label="{t(locale, "guide_aria")}">
  <div class="guide-panel" role="dialog" aria-labelledby="guide-title">
    <div class="guide-panel-body">
      <p class="kicker">{t(locale, "guide_kicker")}</p>
      <h3 id="guide-title">{t(locale, "guide_title_a")}<em>{t(locale, "guide_title_em")}</em></h3>
      <ol>
        <li>{t(locale, "guide_rule_1")}</li>
        <li>{t(locale, "guide_rule_2")}</li>
        <li>{t(locale, "guide_rule_3")}</li>
        <li>{t(locale, "guide_rule_4")}</li>
        <li>{t(locale, "guide_rule_5")}</li>
        <li>{t(locale, "guide_rule_6")}</li>
        <li>{t(locale, "guide_rule_7")}</li>
        <li>{t(locale, "guide_rule_8")}</li>
        <li>{t(locale, "guide_rule_9")}</li>
      </ol>
    </div>
    <div class="guide-panel-foot">
      <p class="foot">{t(locale, "guide_foot")}</p>
      <button type="button" class="guide-close" onclick="document.getElementById('guide').classList.remove('open')">{t(locale, "guide_close")}</button>
    </div>
  </div>
  <button type="button" class="guide-toggle" onclick="document.getElementById('guide').classList.toggle('open')" aria-expanded="false">
    <span class="marker"></span>
    {t(locale, "guide_btn")}
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

  /* Theme toggle — cycles light <-> dark, persists to localStorage. */
  (function(){{
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', function(){{
      var current = document.documentElement.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try {{ localStorage.setItem('henge-theme', next); }} catch (e) {{}}
    }});
    /* Live-react to system changes only when user hasn't picked. */
    if (window.matchMedia) {{
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var handler = function(e){{
        var stored = null;
        try {{ stored = localStorage.getItem('henge-theme'); }} catch(_){{}}
        if (stored === 'light' || stored === 'dark') return;
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }};
      if (mq.addEventListener) mq.addEventListener('change', handler);
      else if (mq.addListener) mq.addListener(handler);
    }}
  }})();
</script>
</body>
</html>"""

    return page
