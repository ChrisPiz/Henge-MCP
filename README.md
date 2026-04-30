# TenthAI

> Mapea el desacuerdo entre 9 agentes IA + 1 disidente obligatorio.

Multi-agent MCP server. Tu pregunta → 9 marcos cognitivos + 1 disidente steel-man → mapa 2D del espacio de respuestas.

Inspirado en la **regla del décimo hombre** (inteligencia israelí post-Yom Kippur, popularizada en *World War Z*): si 9 advisors están de acuerdo, el #10 está OBLIGADO a disentir y construir el caso contrario coherente.

El output no es una respuesta. Es un mapa visual: ves dónde el consenso es robusto y dónde es frágil.

## Quick install (5 min)

```bash
git clone https://github.com/ChrisPiz/tenthai.git
cd tenthai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con ANTHROPIC_API_KEY (requerido) + OPENAI_API_KEY (embed por defecto)
python -m tenthai.server  # debe imprimir "✓ keys validated"
```

Voyage opcional para mejor calidad en español: descomenta en `.env` las líneas `EMBED_PROVIDER=voyage` y `VOYAGE_API_KEY=...`.

## Configurar Claude Code

Edita `~/.config/claude-code/mcp.json`:

```json
{
  "mcpServers": {
    "tenthai": {
      "command": "python",
      "args": ["-m", "tenthai.server"],
      "cwd": "/path/to/TenthAI"
    }
  }
}
```

Reinicia Claude Code.

## Cómo invocar

**Opción A — slash command `/decidir` (recomendado):**

Crea el archivo `~/.claude/commands/decidir.md` con este contenido:

```markdown
---
description: Invoca TenthAI — mapa de desacuerdo de 9 marcos + 1 disidente.
---
Usa la MCP tool `decide` del servidor `tenthai` para analizar:

$ARGUMENTS

Cuando recibas el JSON: cita `viz_path`, resume consenso de los 9, cita literal al #10, reporta `tenth_man_distance` y `max_frame_distance`. No interpretes la decisión por el usuario.
```

Luego desde cualquier proyecto en Claude Code:

```
/decidir si cobro CLP 4M o 6M por el contrato Acme
```

**Opción B — frase libre:**

Como es una MCP tool, también puedes pedirle a Claude que la use sin slash:

- "**Usa TenthAI para ayudarme a decidir** si acepto el contrato de Acme."
- "**Con TenthAI**, evalúa si esta arquitectura del PR es la correcta."
- "**Corre TenthAI sobre**: ¿Postgres o DynamoDB para este workload?"

A los ~60-150s, el navegador abre con el mapa 2D mostrando 9 marcos + el disidente.

## Qué retorna

La tool `decide()` retorna JSON con:

- `viz_path` — ruta absoluta al HTML (auto-abre en navegador).
- `responses` — 10 entradas: role, frame, status, distance_to_centroid_of_9, embedding_2d, response.
- `summary` — tenth_man_distance, max_frame_distance, n_frames_succeeded, embed_provider.
- `cost_clp` — costo aproximado en CLP.

## Los 9 marcos + 1

Cada marco produce un ángulo distinto sobre tu pregunta:

1. **Empírico** — datos, base rates, evidencia.
2. **Histórico** — precedente, casos análogos.
3. **Primer principios** — átomos físicos/económicos básicos.
4. **Analógico** — cross-domain (biología, militar, finanzas).
5. **Sistémico** — efectos segundo orden, feedback loops.
6. **Ético** — deontológico vs consecuencialista.
7. **Soft-contrarian** — replantea un supuesto sin invertir todo.
8. **Optimista-radical** — caso 10× mejor.
9. **Pre-mortem** — asume que ya falló, describe por qué.
10. **Décimo hombre** — steel-man del disenso vs el consenso de los 9.

## Costo

~CLP 270-530 (~USD 0.30-0.60) por invocación. Logueado en cada output.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

5 tests críticos sobre invariantes de diseño + 2 smoke tests + manejo de error de provider. Suite ejecuta en <5s con SDK calls mockeados.

## Limitaciones

- **MDS no PCA:** El mapa usa classical MDS sobre cosine distance. Esto preserva distancias entre pares fielmente (mejor que PCA con N=10 en alta dim, que es estadísticamente trivial). Aún así, valida ranking de distancias contra tu juicio humano en las primeras 3 invocaciones.
- **MVP scope:** sin persistencia, sin streaming, sin auto-tool-use de los 9 agentes. Cada invocación es independiente.
- **Solo Claude Code testeado.** Otros MCP clients deberían funcionar (transport stdio estándar) pero no validados.

## License

MIT.
