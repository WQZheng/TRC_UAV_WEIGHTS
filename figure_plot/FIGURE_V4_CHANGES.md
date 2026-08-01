# Figure revision v3 → v4 — short summary

**Scope:** visual/layout revision of the 11 result figures (Fig 3–13). **No data
changed** — every value still reproduces the v9 main table (η_w = 0.3, n = 200,
seed 12345). Our side edits data/scripts/figures only; no `.tex` was touched.

**Design principle applied throughout:** *form follows data structure, not visual
variety.* "Richer" comes from information layering (hero view + drill-down inset /
marginal) and layout discipline, **not** from novel chart types. Every visual
element encodes a real quantity; no fabricated continuity/density/trend.

---

## Four figures that were reworked

| Fig | v3 problem | v4 fix |
|-----|------------|--------|
| **3(c)** | 200×4 paired raster was ~87 % empty whitespace (conflicts are rare, ~11–12 %) | Replaced with a compact **UpSet-style conflict-pattern view**: only the patterns that actually occur, as episode-count bars + a method-membership dot-matrix. Zero wasted space; exposes which arms conflict together. Panels (a)/(b) unchanged. |
| **4** | An upper-left half-violin inset overlapped the main ECDF curves; a second inset also collided | Removed the half-violin; kept **one clean 30 m zoom inset** (lower-right, clear area); moved the legend to the empty upper-left; added a right-edge **Pr(min-sep < 30 m)** label column (= each arm's conflict rate). |
| **5** | The five certificate arms overplotted into one blob at (effort ≈ 52, CR ≈ 11–12) — indistinguishable | Added a **certificate-arm zoom inset** (true magnification via `indicate_inset_zoom`) that pulls the five arms apart; family convex hulls retained. |
| **13** | Four near-identical line+errorbar panels (monotonous); "conditional on completion" text overlapped the legend | **Panel differentiation** — (a) group lines + SD, (b) filled non-completion **ribbon band**, (c) delay lines, (d) throughput **point-range** with hollow markers; scope note moved into the (c) x-label. |

## Global layout pass (all 11 figures)

- **Unified font hierarchy** (centralised in `figstyle.set_rc`): axis title 10 /
  ticks 8.5 / legend 8 / annotations 7.5 pt. All 11 figures re-rendered so they
  inherit it consistently.
- **Fig 7(b):** moved the two-arm legend from lower-right (it collided with the
  paired-Δ annotation) to the clear upper-right.
- **Fig 6 / 11 / 12:** direction was already correct — only the global typography
  and spacing discipline applied.
- **Fig 8 / 9 / 10:** kept their approved structure (raster / heatmap+marginals /
  three-panel robustness); they only inherit the rc hierarchy.

## Preserved conventions

Okabe–Ito colour-blind-safe palette · certificate-equipped = solid,
certificate-free = dashed · Conformal = hollow · Oracle = black · fixed legend
order · 30 m = thin red dashed · vector PDF, fonts ≥ 8 pt · **no in-figure
title/conclusion text**, only (a)–(d) panel labels.

## One substantive finding surfaced by Fig 3(c) — for the text

Of the 25/200 episodes with ≥ 1 arm in conflict, **21 are all-four-arms
conflicts** (every planner conflicts on the *same* episode); only 4 episodes
distinguish the methods (per-arm counts 22/23/25/24 = main-table CR
11.0/11.5/12.5/12.0 %). Reading: residual conflicts are driven almost entirely by
**episode difficulty**, not planner-specific weakness — reinforcing the Fig 8 /
§5.5 attribution that residual conflicts are actuation-limited, not
prediction-limited. Worth one sentence in §5.2. (Detailed note: item 6 of
`COAUTHOR_TEXT_MEMO.md`.)
