"""Append fig10 (detection lead-time boundary) and the style registry to
PROVENANCE_v2.json.

Run after fig10_leadtime.py has rendered successfully.
"""
import json
import os
import sys

import numpy as np

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
P = f"{DATA}/PROVENANCE_v2.json"


def main():
    if not os.path.exists(P):
        sys.exit(f"missing {P}")
    with open(P) as fh:
        prov = json.load(fh)

    D = np.load(f"{DATA}/leadtime_v2.npz", allow_pickle=True)
    h = D["horizons_s"]
    i3 = int(np.where(h == 3.0)[0][0])

    prov["fig10_leadtime"] = {
        "figure": "figures_v1/fig10_leadtime.pdf",
        "script": "figure_plotting_v1/fig10_leadtime.py",
        "data": "fig_data/leadtime_v2.npz",
        "panels": {
            "a": "conflict rate versus available detection lead time, four "
                 "arms, full 0-90% range, log abscissa to seat the uneven "
                 "1/2/3/4/5/6/7/10/20 s grid. Three regime bands in neutral "
                 "grey. Hosts the mechanism inset.",
            "b": "same series magnified to 0-17%, resolving the Stage-1 hump "
                 "against the Stage-1b matched control and the exact zeros of "
                 "Stage 2 and the oracle. Hosts the shared legend.",
        },
        "stage1_hump": {
            "values_3_to_7s": [float(D["cr__Stage1"][int(np.where(h == x)[0][0])])
                               for x in (3.0, 4.0, 5.0, 6.0, 7.0)],
            "note": "1.5 -> 7.0 -> 10.0 -> 12.0 -> 8.5, non-monotone, matches "
                    "the manuscript verbatim. Hedged in the caption as "
                    "'consistent with predictor-induced spurious avoidance'. "
                    "The stronger reading, that the predictor induces "
                    "conflicts, is not claimed: the data show a rate, not a "
                    "cause.",
        },
        "matched_control": {
            "stage1b_max_from_3s": float(np.max(D["cr__Stage-1b"][i3:])),
            "note": "Stage-1b stays at or below 0.5% from 3 s onward, so the "
                    "hump is specific to Stage 1 and not a property of any "
                    "arm that shares its training regime. This control is "
                    "load-bearing and now rests on a column locked by weight "
                    "filename, not on column position.",
        },
        "zero_cells": {
            "count": 14,
            "note": "14 of the 36 cells are exactly 0.0. This is why the "
                    "ordinate is linear: rule_5 restricts a log ordinate to "
                    "the case where a zero reference carries the meaning "
                    "(fig05's rescue line). Here zero IS the result, and a log "
                    "axis cannot draw it.",
        },
        "attribution_inset": {
            "content": "mechanism split of the lead-time conflicts themselves, "
                       "actuation-limited versus prediction-limited. NOT the "
                       "weak-constraint result, which is a separate figure.",
            "1s": {"n": int(D["n_s2_conflicts"][0]),
                   "pct_actuation": round(100.0 * D["act_limited"][0]
                                          / D["n_s2_conflicts"][0], 1)},
            "2s": {"n": int(D["n_s2_conflicts"][1]),
                   "pct_actuation": round(100.0 * D["act_limited"][1]
                                          / D["n_s2_conflicts"][1], 1)},
            "note": "both rows carry their n because 98% of 165 and 95% of 22 "
                    "are not equally trustworthy. No bar is drawn from 3 s on: "
                    "with zero conflicts the split is undefined, and an empty "
                    "slot would read as a measured zero.",
        },
        "axis_ranges": {
            "panel_a": [-3.0, 92.0],
            "panel_b": [-0.6, 17.0],
            "note": "panel (b) is raised to 17 although its data top out at "
                    "13.0. The two notes needed room: the only curve-free block "
                    "at a ceiling of 14 measured 159 px against a 233 px note, "
                    "so no placement stayed on the canvas. Widening a limit "
                    "cannot hide data, which is the direction rule_7 permits; "
                    "the ranges of (a) and (b) differ and the caption says so.",
        },
        "verification": [
            "occlusion: four overlays (legend, inset, two notes) versus densely "
            "sampled curve vertices in display coordinates, 0 hits each",
            "overlay-versus-overlay: bbox intersection over all pairs, none. "
            "Added after hand placement put the legend under the inset twice; "
            "the data-only test could report this just as an inflated hit count",
            "clipping: every text artist inside the canvas, by extent, naming "
            "the artist. Found a clipped regime label and note that the "
            "edge-pixel probe could only quantify as 105 dark pixels",
            "edge ink: 0 on all four borders of the raster",
            "escapes: no LaTeX escape in any drawn label",
            "colour: every drawn colour registered in figstyle",
        ],
        "falsified_during_construction": [
            "the drafted colours were wrong twice over: Stage-1 was given "
            "#D55E00, which belongs to Vanilla-MPC, and the oracle was given "
            "#7A7A7A, which in this figure family is Stage-1's own grey. A "
            "proposal to hand the oracle #CC79A7 was also wrong, since that is "
            "Soft-IPP's. Resolved by reusing #8E44AD, the colour the oracle "
            "replay already carries in fig05 panel (c)",
            "the regime bands were tinted with #D55E00 and #0072B2, spending "
            "Vanilla-MPC's colour a second time and implying the last band "
            "belongs to Stage 2. A band marks an interval of the abscissa, a "
            "property of the experiment and of no arm, so it is now neutral grey",
            "the labels really did render as literal '\\%' and '\\,s': usetex "
            "is off and the escapes were only visible by extracting the PDF "
            "text layer, not by looking at the image",
        ],
    }

    prov["style_registry"] = {
        "file": "figure_plotting_v1/figstyle.py",
        "purpose": "single source for colour, typography and the geometric "
                   "checks. Created after style drift occurred four times: the "
                   "serif family diverged three times and fig10 was drafted "
                   "with another arm's colour. Root cause was identical each "
                   "time, every script hard-coding its own literals.",
        "authority": "RETROSPECTIVE. The nine locked figures are ground truth; "
                     "each registry value was read out of their sources. Locked "
                     "figures are NOT re-rendered to match it. fig10 onward "
                     "imports it; fig01-fig09 are left alone.",
        "not_to_be_confused_with": {
            "file": "figure_plot/figstyle.py",
            "note": "a different module for an older, differently-numbered "
                    "series (fig11_leadtime, fig13_corridor, ...) imported by "
                    "twelve scripts. It contradicts the locked series on five "
                    "entities: Soft-IPP yellow not #CC79A7, Conformal-MPC "
                    "#CC79A7 not #56B4E9, Stage-1 #56B4E9, Constant-Velocity "
                    "#E69F00, oracle black not #8E44AD. Left untouched because "
                    "editing it would break those twelve scripts.",
        },
        "entities": {
            "Stage-2": "#0072B2", "Stage-1b": "#009E73", "Oracle": "#8E44AD",
            "Conformal-MPC": "#56B4E9", "Vanilla-MPC": "#D55E00",
            "Soft-IPP": "#CC79A7", "Fixed-Predictor": "#E69F00",
            "Constant-Velocity": "#7A7A7A", "zero_reference": "#8A8A8A",
            "Stage-1": "family-dependent, see recorded_inconsistency",
        },
        "recorded_inconsistency": {
            "status": "RECORDED, DELIBERATELY NOT REOPENED",
            "grey": "#7A7A7A means Constant-Velocity in the arm-comparison "
                    "figures (fig01/02/03/06) and Stage 1 in the "
                    "prediction-space figures (fig08/09/10).",
            "orange": "#E69F00 means Fixed-Predictor in most figures and "
                      "Stage-1 in fig06.",
            "why_tolerated": "the two families never share a figure, so no "
                             "reader sees a contradiction. Both were settled "
                             "when those figures were locked; reopening either "
                             "means redrawing three approved figures.",
            "mitigation": "figstyle.color() takes a family argument for "
                          "Stage-1 and refuses to answer without one, so the "
                          "split cannot be resolved by accident.",
            "related": "#CC79A7 went to Soft-IPP and #8E44AD to the oracle for "
                       "the same reason: those two never share a figure either.",
        },
        "provides": ["COLORS", "color(entity, family)", "set_rc", "wilson_ci",
                     "curve_samples", "check_overlays", "check_clipping",
                     "check_escapes", "edge_ink", "assert_registered"],
    }

    # rule_8/rule_9 join the seven existing rules inside "lessons", which is
    # where rule..rule_7 already live. A parallel top-level "rules" key was
    # created first and would have split the rule namespace in two.
    rules = prov["lessons"]
    rules["rule_8"] = (
        "Style is registry-resolved, never literal. A figure asks for an entity "
        "by name and writes no hex. Same entity, same colour, across every "
        "figure: cross-figure identity outranks any within-figure aesthetic, "
        "because a reader tracking one arm between figures is misled by a "
        "colour change in a way no local improvement repays. A colour already "
        "spent on another entity is not free, and 'they never share a figure' "
        "is the ONLY admissible reason to reuse one -- it must be recorded when "
        "invoked. Verify against the rendered canvas, not the source: assert "
        "the retired colour's ink count is 0."
    )
    rules["rule_9"] = (
        "Place overlays by measurement in the coordinate system the API "
        "actually consumes. bbox_to_anchor takes axes fractions; reasoning in "
        "data units put a legend at 'fraction 0.335' meaning 29% conflict rate, "
        "straight through a descending curve, for 1904 collisions. Convert "
        "explicitly, and check overlays against each other as well as against "
        "the data, since two overlays can each clear the curves and still "
        "collide. Remember an annotation's extent includes its arrow and an "
        "inset's tightbbox extends about 0.056 panel heights below its frame."
    )

    with open(P, "w") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")

    print(f"PROVENANCE_v2.json now has {len(prov)} top-level keys")
    nr = len([k for k in rules if k == "rule" or k.startswith("rule_")])
    print(f"  rules in lessons: {nr}")
    nf = sum(len(v.get("falsified_during_construction", []))
             for v in prov.values() if isinstance(v, dict))
    print(f"  falsified diagnoses recorded in figure entries: {nf}")


if __name__ == "__main__":
    main()
