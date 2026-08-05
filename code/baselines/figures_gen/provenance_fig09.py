#!/usr/bin/env python3
"""Record the Fig. 9 provenance, including the encounter-93 sensitivity."""
import json

P = ("/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data/"
     "PROVENANCE_v2.json")

d = json.load(open(P))

d["fig09_fingerprint"] = {
    "purpose": ("Fig. 9, the task-aligned error fingerprint. Gardner-Altman "
                "estimation plot of the paired Stage-2 minus Stage-1b contrast "
                "in episode-mean e_parallel."),
    "data": ("code/baselines/figures_gen/fig_data/errdir_v2.npz, keys "
             "epi_par__Stage-1b / epi_par__Stage2 / epi_fneg__*."),
    "plotted_by": "figure_plotting_v1/fig09_fingerprint.py",
    "why_gardner_altman": (
        "The quantity of record is a paired contrast on the same 200 "
        "encounters, so the difference and its interval belong on their own "
        "axis aligned with the arm-level data. A histogram would discard the "
        "pairing and a raincloud would show two marginal distributions whose "
        "overlap says nothing about the within-encounter difference."),
    "interval": (
        "Panel (b) draws a percentile bootstrap interval, 20000 resamples, "
        "numpy default_rng(12345): [-0.2799,-0.0464] m. The Student-t interval "
        "[-0.2833,-0.0445] is reproduced as a self-check against the "
        "manuscript's [-0.283,-0.045] and the two agree to about 3 mm, so the "
        "choice of interval does not carry the conclusion."),
    "encounter_93": (
        "A lone geometric extreme: Stage-1 -54.086, Stage-1b -47.520, Stage-2 "
        "-46.220 m, against a second-largest Stage-1b magnitude of -1.079 m. "
        "All three arms are extreme together, so the encounter geometry is "
        "anomalous rather than any predictor. It is the only encounter with "
        "|e_par| > 8 m in either arm, and its paired difference (+1.299 m) is "
        "unremarkable, which is the paired design doing its job."),
    "outlier_rendering": (
        "Panel (a) focuses on [-2.2,+2.2] m and marks encounter 93 at the axis "
        "edge with arrows and its numeric values, rather than clipping it "
        "silently or letting it compress the other 199 encounters into a line. "
        "This is annotation off-scale, not a broken axis: nothing is hidden and "
        "the encounter's paired difference is drawn normally in panel (b), "
        "circled and labelled. Panel (b) spans [-2.6,+2.6] because its data "
        "reach -2.289 and +2.399 m and no outlier justifies clipping there; the "
        "two panels therefore carry different ordinate ranges, which the "
        "caption states. A self-check refuses to plot if either panel would "
        "clip a point."),
    "leave_one_out": (
        "Dropping encounter 93 moves the paired mean from -0.1639 to -0.1712 m "
        "and the paired-t p from 0.00740 to 0.00507, so the contrast is robust "
        "to it. The ARM-level dispersion is not: Stage-1b goes from "
        "-0.176 +/- 0.239 to +0.062 +/- 0.025 (the sign of the mean flips and "
        "the SEM falls by a factor of 9.5) and Stage-2 from -0.340 +/- 0.235 to "
        "-0.109 +/- 0.047. tab:errdir's SEMs and its '|mean| > 2 SEM ? NO' "
        "verdict are therefore driven by this single encounter. The n=200 "
        "figures remain the reported ones, but a leave-one-out footnote states "
        "this so a referee's sensitivity analysis finds nothing unannounced."),
    "geometry_inset": (
        "A small inset in panel (a) fixes the sign convention with five "
        "elements only: ego, true neighbour, predicted neighbour, the "
        "line-of-sight axis, and the projection labelled e_parallel < 0. It is "
        "an extension of the axis label rather than a concept schematic, which "
        "is why it is an inset and not a panel. Placed in the empty upper band: "
        "measured occupancy shows no point and no connecting line reaches "
        "[+1.8,+2.2] m, and the Stage-1b column is empty above +0.959 m."),
}

d["lessons"]["plausible_but_falsified_diagnoses"].append({
    "claim": ("A single focused ordinate of [-2.2,+2.2] m would serve both "
              "panels of Fig. 9, since the only extreme value is encounter "
              "93's arm-level e_parallel."),
    "why_it_appealed": ("Encounter 93 is the sole |e_par| > 8 m case and its "
                        "paired difference is a mild +1.299 m, so the "
                        "difference axis looked comfortably inside the same "
                        "range as the arm axis."),
    "how_it_died": ("A clipping assertion on the difference panel. The paired "
                    "differences run from -2.289 to +2.399 m, so six points sat "
                    "outside the shared range and matplotlib was dropping them "
                    "without complaint. Panel (b) was widened to [-2.6,+2.6] "
                    "and the assertion now fails on any clipped point. The "
                    "lesson is that 'the outlier is handled' does not imply "
                    "'the derived quantity fits the same axis'."),
})

d["lessons"]["rule_7"] = (
    "Assert that no axis clips its data. Two separate defects in this figure "
    "set were invisible without it: Fig. 9's difference panel silently dropped "
    "six of 200 points that lay outside a range chosen for the arm panel, and "
    "the left margin clipped nothing but antialiasing. Range choices made for "
    "one panel do not transfer to a panel showing a derived quantity, so check "
    "each axis against min and max of what it actually draws, and check the "
    "canvas edges for ink after rendering.")

json.dump(d, open(P, "w"), indent=2, ensure_ascii=False)
print("wrote", P)
print("fig09 keys:", len(d["fig09_fingerprint"]))
print("falsified diagnoses:",
      len(d["lessons"]["plausible_but_falsified_diagnoses"]))
print("rules:", [k for k in d["lessons"] if k.startswith("rule")])
