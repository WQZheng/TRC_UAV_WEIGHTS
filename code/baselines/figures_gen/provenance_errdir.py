#!/usr/bin/env python3
"""Record the errdir_v2 provenance and deprecate errdir_profile.npz.

No try/except around the data path (rule_2): if a key or file is missing this
must crash rather than silently degrade.
"""
import json
import os

P = ("/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data/"
     "PROVENANCE_v2.json")

d = json.load(open(P))

d["errdir_v2"] = {
    "purpose": ("Fig. 8 (CPA error profile) and Fig. 9 (task-aligned "
                "fingerprint). Replaces errdir_profile.npz, which is "
                "deprecated; see deprecates_errdir_profile below."),
    "produced_by": "code/baselines/figures_gen/export_errdir_v2.py",
    "contents": {
        "profile__<arm>": ("mean |prediction error| per |k - k_CPA| bucket, "
                           "k = 0..7, metres"),
        "counts__<arm>": ("contributing steps per bucket: 200 at k=0 (one CPA "
                          "step per episode), 400 for the symmetric pairs, "
                          "396/389/380 at k=5/6/7 from horizon truncation"),
        "epi_par__<arm>": ("per-episode critical-window mean e_parallel, "
                           "n=200, metres, negative = neighbour predicted "
                           "closer to the ego than it truly is"),
        "epi_fneg__<arm>": "per-episode toward-ego fraction, n=200",
        "epi_ade__<arm>": ("per-episode critical-window mean |error|, n=200, "
                           "metres"),
    },
    "geometry_provenance": (
        "Recomputed with the geometry of diag_error_direction.measure() and "
        "p0_errdir_episode.per_episode(), the functions that wrote "
        "RQ5_PROFILE.txt and P0_ERRDIR_EPISODE.txt. The prediction is carried "
        "to the absolute frame (pred_abs = mean_pred*SCALE + nei_origin) and "
        "compared against nf[:,0,1:h+1,:], the CPA step is taken from the ego "
        "REFERENCE trajectory ref[:,1:h+1,:], and episodes are sampled at "
        "T = T_EPISODE = 20."),
    "binding_self_check": (
        "The profile is bound to tab:rq5-profile by construction rather than "
        "by inspection: the step-count-weighted mean of buckets 0-3 must equal "
        "each arm's tabulated critical error to 5e-3 (achieved 13.5350 / "
        "1.0390 / 4.0678 against 13.53 / 1.04 / 4.07) and the complement must "
        "equal the inert value (23.1369 / 2.0822 / 4.3983 against 23.14 / "
        "2.08 / 4.40). Episode-level means, SEMs and toward-ego fractions "
        "reproduce P0_ERRDIR_EPISODE.txt to 5e-4, and the paired S2-S1b "
        "contrast reproduces -0.1639 m with paired-t p=0.007399 and Wilcoxon "
        "p=0.001137; its t-interval is [-0.2833,-0.0445] against the "
        "manuscript's [-0.283,-0.045]. The script refuses to write the npz if "
        "any of these fails."),
    "deprecates_errdir_profile": (
        "errdir_profile.npz (from collect_fig_data.py::errdir_profile) is "
        "DEPRECATED and must not be plotted. It could not reproduce "
        "tab:rq5-profile by any subsetting: its Stage-1b buckets spanned "
        "1.82-2.14 m while the table's critical value is 1.04 m, so no subset "
        "of larger numbers can average to a smaller one. Three substantive "
        "deviations from the authoritative geometry were found. (1) It anchored "
        "the CPA on the ego's INITIAL position x0[:,0:3], a fixed point, "
        "instead of the moving ego reference trajectory, so the same physical "
        "step landed in a different bucket. (2) It differenced in the "
        "recentred/scaled space and multiplied by SCALE, indexing "
        "nfut[:,0,:h], instead of comparing absolute-frame positions against "
        "nf[:,0,1:h+1,:] - a one-step time offset against a different array. "
        "(3) It sampled T=30 rather than T_EPISODE=20, which changes the "
        "reference trajectory and hence the CPA step. The file also carried no "
        "definition of its own abscissa."),
    "stage2_spike_resolved": (
        "The deprecated file showed an isolated Stage-2 value of 12.638 at "
        "bucket 1 between neighbours of 2.424 and 3.294. Under the "
        "authoritative geometry the spike does not exist: the Stage-2 profile "
        "is 3.733 3.931 4.127 4.313 4.713 4.600 5.327 4.649, with no bucket "
        "above three times its median. The spike was an artefact of the "
        "deprecated CPA anchoring, not a physical feature, so it is not "
        "mentioned in the caption."),
    "profile_shapes": (
        "Outer-to-inner ratio (bucket 7 / bucket 0) quantifies the three "
        "shapes: Stage 1 flat at 1.056 (a predictor that never saw the "
        "encounter distribution has no CPA structure), Stage-1b mildly rising "
        "at 1.163 while lowest everywhere, Stage 2 rising at 1.245. This is "
        "the graphical form of the critical/inert ratios 0.585 / 0.499 / "
        "0.925. Note the two quantities use different denominators: the "
        "profile stops at bucket 7, whereas the inert value aggregates every "
        "step beyond the critical window, which is why Stage 1's inert 23.14 m "
        "exceeds the profile's right-hand end of 14.32 m."),
}

d["lessons"]["plausible_but_falsified_diagnoses"].append({
    "claim": ("The 8-point errdir_profile.npz was the plotting data for the "
              "RQ5 error profile and merely needed a neutral title plus the "
              "Stage-1b curve."),
    "why_it_appealed": ("The file sat in fig_data next to the audited v2 "
                        "products, was loaded by a figure script that named it "
                        "the preferred source, and had one array per arm with "
                        "exactly the three arm names used by the table."),
    "how_it_died": ("Arithmetic, before any plotting. Stage-1b's buckets span "
                    "1.82-2.14 m but the table reports a critical error of "
                    "1.04 m, and no subset of numbers all above 1.8 can "
                    "average to 1.04. Reading the producer then exposed three "
                    "geometric deviations, and the file turned out to define "
                    "no abscissa at all. Recomputation from the authoritative "
                    "geometry hit the table to 5e-3 on all six values."),
})

d["lessons"]["rule_4"] = (
    "A data file is not provenance. Before plotting an archived array, verify "
    "it reduces to an already-published number by an explicit weighted "
    "identity, and confirm its abscissa is defined somewhere. errdir_profile."
    "npz passed every superficial test - right directory, right arm names, "
    "right shape, a script that called it authoritative - and was still "
    "computed with a different CPA anchor, a different frame, a one-step time "
    "offset and a different episode horizon.")

d["lessons"]["rule_5"] = (
    "The log-ordinate veto is conditioned on zero-reference semantics, not on "
    "dynamic range. It was issued for Fig. 5, whose entire claim rests on a "
    "zero line (the rescue threshold) that a log axis cannot represent. An "
    "error profile carries no such reference, so the veto does not reach it. "
    "Fig. 8 still avoids a log ordinate for an independent reason: its "
    "comparison of record is multiplicative (1.04 against 4.07 m critical "
    "error), and a log axis renders a factor of four as a constant visual "
    "offset, hiding the very gap the figure exists to show. Two stacked linear "
    "panels at their own full ranges were used instead. Cite this rather than "
    "re-litigating when cross-range data recurs.")

d["lessons"]["rule_6"] = (
    "Verify overlay placement geometrically, not by pixel colour. A "
    "colour-counting check on Fig. 8 reported all three arms as colliding with "
    "the legends when none did: the curves span the full panel width, so "
    "colour cannot distinguish 'curve beneath the legend' from 'curve "
    "elsewhere in the same pixel rows'. The reliable test compares each "
    "overlay's window extent against the curves' display-coordinate vertices "
    "AND the densely sampled segments between them, which names the offending "
    "segment instead of returning a pixel count. It caught two real "
    "collisions the pixel method had buried in false positives. Note also that "
    "an inset's tight bounding box extends about 0.056 of the host panel's "
    "height below its axes frame once a y-label and tick text are present, so "
    "a gap sized against the axes frame alone is not enough.")

json.dump(d, open(P, "w"), indent=2, ensure_ascii=False)
print("wrote", P)
print("errdir_v2 keys:", len(d["errdir_v2"]))
print("falsified diagnoses:",
      len(d["lessons"]["plausible_but_falsified_diagnoses"]))
print("rules:", [k for k in d["lessons"] if k.startswith("rule")])
