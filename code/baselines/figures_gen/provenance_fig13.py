#!/usr/bin/env python3
"""Provenance record for Figure 13, corridor-scale market penetration.

Appends to PROVENANCE_v2.json. Decisions and their reasons are recorded because
the reasons are not recoverable from the figure or from the data files.
"""
import json
import os

DATA = os.environ.get(
    "FIG_DATA_DIR",
    "/data/lab/TRC_UAV_WEIGHTS/code/baselines/figures_gen/fig_data")
PATH = os.path.join(DATA, "PROVENANCE_v2.json")

REC = {
  "figure": "fig13_corridor",
  "script": "figure_plotting_v1/fig13_corridor.py",
  "exporter": "code/baselines/figures_gen/export_penetration_v2.py",
  "data": "fig_data/penetration_v2.npz",

  "why_the_sweep_was_re_run": (
    "The per-replication values never existed on disk. run_penetration.py "
    "collected each replication's metrics into a list (line 75-81), passed the "
    "list to agg(), which reduced it with np.nanmean/np.nanstd (line 31-37), "
    "and wrote only the aggregate to PENETRATION.txt. The six numbers behind "
    "every published mean were live in memory and then discarded. Because the "
    "review asked for replication points, and because six replications is few "
    "enough that a bare mean invites over-reading, the sweep was re-run with "
    "the individual replications retained."),

  "why_re_running_is_legitimate": (
    "The seed is a deterministic function of the replication index and the "
    "equipped fraction: seed = base + 1000*rep + int(p*97) "
    "(run_penetration.py:78). Every replication is independently reproducible, "
    "so the re-run is the same experiment rather than a new one. The exporter "
    "asserts that the re-run reproduces every published mean, SD and "
    "lateral-exit count, and refuses to write the npz otherwise. Verified: low "
    "demand p=0 reproduced mean 34.43 against 34.4 and SD 7.90 against 7.9, "
    "with the summed n exactly 113 against 113."),

  "aggregation_convention_trap": (
    "The published n is the SUM over replications (np.sum(ns), "
    "run_penetration.py:38), not a mean. A single replication yields about "
    "n/reps agents, so an initial single-replication check produced n=44 "
    "against a table value of 284 and looked like a factor-six discrepancy "
    "when it was only the convention. Recorded because the same confusion "
    "would recur on any future spot check."),

  "simulator_variant_matters": (
    "Panel (d) and (h) need the discard CAUSE, and only "
    "penetration_sim_disc2.py records it: a per-agent `reason` field, "
    "'pass'|'timeout'|'lateral' (line 69), surfaced as n_lateral / n_timeout in "
    "the metrics dict. The base penetration_sim.py returns `passed` but never "
    "why an aircraft failed to pass. Importing the base module and reading "
    "n_lateral would have yielded a silent NaN rather than an error, so the "
    "exporter asserts the four required keys are present. This is the same "
    "class of hazard as the bad PDF decoder recorded under fig12: a tool that "
    "returns a plausible wrong answer instead of failing."),

  "panel_d_departs_from_the_literal_review_spec": (
    "The review asked panel D for 'completed vs lateral-exit share'. Only the "
    "lateral-exit share is drawn. Reason: timeout is zero in every one of the "
    "ten configurations, so an aircraft either completes or exits laterally and "
    "the two shares sum to exactly 100 percent. Drawing both renders one degree "
    "of freedom as two curves that are arithmetic mirrors, which invites a "
    "reader to count them as two pieces of evidence; and it inverts the ink, "
    "since the completed share occupies 89-100 percent of the axis while the "
    "entire signal lives in the 0-11 percent remainder. The two-way split is "
    "itself an empirical finding rather than an assumption, so timeout==0 is "
    "asserted in load(). The caption states that the completed share is the "
    "complement. Approved by the author. Same family of reasoning as the fig12 "
    "ruling that an area mosaic is the wrong chart type for a matrix "
    "containing zeros: form follows data structure."),

  "why_y_axes_are_not_shared_across_rows": (
    "Throughput differs by a factor of 2.5 between demand levels (44.8 against "
    "18.7 passes/min) and conflict rate by roughly 20 pp. A shared y axis would "
    "flatten the low-demand row into a horizontal line and destroy the "
    "comparison the two rows exist to support. Only the x axis is shared, down "
    "a column. A self-check asserts that no column has identical limits in both "
    "rows, which is the executable form of rule_7's prohibition on carrying one "
    "panel's range to another."),

  "why_replication_points_are_ORCA_only": (
    "Three groups times five penetrations times six replications is ninety "
    "points per panel, which would bury the means. The ORCA group was chosen "
    "not merely to thin the ink but because it carries the largest information "
    "gain: the system group's dispersion is already tabulated as mean +/- SD in "
    "the main text (6.5+/-1.4 down to 2.6+/-0.6), whereas the ORCA group's "
    "group-level dispersion sits in an appendix. Plotting its points supplies "
    "uncertainty information the main text lacks. The caption says the points "
    "are for the ORCA-controlled group only, so that their absence elsewhere is "
    "not read as an absence of dispersion."),

  "externality_direction_is_asserted_not_described": (
    "ORCA delay FALLS as equipage rises, 6.5 -> 5.6 -> 5.5 -> 4.4 s at high "
    "demand: equipping part of the fleet is a positive spillover onto the "
    "aircraft that are not equipped. The review found the manuscript had this "
    "backwards, describing a degradation. The figure script asserts the sign in "
    "both demand rows, asserts monotonicity, and asserts that the high-demand "
    "sequence reproduces the four published values, so the figure cannot ship "
    "carrying the inverted claim."),

  "mid_penetration_conflict_bump_carries_no_significance_mark": (
    "High-demand system conflict rate runs 50.3 -> 57.2 -> 53.3 -> 52.1 -> "
    "47.7 percent, a 6.9 pp elevation at p=25 percent against replication SDs "
    "of 3.5 to 12.2 pp. Six replications give no statistical support, so no "
    "star, bracket or band is drawn. The thirty replication points make the "
    "case without an annotation having to concede it: their scatter is wider "
    "than the elevation. The caption names it a descriptive pattern, and the "
    "text uses the same words, so the two cannot drift apart. Same red line as "
    "the refusal to mark a non-significant difference in fig11."),

  "legend_is_retained_against_the_direct_label_default": (
    "fig11 and fig12 both deleted a legend on the ground that direct labels "
    "already carried the semantics and the legend was a redundant layer. Here "
    "the legend is kept. The premise of that earlier ruling was that direct "
    "labels were complete within a single panel; with eight panels sharing one "
    "encoding, direct labelling would repeat the same three labels eight times "
    "and add ink without adding information. The legend states the encoding "
    "once for the whole grid, which is a different function from labelling a "
    "cell. Author-approved as a boundary reading of the earlier ruling, not as "
    "an exception to it."),

  "row_labels_are_on_the_figure_not_only_in_the_caption": (
    "'High demand (arrival 0.16/step)' and 'Low demand (arrival 0.06/step)' are "
    "drawn at the head of each row so that a reader scanning the grid knows "
    "which demand level a row reports without leaving the image. Panels are "
    "lettered continuously (a)-(d) then (e)-(h) so the text and an appendix can "
    "cite the two rows separately."),

  "occlusion_check_scope_on_this_figure": (
    "This figure has no in-panel annotation: semantics are carried by the "
    "legend, the row labels and the panel letters, all of which sit outside "
    "every axes region. check_overlays therefore has no curve samples to test "
    "and does only its pairwise test. The real risk here is a row label or the "
    "legend covering a panel, so an explicit bbox-against-axes rectangle test "
    "was added. Recorded so that the near-empty check_overlays output on this "
    "figure is not later mistaken for a check that was skipped."),

  "data_sources": {
    "high_demand": "arrival=0.16/step/end; PENETRATION.txt, PENETRATION_SD.txt, PENETRATION_DISC2.txt",
    "low_demand": "arrival=0.06/step/end; PENETRATION_LOW.txt, PENETRATION_LOW_SD.txt, PENETRATION_LOW_DISC2.txt",
    "common": "horizon=400, warmup=100, reps=6, K=3, seed=12345, p in {0,25,50,75,100}%",
    "equipped": "PlanGrad-UAV, predictor stage2_final.pt plus CBF-MPC",
    "unequipped": "ORCA",
    "note": ("The DISC2 files are the most complete variant: they carry pass "
             "counts, completion and discard shares, the discard cause split, "
             "and a censored-delay upper bound.")
  },

  "endpoint_groups_are_NaN_by_construction": (
    "No aircraft is equipped at p=0 and none is unequipped at p=100, so those "
    "group metrics are NaN rather than zero. load() asserts this. Were they "
    "stored as zero, a mean line would dive to the axis floor at the edges and "
    "read as a real collapse in delay or conflict rate."),
}


# Recorded as the next free rule numbers. The existing "lessons" dict runs
# rule, rule_2 ... rule_9.
NEW_RULES = [
  ("rule_10",
   "A published table is not a data file. The penetration means were printed "
   "from a list that was reduced by np.nanmean/np.nanstd and then dropped, so "
   "the per-replication values never reached disk. Before promising to plot a "
   "quantity, establish that it was ever written down; if it was not, decide "
   "explicitly between regenerating it and changing the chart. Regeneration is "
   "only legitimate when the run is deterministically reproducible and the "
   "re-run is asserted to recover the published aggregate."),
  ("rule_11",
   "Two variants of the same simulator can differ in what they RECORD rather "
   "than in what they compute. penetration_sim.py and penetration_sim_disc2.py "
   "produce the same trajectories, but only the latter stores a per-agent "
   "discard reason. Reading a key the other variant does not produce yields "
   "NaN, not an exception. Assert that every required key is present at the "
   "point of import, because a NaN propagates into a plotted mean and looks "
   "like data."),
  ("rule_12",
   "When a measured quantity turns out to be degenerate, plot the free "
   "dimension and state the identity. Every discard in the corridor sweep is a "
   "lateral exit, so completed and discarded shares are exact complements; "
   "drawing both would render one degree of freedom as two mirrored curves and "
   "invite a reader to count them as two pieces of evidence. This overrides a "
   "reviewer request for two series, and the override belongs in the caption."),
]

FALSIFIED = [
  {
    "claim": ("The six per-replication values behind each published "
              "penetration mean can be recovered from the result files or the "
              "archived logs."),
    "why_it_appealed": ("Eight result-file variants exist, several carrying SDs, "
                        "completion shares and discard causes, so it looked "
                        "like some variant must have kept the raw values."),
    "how_it_died": ("run_penetration.py collects each replication into a list "
                    "(line 75-81), reduces it with np.nanmean/np.nanstd (line "
                    "31-37) and writes only the aggregate. Every variant is a "
                    "different aggregate of the same discarded list. The points "
                    "had to be regenerated from the deterministic seed formula, "
                    "and are trustworthy only because the re-run reproduces "
                    "every published mean, SD and count."),
  },
  {
    "claim": ("The re-run disagrees with the published table: a replication "
              "returned n=44 where the table reports n=284."),
    "why_it_appealed": ("A factor of roughly six against a published value is "
                        "exactly what a broken reproduction looks like."),
    "how_it_died": ("The published n is the SUM over replications "
                    "(np.sum(ns), run_penetration.py:38), not a mean. Six "
                    "replications averaging 47.3 agents sum to 284. The single "
                    "run was correct and the comparison was wrong."),
  },
]
FALSIFIED_KEYED = {
  "fig13_per_replication_values_assumed_recoverable": FALSIFIED[0],
  "fig13_single_replication_n_misread_as_mismatch": FALSIFIED[1],
}


def main():
    with open(PATH) as f:
        doc = json.load(f)
    # Existing figure records are TOP-LEVEL keys (fig09_fingerprint,
    # fig10_leadtime), and the rules live under "lessons", not a top-level
    # "rules". Follow the established shape instead of introducing a parallel
    # "figures" subtree that would split the record in two places.
    assert "figures" not in doc, (
        "a 'figures' subtree appeared; the layout convention changed and this "
        "script would now write to the wrong place")
    # "lessons" is a DICT of numbered rules (rule, rule_2 ... rule_9) plus
    # "plausible_but_falsified_diagnoses" -- not a list. Appending would have
    # crashed, and guessing a shape would have been worse: it could have
    # silently created a second, parallel record.
    les = doc["lessons"]
    assert isinstance(les, dict), f"lessons is {type(les).__name__}, not dict"
    assert "rule" in les and "rule_9" in les, sorted(les)
    doc["fig13_corridor"] = REC

    for name, text in NEW_RULES:
        if name in les:
            assert les[name] == text, (
                f"{name} already exists with different text; pick the next "
                f"free number rather than overwriting a recorded rule")
        les[name] = text

    fal = les["plausible_but_falsified_diagnoses"]
    if isinstance(fal, list):
        for f_ in FALSIFIED:
            if f_ not in fal:
                fal.append(f_)
    elif isinstance(fal, dict):
        fal.update(FALSIFIED_KEYED)
    else:
        raise AssertionError(
            f"plausible_but_falsified_diagnoses is {type(fal).__name__}")

    with open(PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {PATH}")
    print(f"  figure records now: "
          f"{sorted(k for k in doc if k.startswith('fig'))}")
    print(f"  lessons/rules: {len(les)} -> {sorted(les)}")


if __name__ == "__main__":
    main()
