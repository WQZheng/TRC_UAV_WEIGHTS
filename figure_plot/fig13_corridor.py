#!/usr/bin/env python3
"""Figure 13 -- corridor-scale performance versus framework penetration.

2x2 dashboard, shared x = framework penetration p (%) with ticks 0/25/50/75/100.
  (a) Conflict rate (%): All aircraft (black, thick), Framework-equipped
      (deep blue), ORCA-controlled (grey-orange); mean +- SD.
  (b) Non-completion share (%): high demand solid, low demand dashed
      (grouped points if only endpoints are complete).
  (c) System delay among completed transits (s): mean +- SD, "conditional on
      completion" in small grey text top-right (statistical scope, kept).
  (d) Throughput (completed transits per minute): mean +- SD, y not from 0.

DATA PROVENANCE (authoritative)
  High demand (arrival 0.16): archive PENETRATION_HIGH_CENSOR.txt (preferred;
    git HEAD 5f99459 "[CENSOR] right-censoring audit"), else PENETRATION_SD.txt.
  Low demand (arrival 0.06): archive PENETRATION_LOW_CENSOR.txt (preferred),
    else PENETRATION_LOW_SD.txt.
  reps=6, horizon=400, warmup=100, K=3, seed=12345. All values parsed from the
  penetration-sweep text tables; nothing hand-entered.
  Non-completion share (panel b) = the ALL-group discard%% the CENSOR tables
  report per p (v9 5.5, lines 1037-1039: "from zero to 8.0%% at high demand and
  from 0.9 to 10.8%% at low"; every discarded aircraft exits laterally, none
  times out; right-censoring share = 0%% at every p). If only the older SD
  tables (no discard%%) are found, panel (b) falls back to the honest
  "not reconstructable" annotation and the manifest records it.
"""
import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures_generated")
os.makedirs(OUT, exist_ok=True)

# authoritative penetration sweep tables (prefer the newest CENSOR audit on Lab)
HI_FILES = ["Round1/05_results/robustness/p0_referee/PENETRATION_HIGH_CENSOR.txt",
            "Round1/05_results/robustness/p0_referee/PENETRATION_SD.txt",
            "code/plangrad_sim/PENETRATION.txt",
            "Round1/05_results/robustness/p0_referee/PENETRATION.txt"]
LO_FILES = ["Round1/05_results/robustness/p0_referee/PENETRATION_LOW_CENSOR.txt",
            "Round1/05_results/robustness/p0_referee/PENETRATION_LOW_SD.txt",
            "code/plangrad_sim/PENETRATION_LOW.txt",
            "Round1/05_results/robustness/p0_referee/PENETRATION_LOW.txt"]

C_ALL = fs.STYLE["All aircraft"]["color"]
C_EQ = fs.STYLE["Framework-equipped"]["color"]
C_OR = fs.STYLE["ORCA-controlled"]["color"]

BLOCK = re.compile(r"### p = (\d+)% equipped(.*?)(?=### p =|\n=====|\Z)", re.S)
# Accepts both the SD format  (n=269)  and the CENSOR format
# (n=269  pass=269  completion=100.0%  discard= 0.0%). discard%% is optional.
ROW = re.compile(r"(ALL|EQUIPPED|UNEQUIPPED)\s*:\s*CR=\s*([\d.]+|nan)\+-\s*"
                 r"([\d.]+|nan)%\s*Thr=\s*([\d.]+)(?:\+-\s*([\d.]+))?/min\s*"
                 r"Delay=\s*([\d.]+|nan)(?:\+-\s*([\d.]+))?s\s*"
                 r"\(n=(\d+)(?:\s+pass=\d+\s+completion=\s*([\d.]+|nan)%"
                 r"\s+discard=\s*([\d.]+|nan)%)?")


def _f(x):
    return np.nan if x in (None, "nan", "") else float(x)


def parse(path):
    """Return dict p -> {group -> dict(cr,cr_sd,thr,thr_sd,delay,delay_sd,n)}."""
    txt = open(path).read()
    out = {}
    for m in BLOCK.finditer(txt):
        p = int(m.group(1)); body = m.group(2); g = {}
        for r in ROW.finditer(body):
            grp = r.group(1)
            g[grp] = dict(cr=_f(r.group(2)), cr_sd=_f(r.group(3)),
                          thr=_f(r.group(4)), thr_sd=_f(r.group(5)),
                          delay=_f(r.group(6)), delay_sd=_f(r.group(7)),
                          n=int(r.group(8)),
                          completion=_f(r.group(9)), discard=_f(r.group(10)))
        out[p] = g
    return out


def main():
    fs.set_rc()
    hi_path = fs.find_data(*HI_FILES); lo_path = fs.find_data(*LO_FILES)
    if hi_path is None:
        raise SystemExit("no high-demand penetration file found")
    hi = parse(hi_path)
    lo = parse(lo_path) if lo_path else {}
    print("hi:", hi_path, "| lo:", lo_path)
    ps = sorted(hi.keys())

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    (a, b), (c, d) = axes

    def series(data, grp, key):
        xs, ys, es = [], [], []
        for p in sorted(data.keys()):
            if grp in data[p] and not np.isnan(data[p][grp].get(key, np.nan)):
                xs.append(p); ys.append(data[p][grp][key])
                es.append(data[p][grp].get(key + "_sd", np.nan))
        return np.array(xs), np.array(ys), np.array(es)

    # (a) conflict rate
    for grp, col, lab, lw in [("ALL", C_ALL, "All aircraft", 2.4),
                              ("EQUIPPED", C_EQ, "Framework-equipped", 1.7),
                              ("UNEQUIPPED", C_OR, "ORCA-controlled", 1.7)]:
        x, y, e = series(hi, grp, "cr")
        a.errorbar(x, y, yerr=e, marker="o", ms=4.5, lw=lw, color=col,
                   capsize=2.5, elinewidth=0.9)
    a.set_ylabel("Conflict rate, CR (%)"); a.set_xticks(ps)
    a.set_xlabel("Framework penetration, $p$ (%)"); fs.panel_label(a, "(a)")

    # (b) non-completion share = ALL-group discard% (aircraft retired after
    # warm-up that fail to reach the far exit). The CENSOR audit reports it
    # directly (right-censoring share = 0% at every p, so this is the complete
    # non-completion share, not a lower bound). v9 5.5: 0->8.0% high, 0.9->10.8%
    # low. If only the older SD tables are present, discard is NaN -> annotate.
    xh, yh, _ = series(hi, "ALL", "discard")
    xl2, yl2, _ = (series(lo, "ALL", "discard") if lo else (np.array([]),) * 3)
    have_disc = xh.size and not np.all(np.isnan(yh))
    if have_disc:
        b.plot(xh, yh, marker="o", ms=4.5, lw=2.2, color=C_ALL,
               label="high demand")
        if xl2.size and not np.all(np.isnan(yl2)):
            b.plot(xl2, yl2, marker="s", ms=4.0, lw=1.7, color=C_ALL, ls="--",
                   label="low demand")
        b.legend(loc="upper left", frameon=False, fontsize=7.5)
        b.set_ylim(0, max(12.0, np.nanmax(np.concatenate([yh, yl2]
                   if xl2.size else [yh])) + 2))
    else:
        b.text(0.5, 0.5, "discard share absent in source table\n"
               "(pre-CENSOR SD file); see manifest", transform=b.transAxes,
               ha="center", va="center", fontsize=7.5, color="0.4",
               linespacing=1.4)
    b.set_xticks(ps); b.set_xlabel("Framework penetration, $p$ (%)")
    b.set_ylabel("Non-completion share (%)")
    fs.panel_label(b, "(b)")

    # (c) system delay (ALL), conditional on completion
    x, y, e = series(hi, "ALL", "delay")
    c.errorbar(x, y, yerr=e, marker="o", ms=4.5, lw=2.2, color=C_ALL,
               capsize=2.5, elinewidth=0.9, label="high demand")
    if lo:
        xl, yl, el = series(lo, "ALL", "delay")
        c.errorbar(xl, yl, yerr=el, marker="s", ms=4.0, lw=1.7, color=C_ALL,
                   ls="--", capsize=2.5, elinewidth=0.9, label="low demand")
        c.legend(loc="upper right", frameon=False, fontsize=7.5)
    c.set_ylabel("System delay among\ncompleted transits (s)")
    c.set_xticks(ps); c.set_xlabel("Framework penetration, $p$ (%)")
    c.text(0.97, 0.97, "conditional on completion", transform=c.transAxes,
           ha="right", va="top", fontsize=7, color="0.45")
    fs.panel_label(c, "(c)")

    # (d) throughput
    x, y, e = series(hi, "ALL", "thr")
    d.errorbar(x, y, yerr=e, marker="o", ms=4.5, lw=2.2, color=C_ALL,
               capsize=2.5, elinewidth=0.9)
    d.set_ylabel("Completed transits per minute")
    d.set_xticks(ps); d.set_xlabel("Framework penetration, $p$ (%)")
    lo_y = np.nanmin(y) - 3; hi_y = np.nanmax(y) + 3
    d.set_ylim(max(0, lo_y), hi_y)
    fs.panel_label(d, "(d)")

    handles = [plt.Line2D([], [], color=col, marker="o", ms=5, lw=2, label=lab)
               for col, lab in [(C_ALL, "All"), (C_EQ, "Framework-equipped"),
                                (C_OR, "ORCA-controlled")]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(OUT, "fig13_corridor.pdf")
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
