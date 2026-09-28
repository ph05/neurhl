#!/usr/bin/env python3
"""Figures for the NeurHL paper.

Regenerates every figure in paper/figures/ from files committed in the
repository. One figure (fig_xg_drift) needs gitignored inputs, the per-team-game
tables neurhl/data/tensors/tgx_{season}.parquet. When they are present they are
aggregated to paper/figures/data/xg_drift.csv (committed); when they are absent
that CSV is used as it stands. The figure is always drawn from the CSV, so a
clone without the tensors produces byte-identical output.

Run from the repository root:

    uv run -q --no-project --python 3.12 --with numpy --with "pandas<3" \
        --with pyarrow --with matplotlib --with scipy \
        python paper/figures/make_figures.py

Each figure is written twice: a vector PDF with TrueType fonts embedded (for
LaTeX) and a 200-dpi PNG (for the web). Nothing is random and no timestamp is
written, so repeated runs give identical files. Every number that a caption or
the ledger quotes is recomputed and checked; mismatches are printed at the end.

Colours follow paper/main.tex: NeurHL-G #0072B2 (gcol), NeurHL-H #D55E00
(hcol), Elo #7F7F7F (elocol).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True     # importing neurhl/windows.py must not write caches there

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# ------------------------------------------------------------------ paths
FIG = Path(__file__).resolve().parent
ROOT = FIG.parents[1]
DATA = FIG / "data"
NEURHL = ROOT / "neurhl"
OUTPUT = NEURHL / "output"
PREDS = OUTPUT / "preds"
CONFIGS = NEURHL / "configs"
TENSORS = NEURHL / "data" / "tensors"

Z95 = 1.959963984540054
FULL, HALF = 6.3, 3.1                       # LaTeX text width and half width, inches

# ------------------------------------------------------------------ palette
C_G = "#0072B2"        # NeurHL-G (paper: gcol)
C_H = "#D55E00"        # NeurHL-H (paper: hcol)
C_ELO = "#7F7F7F"      # Elo (paper: elocol)
C_LAYER = "#009E73"    # NeurHL 1.0 player-game layer
C_XG = "#CC79A7"       # house expected goals
C_OBS = "#000000"      # observed outcomes, reference lines
C_MUTED = "#A0A0A0"    # excluded / sensitivity-only estimates
C_SEP = "#C8C8C8"      # separators
# season roles, light to dark = earliest to most protected use (one hue)
RAMP = ["#ECEFF3", "#C5D1DE", "#8EA3BA", "#4A6785", "#1B2F45"]
HATCH_EDGE = "#6E6E6E"

CHECKS = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    CHECKS.append((label, got, want, tol, ok))
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: got {got:.6f}, expected {want:.6f} (tol {tol:g})")
    return ok


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.2,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "axes.labelpad": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "axes.axisbelow": True,
        "text.color": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.8,
        "errorbar.capsize": 0,
        "hatch.linewidth": 0.5,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    })


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=200, metadata={"Software": None})
    plt.close(fig)
    print(f"  wrote {name}.pdf, {name}.png")


# ------------------------------------------------------------------ helpers
def load_windows():
    spec = importlib.util.spec_from_file_location("neurhl_windows", NEURHL / "windows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nll(p, y):
    """Per-game log loss in nats, clipped as in the repository (eval/run_g.py)."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_ci(d):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return m, se, m - Z95 * se, m + Z95 * se


def wilson(k, n, z=Z95):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def equal_count_bins(pred, obs, k):
    """k equal-count bins of `pred` (stable sort); mean pred, mean obs, n, 95% CI."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    order = np.argsort(pred, kind="mergesort")
    rows = []
    for idx in np.array_split(order, k):
        p, y = pred[idx], obs[idx]
        rows.append({"pred": p.mean(), "obs": y.mean(), "n": len(idx),
                     "se": y.std(ddof=1) / np.sqrt(len(idx))})
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(p, y):
    """Logistic regression of y on logit(p) with intercept (Newton-Raphson)."""
    X = np.column_stack([np.ones(len(p)), logit(p)])
    y = np.asarray(y, float)
    b = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ b))
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    cov = np.linalg.inv(H)
    return b[1], np.sqrt(cov[1, 1])


def panel_tag(ax, tag, x=-0.02, y=1.0):
    ax.text(x, y, tag, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def errpoint(ax, x, y, lo, hi, color, marker="o", ms=4.2, filled=True, lw=1.0,
             zorder=3, label=None):
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]] if np.ndim(y) == 0 else [y - lo, hi - y],
                fmt="none", ecolor=color, elinewidth=lw, zorder=zorder)
    ax.plot(x, y, marker, ms=ms, color=color, mfc=color if filled else "white",
            mec=color, mew=0.9, zorder=zorder + 1, label=label, ls="none")


# =================================================================== 1 windows
def fig_windows(W):
    print("fig_windows")
    seasons = list(range(W.TRAIN_FROM, W.G_LIVE + 1))
    assert W.PROJECT == W.G_LIVE

    def role_v1(s):
        if s in W.NO_SCORE:
            return "noscore"
        if s in W.DEV:
            return "dev"
        if s in W.TUNE:
            return "tune"
        if s in W.CONFIRM:
            return "confirm"
        if s == W.PROJECT:
            return "live"
        return "train"                   # corpus start, training only

    def role_g(s):
        if s in W.NO_SCORE:
            return "noscore"
        if s in W.G_BURN_IN:
            return "burn"
        if s in W.G_ITER:
            return "iter"
        if s in W.G_GATE:
            return "gate"
        if s in W.SEALED:
            return "sealed"
        if s == W.G_LIVE:
            return "live"
        raise ValueError(f"season {s} has no NeurHL-G role in windows.py")

    level = {"train": 0, "burn": 0, "dev": 1, "iter": 1, "tune": 2, "gate": 2,
             "confirm": 3, "sealed": 3, "live": 4}
    rows = [
        ("NeurHL 1.0", role_v1, {"train": "train", "dev": "development", "tune": "tune",
                                 "confirm": "confirmation (one shot)", "live": "live"}),
        ("NeurHL-G", role_g, {"burn": "burn-in", "iter": "G_ITER", "gate": "G_GATE",
                              "sealed": "SEALED", "live": "live"}),
    ]
    fig = plt.figure(figsize=(FULL, 1.75))
    ax = fig.add_axes([0.115, 0.36, 0.875, 0.60])
    ys = [1.0, 0.0]
    h, gap = 0.66, 0.07
    for (name, role, labels), yc in zip(rows, ys):
        roles = [role(s) for s in seasons]
        runs, start = [], 0                    # contiguous seasons with the same role
        for i in range(1, len(roles) + 1):
            if i == len(roles) or roles[i] != roles[start]:
                runs.append((roles[start], seasons[start], seasons[i - 1]))
                start = i
        for r, a, b in runs:
            x0, wd = a - 0.5 + gap / 2, b - a + 1 - gap
            if r == "noscore":
                ax.add_patch(Rectangle((x0, yc - h / 2), wd, h, facecolor="white",
                                       edgecolor=HATCH_EDGE, hatch="/////", lw=0.0))
                ax.add_patch(Rectangle((x0, yc - h / 2), wd, h, facecolor="none",
                                       edgecolor=HATCH_EDGE, lw=0.5))
                continue
            ax.add_patch(Rectangle((x0, yc - h / 2), wd, h, facecolor=RAMP[level[r]],
                                   edgecolor="none"))
            for s in range(a, b):              # season boundaries inside a run: edge notches
                for y0, y1 in ((yc - h / 2, yc - h / 2 + 0.14), (yc + h / 2 - 0.14, yc + h / 2)):
                    ax.plot([s + 0.5] * 2, [y0, y1], color="white", lw=0.8,
                            solid_capstyle="butt")
        # label each role once, centred on its longest contiguous run
        for r in labels:
            cand = [x for x in runs if x[0] == r]
            if not cand:
                raise ValueError(f"{name}: no season has role {r}")
            _, a, b = max(cand, key=lambda x: (x[2] - x[1], -x[1]))
            ax.text((a + b) / 2, yc, labels[r], ha="center", va="center", fontsize=8,
                    color="white" if level[r] >= 3 else "#1A1A1A")
    ax.set_xlim(seasons[0] - 0.55, seasons[-1] + 0.55)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
    ax.set_xticks(seasons, [str(s) for s in seasons], fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", pad=3)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Season (named by the year it ends)", labelpad=4)
    why = ", ".join(f"{y} {W.SEASON_NOTES[y].split(':')[0].split()[-1]}"
                    for y in sorted(W.NO_SCORE))            # "2013 lockout, 2021 COVID"
    hatch = Patch(facecolor="white", edgecolor=HATCH_EDGE, hatch="/////", lw=0.5,
                  label=f"trained on, never scored ({why})")
    fig.legend(handles=[hatch], loc="lower left", bbox_to_anchor=(0.115 - 0.004, 0.0),
               handlelength=1.4, handleheight=1.0, fontsize=8)
    save(fig, "fig_windows")


# =================================================================== 2 C1 by season
def fig_c1_seasons(W):
    print("fig_c1_seasons")
    js = json.loads((OUTPUT / "hier_restatement.json").read_text())
    prim, inc = js["primary"], js["including_2021"]
    g = pd.read_csv(PREDS / "hier_restatement_games.csv")
    g["d"] = nll(g.p_neurhl_h, g.y) - nll(g.p_elo, g.y)
    # the JSON carries per-season 95% CIs; recompute them from the per-game file
    worst = 0.0
    for s, grp in g.groupby("season"):
        m, _, lo, hi = mean_ci(grp.d)
        rec = inc["per_season"][str(int(s))]
        assert rec["n"] == len(grp)
        worst = max(worst, abs(m - rec["diff"]), abs(lo - rec["ci95"][0]), abs(hi - rec["ci95"][1]))
    check("C1 per-season diff/CI, per-game file vs JSON (max abs gap)", worst, 0.0, 1e-6)
    P = g[~g.season.isin(W.NO_SCORE)]
    m, se, lo, hi = mean_ci(P.d)
    check("C1 pooled n", len(P), 10184, 0)
    check("C1 pooled diff", m, -0.00478, 5e-6)
    check("C1 pooled CI low", lo, -0.00650, 5e-6)
    check("C1 pooled CI high", hi, -0.00306, 5e-6)
    check("C1 pooled diff, JSON", prim["diff"], m, 1e-6)

    K = 1e3
    fig, ax = plt.subplots(figsize=(FULL, 2.9), layout="constrained")
    seasons = sorted(int(s) for s in inc["per_season"])
    x_pool, x_pool21, x_sep = seasons[-1] + 1.45, seasons[-1] + 2.75, seasons[-1] + 0.7
    ax.fill_between([seasons[0] - 0.4, seasons[-1] + 0.4], prim["ci95"][0] * K,
                    prim["ci95"][1] * K, color=C_H, alpha=0.10, lw=0, zorder=1)
    ax.plot([seasons[0] - 0.4, seasons[-1] + 0.4], [prim["diff"] * K] * 2, color=C_H, lw=0.7,
            ls=(0, (4, 2)), zorder=2)
    ax.axhline(0, color=C_OBS, lw=0.6, zorder=2)
    ax.axvline(x_sep, color=C_SEP, lw=0.6, zorder=1)
    for s in seasons:
        rec = inc["per_season"][str(s)]
        excluded = s in W.NO_SCORE
        errpoint(ax, s, rec["diff"] * K, rec["ci95"][0] * K, rec["ci95"][1] * K,
                 C_MUTED if excluded else C_H, filled=not excluded)
    errpoint(ax, x_pool, prim["diff"] * K, prim["ci95"][0] * K, prim["ci95"][1] * K, C_H,
             marker="D", ms=4.6, lw=1.3)
    errpoint(ax, x_pool21, inc["diff"] * K, inc["ci95"][0] * K, inc["ci95"][1] * K, C_MUTED,
             marker="D", ms=4.6, filled=False, lw=1.3)
    labels = [f"{s}\n{inc['per_season'][str(s)]['n']:,}" for s in seasons]
    labels += [f"Pooled\n{prim['n_games']:,}", f"With 2021\n{inc['n_games']:,}"]
    ax.set_xticks(seasons + [x_pool, x_pool21], labels)
    for t in ax.get_xticklabels():
        t.set_linespacing(1.15)
    ax.set_xlim(seasons[0] - 0.6, x_pool21 + 0.65)
    ax.set_ylim(-13.2, 4.2)
    ax.set_yticks(np.arange(-12, 4.1, 2))
    ax.set_ylabel("NeurHL-H $-$ Elo log loss\n(10$^{-3}$ nats per game)")
    ax.set_xlabel("Season and number of games (confirmation window)")
    ax.annotate("", xy=(seasons[0] - 0.42, -12.6), xytext=(seasons[0] - 0.42, -9.9),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.6,
                                color="#333333", shrinkA=0, shrinkB=0))
    ax.text(seasons[0] - 0.28, -11.25, "NeurHL-H\nbetter", fontsize=8, va="center",
            ha="left", color="#333333", linespacing=1.0)
    handles = [
        Line2D([], [], marker="o", color=C_H, mfc=C_H, ls="none", ms=4.2,
               label="Primary-test season, 95% CI"),
        Line2D([], [], marker="o", color=C_MUTED, mfc="white", ls="none", ms=4.2,
               label="2021 (excluded)"),
        (Patch(facecolor=C_H, alpha=0.10, lw=0),
         Line2D([], [], color=C_H, lw=0.7, ls=(0, (4, 2)))),
    ]
    ax.legend(handles=handles, labels=[h.get_label() for h in handles[:2]]
              + ["Pooled primary estimate, 95% CI"], loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    save(fig, "fig_c1_seasons")
    return P


# =================================================================== 3 reliability
def fig_reliability(W):
    print("fig_reliability")
    g = pd.read_csv(PREDS / "hier_restatement_games.csv")
    P = g[~g.season.isin(W.NO_SCORE)]
    check("reliability n", len(P), 10184, 0)
    base = P.y.mean()
    fig = plt.figure(figsize=(HALF, 3.55), layout="constrained")
    hr = 0.3                                 # histogram height / width
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, hr])
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)
    ax.set_box_aspect(1.0)                   # square panel; equal widths keep x aligned
    axh.set_box_aspect(hr)
    lo, hi = 0.15, 0.87
    ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=0.6, zorder=1)
    ax.axhline(base, color=C_OBS, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    specs = [("p_neurhl_h", "NeurHL-H", C_H, "o"), ("p_elo", "Elo", C_ELO, "s")]
    for col, name, c, mk in specs:
        b = equal_count_bins(P[col], P.y, 10)
        k = (b.obs * b.n).round().to_numpy()
        wl, wh = wilson(k, b.n.to_numpy())
        ax.errorbar(b.pred, b.obs, yerr=[b.obs - wl, wh - b.obs], fmt="none", ecolor=c,
                    elinewidth=0.9, zorder=3)
        ax.plot(b.pred, b.obs, "-", color=c, lw=0.8, zorder=3)
        ax.plot(b.pred, b.obs, mk, color=c, ms=3.8 if mk == "o" else 3.4, mec="white",
                mew=0.5, zorder=4, label=name)
        edges = np.arange(0.10, 0.9001, 0.02)
        axh.hist(P[col], bins=edges, histtype="step", color=c, lw=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(np.arange(0.2, 0.81, 0.1))
    ax.set_yticks(np.arange(0.2, 0.81, 0.1))
    ax.set_ylabel("Observed home-win frequency")
    ax.tick_params(labelbottom=False)
    handles = [Line2D([], [], color=C_H, marker="o", ms=3.8, mec="white", mew=0.5, lw=0.8,
                      label="NeurHL-H"),
               Line2D([], [], color=C_ELO, marker="s", ms=3.4, mec="white", mew=0.5, lw=0.8,
                      label="Elo"),
               Line2D([], [], color=C_OBS, lw=0.6, label="Perfect calibration"),
               Line2D([], [], color=C_OBS, lw=0.8, ls=(0, (4, 2.5)),
                      label=f"Home-win rate ({base:.3f})")]
    ax.legend(handles=handles, loc="upper left", handlelength=2.0)
    axh.set_xlabel("Predicted home-win probability")
    axh.set_ylabel("Games")
    axh.set_ylim(0, None)
    axh.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    save(fig, "fig_reliability")


# =================================================================== 4 G_GATE by season
def sstop_thresholds():
    src = (NEURHL / "eval" / "gate_g.py").read_text()
    te = float(re.search(r'"g_minus_elo"\]\s*<=\s*(-?[0-9.]+)', src).group(1))
    th = float(re.search(r'"g_minus_h"\]\s*<=\s*(-?[0-9.]+)', src).group(1))
    return te, th


def fig_g_gate_seasons(W):
    print("fig_g_gate_seasons")
    g = pd.read_csv(PREDS / "g_gate_games.csv")
    assert sorted(g.season_end.unique()) == list(W.G_GATE)
    assert (g.y == g.outcome4.isin([0, 2]).astype(float)).all()
    for col, want in (("p_stack", 0.66006), ("p_elo", 0.66566), ("p_h", 0.66056)):
        check(f"G_GATE pooled log loss {col}", nll(g[col], g.y).mean(), want, 5e-6)
    check("G_GATE n", len(g), 6289, 0)
    de = nll(g.p_stack, g.y) - nll(g.p_elo, g.y)
    dh = nll(g.p_stack, g.y) - nll(g.p_h, g.y)
    rec = json.loads((OUTPUT / "g_gates.json").read_text())["S_STOP"]
    check("G_GATE G-Elo vs g_gates.json", de.mean(), rec["g_minus_elo"], 1e-6)
    check("G_GATE G-H vs g_gates.json", dh.mean(), rec["g_minus_h"], 1e-6)
    check("G_GATE SE G-Elo vs g_gates.json", mean_ci(de)[1], rec["se_g_minus_elo"], 1e-6)
    check("G_GATE SE G-H vs g_gates.json", mean_ci(dh)[1], rec["se_g_minus_h"], 1e-6)
    t_elo, t_h = sstop_thresholds()
    check("S-STOP threshold vs Elo (gate_g.py)", t_elo, -0.0045, 0)
    check("S-STOP threshold vs NeurHL-H (gate_g.py)", t_h, -0.0010, 0)

    K = 1e3
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(FULL, 2.9), sharey=True, layout="constrained",
                                  gridspec_kw={"width_ratios": [4.1, 1.65]})
    off = 0.14
    for s in W.G_GATE:
        m = g.season_end == s
        for d, dx, filled in ((de, -off, True), (dh, off, False)):
            mm, _, lo, hi = mean_ci(d[m])
            errpoint(ax, s + dx, mm * K, lo * K, hi * K, C_G, marker="o" if filled else "s",
                     ms=4.2 if filled else 3.9, filled=filled)
        check(f"G_GATE {s} per-season G-Elo vs g_gates.json", de[m].mean(),
              rec["per_season"][str(s)]["de"], 1e-6)
    for a in (ax, axp):
        a.axhline(0, color=C_OBS, lw=0.6, zorder=1)
    ax.set_xlim(min(W.G_GATE) - 0.6, max(W.G_GATE) + 0.6)
    gap = [s for s in range(min(W.G_GATE), max(W.G_GATE)) if s not in W.G_GATE]
    for s in gap:
        ax.text(s, 2.4, f"{s}\nnot scored", ha="center", va="center", fontsize=8,
                color="#666666", linespacing=1.0)
    if gap:
        xa = gap[0] - 0.3
        ax.annotate("", xy=(xa, -13.6), xytext=(xa, -10.6),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.6,
                                    color="#333333", shrinkA=0, shrinkB=0))
        ax.text(xa + 0.12, -12.1, "NeurHL-G\nbetter", fontsize=8, va="center", ha="left",
                color="#333333", linespacing=1.0)
    n_by = g.season_end.value_counts()
    ax.set_xticks(W.G_GATE, [f"{s}\n{n_by[s]:,}" for s in W.G_GATE])
    ax.set_xlabel("Season and number of games (G_GATE)")
    ax.set_ylabel("Log-loss difference\n(10$^{-3}$ nats per game)")
    # pooled panel with the S-STOP thresholds
    for x, d, filled, thr in ((0, de, True, t_elo), (1, dh, False, t_h)):
        mm, se, lo, hi = mean_ci(d)
        errpoint(axp, x, mm * K, lo * K, hi * K, C_G, marker="o" if filled else "s",
                 ms=4.8 if filled else 4.4, filled=filled, lw=1.3)
        axp.plot([x - 0.34, x + 0.34], [thr * K] * 2, color=C_OBS, lw=0.9, ls=(0, (1, 1.4)),
                 zorder=2)
        axp.text(x + 0.38, thr * K, f"{thr * K:+.1f}".replace("-", "−"), fontsize=8,
                 va="center", ha="left", color="#222222")
    axp.set_xticks([0, 1], ["vs\nElo", "vs\nNeurHL-H"])
    axp.set_xlim(-0.5, 1.95)
    axp.set_xlabel(f"Pooled, n = {len(g):,}")
    axp.tick_params(axis="y", left=False)
    axp.spines["left"].set_visible(False)
    for t in ax.get_xticklabels() + axp.get_xticklabels():
        t.set_linespacing(1.15)
    ax.set_ylim(-15.2, 4.0)
    ax.set_yticks(np.arange(-14, 4.1, 2))
    handles = [
        Line2D([], [], marker="o", color=C_G, mfc=C_G, ls="none", ms=4.2,
               label="NeurHL-G $-$ Elo, 95% CI"),
        Line2D([], [], marker="s", color=C_G, mfc="white", ls="none", ms=3.9,
               label="NeurHL-G $-$ NeurHL-H, 95% CI"),
        Line2D([], [], color=C_OBS, lw=0.9, ls=(0, (1, 1.4)), label="S-STOP threshold"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    fig.align_xlabels([ax, axp])
    save(fig, "fig_g_gate_seasons")
    return de, dh


# =================================================================== 5 G_ITER ladder
RUNGS = [("r6", "R6", "network"), ("r6e", "R6e", "Elo anchor"),
         ("r6L", "R6L", "lineup\nmultipliers"), ("r6c", "R6c", "H projections\nas inputs"),
         ("r13", "R13", "R6c + H\nterms"), ("r9", "R9", "attention"),
         ("r11", "R11", "R6 + H\nin stack")]


def plan_candidate_table():
    """Stack log loss and decision per rung from PLAN_NeurHL4.md, section CANDIDATE."""
    txt = (ROOT / "PLAN_NeurHL4.md").read_text()
    sec = txt.split("## CANDIDATE", 1)[1].split("\n## ", 1)[0]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\|\s*(R\d+\w*)\s[^|]*\|\s*(0\.\d{5})\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), m.group(3))
    ref = re.search(r"Elo (0\.\d{5}),\s*NeurHL-H (0\.\d{5})", sec)
    return out, float(ref.group(1)), float(ref.group(2))


def fig_g_ladder(W):
    print("fig_g_ladder")
    L = pd.read_csv(PREDS / "g_iter_ladder_games.csv")
    assert sorted(L.season_end.unique()) == list(W.G_ITER)
    led = pd.read_csv(CONFIGS / "search_ledger_g.csv")
    plan, plan_elo, plan_h = plan_candidate_table()
    ll = {}
    for key in ["elo", "h", "r4_quick"] + [r[0] for r in RUNGS]:
        ok = L[f"p_{key}"].notna()
        ll[key] = nll(L.loc[ok, f"p_{key}"], L.loc[ok, "y"]).mean()
    check("G_ITER n", len(L), 7421, 0)
    check("G_ITER Elo vs PLAN_NeurHL4", ll["elo"], plan_elo, 5e-6)
    check("G_ITER NeurHL-H vs PLAN_NeurHL4", ll["h"], plan_h, 5e-6)
    rows = []
    for key, tag, _ in RUNGS:
        lrow = led[led.run_id == f"{key}:full:iter"].iloc[-1]     # latest full run
        check(f"G_ITER {tag} vs ledger ({lrow.git})", ll[key], lrow.ll_stack, 5e-7)
        check(f"G_ITER {tag} vs PLAN_NeurHL4", round(ll[key], 5), plan[tag][0], 0)
        check(f"G_ITER {tag} ledger Elo", lrow.ll_elo, ll["elo"], 5e-7)
        d = nll(L[f"p_{key}"], L.y) - nll(L.p_h, L.y)
        rows.append({"key": key, "tag": tag, "ll": ll[key], "decision": plan[tag][1],
                     "se_vs_h": mean_ci(d)[1]})
    lq = led[led.run_id == "r4:quick:iter"].iloc[-1]
    check("G_ITER R4 quick vs ledger", ll["r4_quick"], lq.ll_stack, 5e-7)
    R = pd.DataFrame(rows)
    print("  paired SE vs NeurHL-H by rung:", ", ".join(f"{r.tag} {r.se_vs_h:.5f}" for r in R.itertuples()))

    fig, ax = plt.subplots(figsize=(FULL, 2.9), layout="constrained")
    x = np.arange(len(R))
    x0 = -0.55
    ax.axhline(ll["elo"], color=C_ELO, lw=1.1, zorder=1)
    ax.axhline(ll["h"], color=C_H, lw=1.1, zorder=1)
    ax.text(x0 + 0.05, ll["elo"] + 0.00005, f"Elo {ll['elo']:.5f}", ha="left", va="bottom",
            fontsize=8, color="#555555")
    ax.text(x0 + 0.05, ll["h"] - 0.00005, f"NeurHL-H {ll['h']:.5f}", ha="left", va="top",
            fontsize=8, color=C_H)
    style_of = {"parent": ("o", True, 4.8), "carried forward": ("D", True, 5.2)}
    for xi, r in zip(x, R.itertuples()):
        mk, filled, ms = style_of.get(r.decision, ("o", False, 4.8))
        ax.plot(xi, r.ll, mk, color=C_G, mfc=C_G if filled else "white", mec=C_G, mew=1.0,
                ms=ms, zorder=3)
        ax.text(xi, r.ll + 0.00012, f"{r.ll:.5f}", ha="center", va="bottom", fontsize=8,
                color="#222222")
    ax.set_xticks(x, [rf"$\mathbf{{{r.tag}}}$" + f"\n{lab}"
                      for r, (_, _, lab) in zip(R.itertuples(), RUNGS)])
    for t in ax.get_xticklabels():
        t.set_linespacing(1.1)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(x0, len(R) - 0.45)
    ax.set_ylim(0.6720, 0.6758)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.0005))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.set_ylabel("Log loss of the stacked\nprobability (nats per game)")
    ax.set_xlabel(f"Full-run rung on G_ITER ({len(L):,} games)")
    handles = [Line2D([], [], marker="o", color=C_G, mfc=C_G, ls="none", ms=4.8, label="parent"),
               Line2D([], [], marker="o", color=C_G, mfc="white", ls="none", ms=4.8,
                      label="rejected"),
               Line2D([], [], marker="D", color=C_G, mfc=C_G, ls="none", ms=5.2,
                      label="carried forward (candidate g1)")]
    ax.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 0.86))
    save(fig, "fig_g_ladder")
    return R, ll


# =================================================================== 6 player heads
HEADS = [("toi_share", "toi_share", "TOI share\n(MAE)"),
         ("shots", "sog", "Shots on goal\n(Poisson loss)"),
         ("goal1", "p_goal", "P(goal)\n(log loss)"),
         ("assist1", "p_assist", "P(assist)\n(log loss)")]


def fig_player_heads():
    print("fig_player_heads")
    tw = json.loads((CONFIGS / "player_game_twoway_2018_2020.json").read_text())
    conf = json.loads((CONFIGS / "player_game_confirm_2018_2020.json").read_text())
    pg = json.loads((OUTPUT / "g_gates.json").read_text())["PG"]
    left, right = [], []
    for k1, k2, _ in HEADS:
        h = tw["heads"][k1]
        assert h["reproduces_record"] and abs(h["mean_diff"] - conf["gates"][k1]["mean_diff"]) < 1e-6
        rel = 100 * h["relative_to_baseline"]
        half = Z95 * abs(rel / h["z_twoway"])
        left.append((rel, rel - half, rel + half))
        r = pg["heads"][k2]
        rel2, up = 100 * r["rel_diff"], 100 * r["rel_upper95"]
        right.append((rel2, 2 * rel2 - up, up))
    for (k1, _, _), want, (v, _, _) in zip(HEADS, (-4.1, -10.2, -2.8, -2.6), left):
        check(f"player-game layer {k1} relative change (%) vs EVIDENCE.md", round(v, 1), want, 0)
    for (_, k2, _), want, (v, _, _) in zip(HEADS, (-0.59, -0.45, -0.45, -0.36), right):
        check(f"NeurHL-G PG {k2} relative change (%) vs PLAN_NeurHL4", round(v, 2), want, 0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True, layout="constrained")
    y = np.arange(len(HEADS))[::-1]
    for ax, vals, c, fmt in ((a1, left, C_LAYER, "{:.1f}%"), (a2, right, C_G, "{:.2f}%")):
        ax.axvline(0, color=C_OBS, lw=0.6, zorder=1)
        for yi, (v, lo, hi) in zip(y, vals):
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=c, elinewidth=1.1,
                        zorder=3)
            ax.plot(v, yi, "o", color=c, ms=4.4, zorder=4)
            ax.text(v, yi + 0.17, fmt.format(v).replace("-", "−"), ha="center",
                    va="bottom", fontsize=8, color="#222222")
    a1.set_yticks(y, [h[2] for h in HEADS])
    for t in a1.get_yticklabels():
        t.set_linespacing(1.0)
    a1.tick_params(axis="y", length=0)
    a2.tick_params(axis="y", length=0)
    a1.set_ylim(-0.55, len(HEADS) - 0.4)
    a1.set_xlim(-12.5, 1.0)
    a2.set_xlim(-0.95, 1.25)
    a2.axvline(1.0, color=C_OBS, lw=0.9, ls=(0, (1, 1.4)), zorder=1)
    a2.text(0.96, len(HEADS) - 0.55, "PG gate\nmargin +1%", ha="right", va="top", fontsize=8,
            color="#222222", linespacing=1.0)
    a1.set_xlabel("Change in loss vs own recent average (%)")
    a2.set_xlabel("Change in loss vs player-game layer (%)")
    a1.text(-12.3, -0.4, "NeurHL 1.0 player-game layer,\n2018–2020 confirmation",
            ha="left", va="bottom", fontsize=8, color=C_LAYER, linespacing=1.05)
    a2.text(0.94, -0.4, "NeurHL-G player heads,\nG_GATE",
            ha="right", va="bottom", fontsize=8, color=C_G, linespacing=1.05)
    for ax in (a1, a2):
        ax.spines["left"].set_visible(False)
    panel_tag(a1, "(a)", x=-0.01, y=1.0)
    panel_tag(a2, "(b)", x=-0.01, y=1.0)
    save(fig, "fig_player_heads")
    return tw, pg


# =================================================================== 7 xG drift
XG_SEASONS = list(range(2010, 2027))


def xg_drift_table():
    csv = DATA / "xg_drift.csv"
    files = [TENSORS / f"tgx_{s}.parquet" for s in XG_SEASONS]
    if all(f.exists() for f in files):
        rows = []
        for s, f in zip(XG_SEASONS, files):
            d = pd.read_parquet(f, columns=["game_id", "is_home", "gf_all", "xgf_all", "en_gf",
                                            "gf_ev", "gf_pp", "gf_sh", "xgf_ev", "xgf_pp",
                                            "xgf_sh"]).astype({"gf_all": float, "xgf_all": float})
            assert (d.groupby("game_id").is_home.agg(["count", "sum"]) == [2, 1]).all().all()
            d["xgf_str"] = d.xgf_ev.astype(float) + d.xgf_pp + d.xgf_sh
            d["gf_str"] = d.gf_ev.astype(float) + d.gf_pp + d.gf_sh
            G = d.groupby("game_id")[["gf_all", "xgf_all", "en_gf", "gf_str", "xgf_str"]].sum()
            n = len(G)
            gm, xm = G.gf_all.mean(), G.xgf_all.mean()
            vg, vx = G.gf_all.var(ddof=1), G.xgf_all.var(ddof=1)
            cgx = np.cov(G.gf_all, G.xgf_all, ddof=1)[0, 1]
            ratio = gm / xm
            ratio_se = ratio * np.sqrt(vg / (n * gm ** 2) + vx / (n * xm ** 2)
                                       - 2 * cgx / (n * gm * xm))
            rows.append({"season": s, "n_games": n, "n_team_games": 2 * n,
                         "goals_pg": gm / 2, "goals_se": np.sqrt(vg / n) / 2,
                         "xg_pg": xm / 2, "xg_se": np.sqrt(vx / n) / 2,
                         "ratio": ratio, "ratio_se": ratio_se,
                         "en_goals_pg": G.en_gf.mean() / 2,
                         "goals_no_en_pg": G.gf_str.mean() / 2,
                         "xg_no_en_pg": G.xgf_str.mean() / 2})
        DATA.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.6f")
        print(f"  aggregated {len(files)} tensor tables to {csv.relative_to(ROOT)}")
    elif csv.exists():
        print(f"  tensors absent; using {csv.relative_to(ROOT)}")
    else:
        raise FileNotFoundError("neither the tgx tensors nor paper/figures/data/xg_drift.csv exist")
    # always plot from the committed CSV, so a clone without tensors draws the same figure
    return pd.read_csv(csv)


def fig_xg_drift():
    print("fig_xg_drift")
    T = xg_drift_table()
    t = T.set_index("season")
    for s, g_, x_ in ((2024, 3.08, 2.89), (2025, 3.01, 3.03), (2026, 3.08, 3.32)):
        check(f"goals per team-game {s}", round(t.goals_pg[s], 2), g_, 0)
        check(f"xG per team-game {s}", round(t.xg_pg[s], 2), x_, 0)
    check("goals/xG ratio 2024", round(t.ratio[2024], 3), 1.067, 0)
    check("goals/xG ratio 2026", round(t.ratio[2026], 3), 0.927, 0)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(FULL, 3.2), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [1.55, 1.0]})
    s = T.season.to_numpy()
    for col, se, c, mk, lab in (("goals_pg", "goals_se", C_OBS, "o", "Goals"),
                                ("xg_pg", "xg_se", C_XG, "s", "House xG")):
        v, e = T[col].to_numpy(), T[se].to_numpy()
        a1.fill_between(s, v - Z95 * e, v + Z95 * e, color=c, alpha=0.13, lw=0)
        a1.plot(s, v, "-", color=c, lw=1.0)
        a1.plot(s, v, mk, color=c, ms=3.6 if mk == "o" else 3.3, mec="white", mew=0.5, label=lab)
    a1.set_ylabel("Goals per team-game")
    a1.set_ylim(2.45, 3.45)
    a1.legend(loc="upper left", ncol=2)
    r, re_ = T.ratio.to_numpy(), T.ratio_se.to_numpy()
    a2.axhline(1.0, color=C_OBS, lw=0.6)
    a2.fill_between(s, r - Z95 * re_, r + Z95 * re_, color=C_OBS, alpha=0.12, lw=0)
    a2.plot(s, r, "-o", color=C_OBS, lw=1.0, ms=3.4, mec="white", mew=0.5)
    for yr, va, dy in ((2024, "bottom", 0.012), (2026, "top", -0.012)):
        a2.text(yr, t.ratio[yr] + dy + (Z95 * t.ratio_se[yr] if dy > 0 else -Z95 * t.ratio_se[yr]),
                f"{t.ratio[yr]:.3f}", ha="center", va=va, fontsize=8)
    a2.set_ylabel("Goals / xG")
    a2.set_ylim(0.86, 1.12)
    a2.set_xlabel("Season (named by the year it ends)")
    a2.set_xticks(np.arange(2010, 2027, 2))
    a2.set_xlim(2009.5, 2026.5)
    fig.align_ylabels([a1, a2])
    panel_tag(a1, "(a)", x=-0.075)
    panel_tag(a2, "(b)", x=-0.075)
    save(fig, "fig_xg_drift")
    return T


# =================================================================== 8 outcome calibration
def fig_outcome_calib(W):
    print("fig_outcome_calib")
    g = pd.read_csv(PREDS / "g_gate_games.csv")
    rec = json.loads((OUTPUT / "g_gates.json").read_text())["C"]
    tie = g.outcome4.isin([2, 3]).astype(float)       # 2/3 = home/away win in OT or shootout
    check("OT share predicted (pooled)", g.p_tie.mean(), 0.224, 5e-4)
    check("OT share observed (pooled)", tie.mean(), 0.220, 5e-4)
    check("OT share predicted vs g_gates.json", g.p_tie.mean(), rec["ot_share"]["pred"], 1e-6)
    slope, slope_se = calibration_slope(g.p_stack, g.y)
    check("calibration slope vs g_gates.json", slope, rec["slope"], 2e-3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.9), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.0, 1.35]})
    lo, hi = 0.15, 0.88
    base = g.y.mean()
    a1.plot([lo, hi], [lo, hi], color=C_OBS, lw=0.6, zorder=1)
    a1.axhline(base, color=C_OBS, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    b = equal_count_bins(g.p_stack, g.y, 10)
    k = (b.obs * b.n).round().to_numpy()
    wl, wh = wilson(k, b.n.to_numpy())
    a1.errorbar(b.pred, b.obs, yerr=[b.obs - wl, wh - b.obs], fmt="none", ecolor=C_G,
                elinewidth=0.9, zorder=3)
    a1.plot(b.pred, b.obs, "-o", color=C_G, lw=0.8, ms=3.8, mec="white", mew=0.5, zorder=4)
    a1.text(0.97, 0.03, f"calibration slope {slope:.3f}\n95% CI {slope - Z95 * slope_se:.3f}"
                        f"–{slope + Z95 * slope_se:.3f}",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8, linespacing=1.1)
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_aspect("equal", adjustable="box")
    a1.set_xticks(np.arange(0.2, 0.81, 0.2))
    a1.set_yticks(np.arange(0.2, 0.81, 0.2))
    a1.set_xlabel("Predicted home-win probability")
    a1.set_ylabel("Observed home-win frequency")
    a1.legend(handles=[Line2D([], [], color=C_G, marker="o", ms=3.8, mec="white", mew=0.5,
                              lw=0.8, label="NeurHL-G (stacked)"),
                       Line2D([], [], color=C_OBS, lw=0.6, label="Perfect calibration"),
                       Line2D([], [], color=C_OBS, lw=0.8, ls=(0, (4, 2.5)),
                              label=f"Home-win rate ({base:.3f})")],
              loc="upper left", handlelength=2.0)
    # (b) share of games tied after regulation, by season
    K = 100
    xs = list(W.G_GATE)
    x_pool = max(xs) + 1.35
    a2.axvline(max(xs) + 0.68, color=C_SEP, lw=0.6)
    for s in xs + ["pooled"]:
        m = np.ones(len(g), bool) if s == "pooled" else (g.season_end == s).to_numpy()
        x = x_pool if s == "pooled" else s
        n, kk = m.sum(), tie[m].sum()
        wl_, wh_ = wilson(kk, n)
        errpoint(a2, x - 0.1, kk / n * K, wl_ * K, wh_ * K, C_OBS, ms=4.0)
        a2.plot(x + 0.1, g.p_tie[m].mean() * K, "D", color=C_G, ms=4.4, mec=C_G, zorder=5)
    for s in [s for s in range(min(xs), max(xs)) if s not in xs]:
        a2.text(s, 17.0, f"{s}\nnot scored", ha="center", va="center", fontsize=8,
                color="#666666", linespacing=1.0)
    a2.set_xticks(xs + [x_pool], [str(s) for s in xs] + ["Pooled"])
    a2.set_xlim(min(xs) - 0.6, x_pool + 0.6)
    a2.set_ylim(15, 28)
    a2.set_ylabel("Games tied after regulation (%)")
    a2.set_xlabel("Season (G_GATE)")
    a2.legend(handles=[Line2D([], [], color=C_OBS, marker="o", ms=4.0, ls="none",
                              label="Observed, 95% CI"),
                       Line2D([], [], color=C_G, marker="D", ms=4.4, ls="none",
                              label="Predicted, mean P(tie)")],
              loc="upper left", ncol=2)
    panel_tag(a1, "(a)", x=-0.2)
    panel_tag(a2, "(b)", x=-0.1)
    save(fig, "fig_outcome_calib")
    return slope, slope_se, g, tie


# =================================================================== 9 team box score
BOX = [("sogf", "y_sogf", "Team shots on goal", 2.0),
       ("xgf", "y_xgf_all", "Team expected goals", 0.2),
       ("goals", "y_gf_reg", "Team regulation goals", 0.2)]


def fig_team_box():
    print("fig_team_box")
    g = pd.read_csv(PREDS / "g_gate_games.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), layout="constrained")
    out = {}
    for ax, (pc, yc, name, step), tag in zip(axes, BOX, "abc"):
        p = np.r_[g[f"{pc}_h"], g[f"{pc}_a"]]
        y = np.r_[g[f"{yc}_h"], g[f"{yc}_a"]]
        ok = np.isfinite(y) & np.isfinite(p)
        b = equal_count_bins(p[ok], y[ok], 10)
        out[pc] = (int(ok.sum()), p[ok].mean(), y[ok].mean(), b)
        vmin = min(b.pred.min(), (b.obs - Z95 * b.se).min())
        vmax = max(b.pred.max(), (b.obs + Z95 * b.se).max())
        pad = 0.06 * (vmax - vmin)
        lo = np.floor((vmin - pad) / step) * step
        hi = np.ceil((vmax + pad) / step) * step
        ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=0.6, zorder=1)
        ax.errorbar(b.pred, b.obs, yerr=Z95 * b.se, fmt="none", ecolor=C_G, elinewidth=0.9,
                    zorder=3)
        ax.plot(b.pred, b.obs, "o", color=C_G, ms=3.8, mec="white", mew=0.5, zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        tick = 4 if pc == "sogf" else 0.5
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick))
        ax.text(0.04, 0.97, name, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xlabel("Predicted per team-game")
        panel_tag(ax, f"({tag})", x=-0.12, y=1.04)
        print(f"  {pc}: n = {ok.sum():,} team-games, mean predicted {p[ok].mean():.3f}, "
              f"observed {y[ok].mean():.3f}")
    axes[0].set_ylabel("Observed per team-game")
    save(fig, "fig_team_box")
    return out


# =================================================================== main
def main():
    style()
    W = load_windows()
    fig_windows(W)
    fig_c1_seasons(W)
    fig_reliability(W)
    fig_g_gate_seasons(W)
    fig_g_ladder(W)
    fig_player_heads()
    fig_xg_drift()
    fig_outcome_calib(W)
    fig_team_box()
    bad = [c for c in CHECKS if not c[4]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks reproduce.")
    for label, got, want, tol, _ in bad:
        print(f"  MISMATCH {label}: got {got:.6f}, expected {want:.6f}")


if __name__ == "__main__":
    main()
