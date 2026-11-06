"""Figures for the v2 analysis. Every figure shows one point per seed plus the across-seed
mean and 95 % t-interval. Colours: lambda_s levels take the first three slots of the
reference categorical palette (validated all-pairs for colour-vision deficiency); baselines
and the random-mask control are neutral grey so they never compete with the learned series."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
NEUTRAL = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def _t(x):
    from scipy import stats as sps

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(x.mean()) if x.size else np.nan), np.nan
    return float(x.mean()), float(sps.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pick(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["seed"])


def _ev(r, k):
    return (r.get("events") or {}).get(k, np.nan)


def _mean_marker(ax, x, m, h, color, label=None, marker="o", xerr=None):
    ax.errorbar([x], [m], yerr=None if not np.isfinite(h) else [h], xerr=xerr, fmt=marker, color=color, ms=8,
                mec=SURFACE, mew=2, elinewidth=1.6, capsize=0, zorder=4, label=label)


def _baseline_band(ax, rows, kind, label):
    v = [r["episode_return"] for r in rows if r["kind"] == kind]
    if not v:
        return
    m, h = _t(v)
    ax.axhspan(m - (h if np.isfinite(h) else 0), m + (h if np.isfinite(h) else 0), color=NEUTRAL, alpha=0.12, lw=0)
    ax.axhline(m, color=NEUTRAL, lw=1.2, ls="--", zorder=1)
    ax.annotate(f"{label} {m:.2f}", xy=(1.0, m), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", va="center", ha="left", color=INK2, fontsize=9)


def fig_return_vs_density(rows, res, out, tag, key="episode_return", ylabel="Evaluation return (higher is better)"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if key == "episode_return":
        _baseline_band(ax, rows, "fullcomm", "FullComm")
        _baseline_band(ax, rows, "nocomm", "NoComm")
    else:
        for kind, lab in (("fullcomm", "FullComm"), ("nocomm", "NoComm")):
            v = [r[key] for r in rows if r["kind"] == kind]
            if v:
                m, _ = _t(v)
                ax.axhline(m, color=NEUTRAL, lw=1.2, ls="--")
                ax.annotate(f"{lab} {m:.2f}", xy=(1.0, m), xycoords=("axes fraction", "data"), xytext=(4, 0),
                            textcoords="offset points", va="center", color=INK2, fontsize=9)
    for i, ls in enumerate(res["lambda_s"]):
        for rho in res["rhos"]:
            rs = _pick(rows, kind="gated", rho=rho, ls=ls)
            if not rs:
                continue
            x = np.array([r["active_link_ratio"] for r in rs])
            y = np.array([r[key] for r in rs])
            ax.scatter(x, y, s=14, color=SERIES[i % 3], alpha=0.45, lw=0, zorder=3)
            mx, hx = _t(x)
            my, hy = _t(y)
            _mean_marker(ax, mx, my, hy, SERIES[i % 3], label=f"learned gates, λs = {ls:g}" if rho == res["rhos"][0] else None)
    for rho in res["rhos"]:
        rs = _pick(rows, kind="c1", rho=rho)
        if rs:
            x = np.array([r["active_link_ratio"] for r in rs])
            y = np.array([r[key] for r in rs])
            ax.scatter(x, y, s=14, color=NEUTRAL, alpha=0.5, marker="s", lw=0, zorder=2)
            _mean_marker(ax, _t(x)[0], *_t(y), NEUTRAL, label="random fixed mask (C1)" if rho == res["rhos"][0] else None,
                         marker="s")
    ax.set_xlabel("Achieved link density (fraction of links open)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.0)
    ax.set_title(f"RQ1: {'return' if key == 'episode_return' else 'success'} vs. link density "
                 f"(obs radius {res['radius']:g}, {tag} gates, dots = seeds)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=2, fontsize=9)
    _save(fig, out / f"v2_rq1_{'return' if key == 'episode_return' else 'success'}_vs_density_{tag}.png")


def fig_rq2(rows, res, out, tag):
    metrics = [("gate_switch_rate", "Gate switch rate\n(per link-step)"), ("mean_run_length", "Mean link run\nlength (steps)"),
               ("active_link_ratio", "Achieved density"), ("episode_return", "Evaluation return")]
    rhos, lss = res["rhos"], res["lambda_s"]
    fig, axes = plt.subplots(len(metrics), len(rhos), figsize=(3.0 * len(rhos) + 1, 2.3 * len(metrics)), sharex=True,
                             squeeze=False)
    for c, rho in enumerate(rhos):
        by_seed = {}
        for i, ls in enumerate(lss):
            for r in _pick(rows, kind="gated", rho=rho, ls=ls):
                by_seed.setdefault(r["seed"], {})[i] = r
        for rr, (k, lab) in enumerate(metrics):
            ax = axes[rr, c]
            for s, d in by_seed.items():  # paired seeds share initialisation and warm-up
                xs = sorted(d)
                ax.plot(xs, [d[i][k] for i in xs], color=GRID, lw=1.2, zorder=1)
            for i, ls in enumerate(lss):
                y = [d[i][k] for d in by_seed.values() if i in d]
                ax.scatter([i] * len(y), y, s=12, color=SERIES[i % 3], alpha=0.5, lw=0, zorder=2)
                _mean_marker(ax, i, *_t(y), SERIES[i % 3])
            if k == "active_link_ratio":
                ax.axhline(rho, color=MUTED, lw=1, ls=":")
            if c == 0:
                ax.set_ylabel(lab)
            if rr == 0:
                ax.set_title(f"target density {rho:g}", loc="left")
            ax.set_xticks(range(len(lss)), [f"{ls:g}" for ls in lss])
        axes[-1, c].set_xlabel("Switching penalty λs")
    fig.suptitle("RQ2: persistence at matched link density (grey lines join runs with the same seed)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, out / f"v2_rq2_persistence_{tag}.png")


def fig_rq3(rows, res, out, tag):
    rhos, lss = res["rhos"], res["lambda_s"]
    panels = [("log_or_pooled", "Pooled (no control)"), ("log_or_time", "Within time step"),
              ("log_or_time_dist", "Within time step × distance\n(primary)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.9), sharey=True)
    w = 0.22
    for ax, (k, title) in zip(axes, panels):
        ax.axhline(0, color=AXIS, lw=1.2)
        for i, ls in enumerate(lss):
            for j, rho in enumerate(rhos):
                y = [_ev(r, k) for r in _pick(rows, kind="gated", rho=rho, ls=ls)]
                x = j + (i - (len(lss) - 1) / 2) * w
                ax.scatter([x] * len(y), y, s=12, color=SERIES[i % 3], alpha=0.5, lw=0, zorder=2)
                _mean_marker(ax, x, *_t(y), SERIES[i % 3], label=f"λs = {ls:g}" if j == 0 else None)
        ax.set_xticks(range(len(rhos)), [f"{r:g}" for r in rhos])
        ax.set_xlabel("Target link density")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("log odds ratio, gate open vs. event\n(0 = no association)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower left", bbox_to_anchor=(0.01, -0.06), ncol=len(lab), fontsize=9)
    fig.suptitle("RQ3: association between open links and the event (Mantel–Haenszel, dots = seeds)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, out / f"v2_rq3_event_association_{tag}.png")


def fig_time_profile(rows, res, out, tag):
    rhos, lss = res["rhos"], res["lambda_s"]
    fig, axes = plt.subplots(1, len(rhos), figsize=(3.4 * len(rhos), 3.4), sharey=True, squeeze=False)
    for c, rho in enumerate(rhos):
        ax = axes[0, c]
        for i, ls in enumerate(lss):
            rs = _pick(rows, kind="gated", rho=rho, ls=ls)
            if not rs:
                continue
            prof = np.array([r["density_by_t"] for r in rs])
            m = prof.mean(0)
            ax.plot(np.arange(m.size), m, color=SERIES[i % 3], label=f"λs = {ls:g}")
            ax.fill_between(np.arange(m.size), prof.min(0), prof.max(0), color=SERIES[i % 3], alpha=0.10, lw=0)
        ev = [r["event_by_t"] for r in rows if r["kind"] == "gated" and r["rho"] == rho]
        if ev:
            ax.plot(np.mean(ev, 0), color=MUTED, lw=1.4, ls="--", label="event rate")
        ax.set_title(f"target density {rho:g}", loc="left")
        ax.set_xlabel("Time step in episode")
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("Fraction of links open")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("When are links open? (line = mean over seeds, band = seed range)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, out / f"v2_time_profile_{tag}.png")


def make_all(rows, res, out: Path, tag: str):
    fig_return_vs_density(rows, res, out, tag)
    fig_return_vs_density(rows, res, out, tag, key="success_rate", ylabel="Success rate")
    fig_rq2(rows, res, out, tag)
    fig_rq3(rows, res, out, tag)
    fig_time_profile(rows, res, out, tag)


# ----------------------------------------------------------------------------- demonstration
def _episode(run_dir, tag, ep):
    from eval import load_logs
    from omegaconf import OmegaConf

    logs, _ = load_logs(Path(run_dir) / f"eval_{tag}.npz")
    out = {k: logs[k][ep] for k in ("positions", "landmark_pos", "g", "mask", "E", "covered", "reward")}
    out["zero_messages"] = bool(OmegaConf.load(Path(run_dir) / "config.yaml").comm.zero_messages)
    return out


def link_raster(models: dict, out_path, ep: int = 0, tag: str = "stoch"):
    """Gate state of every directed link over one episode (same layout for every model)."""
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 1.5 * len(models) + 0.6), sharex=True, squeeze=False)
    for ax, (name, d) in zip(axes[:, 0], models.items()):
        e = _episode(d, tag, ep)
        N = e["g"].shape[-1]
        links = [(i, j) for i in range(N) for j in range(N) if i != j]
        on = np.array([[(e["g"][t, i, j] > 0) and e["mask"][t, i, j] for t in range(e["g"].shape[0])] for i, j in links])
        ev = np.array([[e["E"][t, i, j] for t in range(e["g"].shape[0])] for i, j in links])
        ax.imshow(np.where(on, 1.0, np.nan), aspect="auto", cmap=matplotlib.colors.ListedColormap([SERIES[0]]),
                  interpolation="nearest", vmin=0, vmax=1)
        yy, xx = np.nonzero(ev)
        ax.scatter(xx, yy, s=5, color=SERIES[1], marker="|", lw=1.2, label="event (sender sees a landmark the receiver cannot)")
        ax.set_yticks(range(len(links)), [f"{i}→{j}" for i, j in links], fontsize=6)
        ax.grid(False)
        ax.set_title(f"{name}: return {e['reward'].sum():.2f}", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("Time step")
    h = [matplotlib.patches.Patch(color=SERIES[0], label="link open"),
         matplotlib.lines.Line2D([], [], color=SERIES[1], marker="|", ls="", label="event on this link")]
    axes[0, 0].legend(handles=h, loc="upper right", fontsize=8, ncol=2, bbox_to_anchor=(1.0, 1.45))
    fig.tight_layout()
    _save(fig, out_path)


def animate(models: dict, out_path, ep: int = 0, tag: str = "stoch", fps: int = 6):
    """Side-by-side animation of the same episode layout under each communication setting:
    agents (dots), landmarks (x, filled when covered), open links (arrows sender -> receiver)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    eps = {k: _episode(d, tag, ep) for k, d in models.items()}
    T = next(iter(eps.values()))["positions"].shape[0]
    allp = np.concatenate([np.concatenate([e["positions"].reshape(-1, 2), e["landmark_pos"].reshape(-1, 2)]) for e in eps.values()])
    lim = np.abs(allp).max() * 1.08
    fig, axes = plt.subplots(1, len(eps), figsize=(3.3 * len(eps), 3.6), squeeze=False)
    axes = axes[0]

    def draw(t):
        for ax, (name, e) in zip(axes, eps.items()):
            ax.clear()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():  # arena boundary
                sp.set_visible(True)
                sp.set_color(AXIS)
            p, lp = e["positions"][t], e["landmark_pos"][t]
            cov = e["covered"][t]
            N = p.shape[0]
            n_open = 0
            for i in range(N):
                for j in range(N):
                    if not e["zero_messages"] and i != j and e["mask"][t, i, j] and e["g"][t, i, j] > 0:
                        n_open += 1
                        ax.annotate("", xy=p[j], xytext=p[i], arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=1.4,
                                                                             alpha=0.75, shrinkA=6, shrinkB=6))
            ax.scatter(lp[:, 0], lp[:, 1], marker="X", s=70, color=np.where(cov, SERIES[2], AXIS), zorder=3)
            ax.scatter(p[:, 0], p[:, 1], s=60, color=INK, edgecolor=SURFACE, lw=1.5, zorder=4)
            links = "messages zeroed" if e["zero_messages"] else f"open links {n_open}/{N * (N - 1)}"
            ax.set_title(f"{name}\nt = {t:2d}   {links}", fontsize=9, loc="left")
        return []

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
