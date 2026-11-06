"""E3 Persistence at matched density (RQ2).

3x3 grid lambda_c in {0, small, large} x lambda_s in {0, small, large}. Models are binned by
link density; within a bin, return and switch rate are compared between lambda_s settings
(matched-density rule: all-off and all-on gates both have zero switching cost).

    python scripts/e3_persistence_grid.py --lc 0 0.03 0.3 --ls 0 0.03 0.3 --seeds 0 1 2 3 4
"""
from common import (eval_runs, finalize_args, gated_overrides, lam_name, parser, results_dir, save_json, table,
                    train_runs, write_table_csv)

import numpy as np

from analysis import plots

KEYS = ["episode_return", "success_rate", "active_link_ratio", "gate_switch_rate", "mask_switch_rate",
        "mean_run_length", "event_rate"]


def matched_density_comparison(rows, bin_width):
    """rows: list of dicts (one per seed-run) with lc, ls, density, return, switch.
    For every density bin, compare lambda_s = 0 against lambda_s > 0 runs."""
    out = []
    edges = np.arange(0, 1 + bin_width, bin_width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["density"] < hi or (hi >= 1 and r["density"] == 1)]
        a = [r for r in sel if r["ls"] == 0]
        b = [r for r in sel if r["ls"] > 0]
        if not a or not b:
            continue
        ra, rb = np.array([r["return"] for r in a]), np.array([r["return"] for r in b])
        sa, sb = np.array([r["switch"] for r in a]), np.array([r["switch"] for r in b])
        # bootstrap the difference in means (unpaired)
        rng = np.random.default_rng(0)
        diffs = [rng.choice(rb, rb.size).mean() - rng.choice(ra, ra.size).mean() for _ in range(5000)]
        out.append(dict(density_bin=[float(lo), float(hi)], n_ls0=len(a), n_ls_pos=len(b),
                        return_ls0=float(ra.mean()), return_ls_pos=float(rb.mean()),
                        return_diff=float(rb.mean() - ra.mean()), return_diff_ci=np.percentile(diffs, [2.5, 97.5]).tolist(),
                        switch_ls0=float(sa.mean()), switch_ls_pos=float(sb.mean())))
    return out


def main():
    ap = parser(__doc__)
    ap.add_argument("--lc", type=float, nargs=3, default=[0.0, 0.03, 0.3], help="lambda_c: 0 small large")
    ap.add_argument("--ls", type=float, nargs=3, default=[0.0, 0.03, 0.3], help="lambda_s: 0 small large")
    ap.add_argument("--bin-width", type=float, default=0.1)
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e3")

    conds = {lam_name("gated", lc, ls): gated_overrides(lc, ls) for lc in a.lc for ls in a.ls}
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag="det")
    eval_runs(runs, tag="stoch", deterministic=False)  # stochastic-gate stats, reported separately (E5)
    tab = table(summ, KEYS)
    write_table_csv(out / "e3_table.csv", tab, KEYS)

    mats = {k: np.full((3, 3), np.nan) for k in ("return", "density", "gate switch rate")}
    rows = []
    for r, lc in enumerate(a.lc):
        for c, ls in enumerate(a.ls):
            n = lam_name("gated", lc, ls)
            mats["return"][r, c] = tab[n]["episode_return"]["mean"]
            mats["density"][r, c] = tab[n]["active_link_ratio"]["mean"]
            mats["gate switch rate"][r, c] = tab[n]["gate_switch_rate"]["mean"]
            for s in summ[n]:
                rows.append(dict(name=n, lc=lc, ls=ls, density=s["active_link_ratio"],
                                 switch=s["gate_switch_rate"], run_length=s["mean_run_length"],
                                 mask_switch=s["mask_switch_rate"], **{"return": s["episode_return"]}))
    cmp_ = matched_density_comparison(rows, a.bin_width)
    save_json(out / "e3_report.json", dict(grid_lc=a.lc, grid_ls=a.ls, table=tab, per_run=rows, matched_density=cmp_,
                                           readout="return differing between lambda_s settings within a density bin => "
                                                   "persistence matters independently; only switch rate differing => "
                                                   "affects stability, not performance"))
    labels = lambda v: [f"{x:g}" for x in v]
    plots.heatmaps(mats, labels(a.lc), labels(a.ls), "lambda_c", "lambda_s", out / "e3_heatmaps.png", fmt="{:.2f}")
    plots.scatter_colored([r["switch"] for r in rows], [r["return"] for r in rows], [r["density"] for r in rows],
                          "gate switch rate", "episode return", "active link ratio", out / "e3_return_vs_switch.png",
                          "E3 return vs switch rate (color = density, one point per seed)")
    for c in cmp_:
        print(f"density {c['density_bin']}: return ls0 {c['return_ls0']:.3f} vs ls>0 {c['return_ls_pos']:.3f} "
              f"(diff CI {np.round(c['return_diff_ci'], 3)}), switch {c['switch_ls0']:.3f} vs {c['switch_ls_pos']:.3f}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
