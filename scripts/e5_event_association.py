"""E5 Event association (RQ3): are active links associated with the task event more than chance?

Event E[t,i,j] = sender i sees an uncovered landmark outside receiver j's observation radius
(permitted links only). Statistics per seed: signed phi, Cramer's V, lift, AUROC (binary g and
rank-AUROC of alpha), pooled and within distance bins. Null: >= 1000 circular time-shifts per
(episode, link). Aggregation: one statistic per seed, bootstrap CI across seeds.
Report association, not causation.

    python scripts/e5_event_association.py --lc 0 0.03 0.3 --ls 0 0.03 0.3 --seeds 0 1 2 3 4
"""
from common import (eval_runs, event_analysis_run, finalize_args, gated_overrides, lam_name, parser, per_seed,
                    results_dir, save_json, train_runs)

import numpy as np

from analysis import plots
from analysis.stats import summarize_seeds

STATS = ["phi", "cramers_v", "lift", "auroc", "auroc_alpha", "phi_stratified", "lift_stratified"]


def main():
    ap = parser(__doc__)
    ap.add_argument("--lc", type=float, nargs="+", default=[0.0, 0.03, 0.3])
    ap.add_argument("--ls", type=float, nargs="+", default=[0.0, 0.03, 0.3])
    ap.add_argument("--n-shifts", type=int, default=1000)
    ap.add_argument("--bin-width", type=float, default=0.1, help="density bin for the matched-density comparison")
    a = finalize_args(ap.parse_args())
    if a.quick:
        a.n_shifts = min(a.n_shifts, 100)
    out = results_dir(a, "e5")

    conds = {lam_name("gated", lc, ls): gated_overrides(lc, ls) for lc in a.lc for ls in a.ls}
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag="det")
    eval_runs(runs, tag="stoch", deterministic=False)

    report, scatter = {}, []
    for name, dirs in runs.items():
        rep = {}
        for tag in ("det", "stoch"):
            per = [event_analysis_run(d, tag=tag, n_shifts=a.n_shifts, seed=k) for k, d in enumerate(dirs)]
            agg = {s: summarize_seeds([p["observed"].get(s, np.nan) for p in per]) for s in STATS}
            agg["phi_excess_over_null"] = summarize_seeds([p["phi_vs_null"]["excess"] for p in per])
            agg["phi_stratified_excess_over_null"] = summarize_seeds([p["phi_stratified_vs_null"]["excess"] for p in per])
            agg["lift_excess_over_null"] = summarize_seeds([p["lift_vs_null"]["excess"] for p in per])
            agg["phi_null_percentile"] = [p["phi_vs_null"]["percentile"] for p in per]
            agg["phi_p_value"] = [p["phi_vs_null"]["p_value"] for p in per]
            rep[tag] = agg
            if tag == "det":
                plots.null_histogram(per[0]["null_phi_samples"], per[0]["observed"]["phi"], "phi (pooled)",
                                     out / f"e5_null_{name}_seed{a.seeds[0]}.png", f"E5 null vs observed: {name}")
                for p, s in zip(per, summ[name]):
                    scatter.append(dict(name=name, density=s["active_link_ratio"], switch=s["gate_switch_rate"],
                                        lift=p["observed"]["lift"], phi=p["observed"]["phi"],
                                        lift_excess=p["lift_vs_null"]["excess"]))
        report[name] = rep

    # lift vs switch rate at matched density: compare lambda_s settings within density bins
    matched = []
    edges = np.arange(0, 1 + a.bin_width, a.bin_width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in scatter if lo <= r["density"] < hi or (hi >= 1 and r["density"] == 1)]
        groups = {}
        for r in sel:
            groups.setdefault(r["name"], []).append(r)
        if len(groups) > 1:
            matched.append(dict(density_bin=[float(lo), float(hi)],
                                models={k: dict(lift=float(np.nanmean([r["lift"] for r in v])),
                                                switch=float(np.mean([r["switch"] for r in v])), n=len(v))
                                        for k, v in groups.items()}))
    save_json(out / "e5_report.json", dict(per_model=report, per_run=scatter, matched_density=matched, n_shifts=a.n_shifts,
                                           note="association, not causation; one statistic per seed, bootstrap across seeds"))
    good = [r for r in scatter if np.isfinite(r["lift"])]
    if good:
        plots.scatter_colored([r["switch"] for r in good], [r["lift"] for r in good], [r["density"] for r in good],
                              "gate switch rate", "lift  P(g|E=1) / P(g|E=0)", "active link ratio",
                              out / "e5_lift_vs_switch.png", "E5 lift vs switch rate (color = density)")
    plots.points_with_ci({n: [r["phi"] for r in scatter if r["name"] == n] for n in runs}, "phi (deterministic gates)",
                         out / "e5_phi_per_model.png", "E5 signed phi per model")
    for n, r in report.items():
        d = r["det"]
        print(f"{n:22s} phi {d['phi']['mean']:+.3f} [{d['phi']['ci_lo']:+.3f},{d['phi']['ci_hi']:+.3f}]  "
              f"excess {d['phi_excess_over_null']['mean']:+.3f}  lift {d['lift']['mean']:.2f}  p {np.round(d['phi_p_value'], 3)}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
