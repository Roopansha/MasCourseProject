"""E2 Sparsity sweep (RQ1): can agents use fewer links without a substantial loss?

lambda_c swept over ~6 values with lambda_s = 0. Calibrate the values with ``--pilot`` (1 seed,
shorter training) so that link density spans ~100% down to 10-20%.

    python scripts/e2_sparsity_sweep.py --pilot --lambdas 0 0.01 0.03 0.1 0.3 1
    python scripts/e2_sparsity_sweep.py --lambdas 0 0.01 0.03 0.1 0.3 1 --seeds 0 1 2 3 4

Readout: the density below which reward drops (knee: lowest density whose mean return stays
within ``--tol`` of the lambda_c=0 model) and gate collapse (density < 2% at any lambda).
"""
from common import (eval_runs, finalize_args, gated_overrides, lam_name, parser, per_seed, results_dir,
                    save_json, table, train_runs, write_table_csv, fmt_ci)

import numpy as np

from analysis import plots

DEFAULT_LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]


def main():
    ap = parser(__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--pilot", action="store_true", help="1 seed, 1/3 of the training budget, separate namespace")
    ap.add_argument("--tol", type=float, default=0.05, help="relative return drop counted as 'substantial'")
    a = ap.parse_args()
    if a.pilot:
        a.seeds = a.seeds[:1]
        a.exp_name = "pilot_" + a.exp_name
        a.set = a.set + ["algo.total_updates=500"]
    a = finalize_args(a)
    out = results_dir(a, "e2_pilot" if a.pilot else "e2")

    conds = {lam_name("gated", lc, 0.0): gated_overrides(lc, 0.0) for lc in a.lambdas}
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag="det")
    summ_st = eval_runs(runs, tag="stoch", deterministic=False)
    keys = ["episode_return", "success_rate", "active_link_ratio", "gate_switch_rate", "mean_run_length"]
    tab = table(summ, keys)
    write_table_csv(out / "e2_table_det.csv", tab, keys)
    write_table_csv(out / "e2_table_stochastic_gates.csv", table(summ_st, keys), keys)

    dens, ret = per_seed(summ, "active_link_ratio"), per_seed(summ, "episode_return")
    names = list(conds)
    base = np.mean(ret[names[0]])
    ok = [n for n in names if np.mean(ret[n]) >= base - a.tol * abs(base)]
    knee = min((np.mean(dens[n]) for n in ok), default=float("nan"))
    collapse = {n: bool(np.mean(dens[n]) < 0.02) for n in names}
    report = dict(lambdas=a.lambdas, table=tab, knee_density=knee, within_tol_conditions=ok, gate_collapse=collapse,
                  density_range=[float(min(np.mean(v) for v in dens.values())), float(max(np.mean(v) for v in dens.values()))],
                  calibration_hint="aim for densities spanning ~1.0 down to 0.1-0.2; if collapse, raise warmup_frac, "
                                   "lower lambdas, raise comm.gate_init_logit or set comm.gate_entropy_coef > 0")
    save_json(out / "e2_report.json", report)

    pts = {"gated (deterministic gates)": [(dens[n], ret[n]) for n in names]}
    plots.xy_by_group(pts, "active link ratio", "episode return", out / "e2_return_vs_density.png",
                      "E2 return vs link density (one marker per lambda_c)", xlim=(0, 1.02))
    pts_st = {"gated (stochastic gates)": [(per_seed(summ_st, "active_link_ratio")[n], per_seed(summ_st, "episode_return")[n]) for n in names]}
    plots.xy_by_group(pts_st, "active link ratio", "episode return", out / "e2_return_vs_density_stochastic.png",
                      "E2 return vs link density, stochastic gates", xlim=(0, 1.02))
    for n in names:
        print(f"{n:22s} density {fmt_ci(tab[n]['active_link_ratio'])}  return {fmt_ci(tab[n]['episode_return'])}")
    print(f"knee density: {knee:.3f}; collapse: {collapse}\nresults in {out}")


if __name__ == "__main__":
    main()
