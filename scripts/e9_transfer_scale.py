"""E9 Transfer and scale: are the findings specific to one N, one gate estimator, or one task?

Runs:
  * N in {4, 10, 50} (K=10 neighbours only bites at N >= 12, so scaling claims come from N=50)
  * zero-shot transfer N=50 -> N=100 with frozen weights
  * gate estimator swap: straight-through and REINFORCE gates at key (lambda_c, lambda_s)
  * second observation radius (env=spread_small_robs)
Metrics: throughput (env-steps/s), peak memory, and the E3/E5 headline numbers (density,
switch rate, run length, phi / lift vs circular-shift null) for every gated model.
Fallback if memory runs out: lower --large-num-envs and raise --grad-accum.

    python scripts/e9_transfer_scale.py --seeds 0 1 2 --lambda-c 0.1 --lambda-s 0.1
"""
from common import (eval_runs, event_analysis_run, finalize_args, gated_overrides, lam_name, parser, results_dir,
                    save_json, train_runs)

import json

import numpy as np

from analysis import plots
from analysis.stats import summarize_seeds
from eval import evaluate_run


def main():
    ap = parser(__doc__, default_seeds=(0, 1, 2))
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 10, 50])
    ap.add_argument("--transfer-from", type=int, default=50)
    ap.add_argument("--transfer-to", type=int, default=100)
    ap.add_argument("--transfer-episodes", type=int, default=20)
    ap.add_argument("--lambda-c", type=float, default=0.1)
    ap.add_argument("--lambda-s", type=float, default=0.1)
    ap.add_argument("--estimator-n", type=int, default=4)
    ap.add_argument("--large-n", type=int, default=20, help="N at/above which the large-N settings apply")
    ap.add_argument("--large-num-envs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--large-eval-episodes", type=int, default=30)
    ap.add_argument("--n-shifts", type=int, default=1000)
    a = finalize_args(ap.parse_args())
    if a.quick:
        a.ns = [n for n in a.ns if n <= 10] + [12]
        a.transfer_from, a.transfer_to, a.transfer_episodes, a.n_shifts = 12, 16, 4, 50
    out = results_dir(a, "e9")
    lc, ls = a.lambda_c, a.lambda_s

    def size_ov(n):
        ov = [f"env.n_agents={n}"]
        if n >= a.large_n and not a.quick:
            ov += [f"algo.num_envs={a.large_num_envs}", f"algo.grad_accum_steps={a.grad_accum}",
                   f"eval.episodes={a.large_eval_episodes}", f"eval.batch_size={min(a.large_eval_episodes, 10)}"]
        return ov

    conds = {}
    for n in a.ns:
        conds[f"fullcomm_N{n}"] = ["comm=fullcomm"] + size_ov(n)
        conds[f"nocomm_N{n}"] = ["comm=nocomm"] + size_ov(n)
        conds[f"{lam_name('gated', lc, ls)}_N{n}"] = gated_overrides(lc, ls) + size_ov(n)
    for est in ("gated_ste", "gated_reinforce"):
        for (c, s) in ((lc, 0.0), (lc, ls)):
            conds[f"{lam_name(est, c, s)}_N{a.estimator_n}"] = gated_overrides(c, s, comm=est) + size_ov(a.estimator_n)
    for base, ov in (("fullcomm", ["comm=fullcomm"]), ("nocomm", ["comm=nocomm"]), (lam_name("gated", lc, ls), gated_overrides(lc, ls))):
        conds[f"{base}_smallRobs"] = ov + ["env=spread_small_robs"]

    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag="det")

    report = {}
    for name, dirs in runs.items():
        metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
        r = {k: summarize_seeds([s[k] for s in summ[name]]) for k in
             ("episode_return", "success_rate", "active_link_ratio", "gate_switch_rate", "mask_switch_rate", "mean_run_length")}
        r["fps_env_steps"] = summarize_seeds([m["fps_env_steps"] for m in metas])
        r["peak_memory_mb"] = summarize_seeds([m["peak_memory_mb"] for m in metas])
        if "gated" in name:
            ev = [event_analysis_run(d, tag="det", n_shifts=a.n_shifts, seed=k) for k, d in enumerate(dirs)]
            r["phi"] = summarize_seeds([e["observed"]["phi"] for e in ev])
            r["phi_excess_over_null"] = summarize_seeds([e["phi_vs_null"]["excess"] for e in ev])
            r["lift"] = summarize_seeds([e["observed"]["lift"] for e in ev])
            r["phi_p_value"] = [e["phi_vs_null"]["p_value"] for e in ev]
        report[name] = r

    transfer = {}
    for base in ("fullcomm", "nocomm", lam_name("gated", lc, ls)):
        src = f"{base}_N{a.transfer_from}"
        if src not in runs:
            continue
        ss = [evaluate_run(d, tag=f"transfer_N{a.transfer_to}", save=False, light=True, n_episodes=a.transfer_episodes,
                           batch_size=min(a.transfer_episodes, 10), env_overrides={"n_agents": a.transfer_to})
              for d in runs[src]]
        transfer[base] = dict(
            in_distribution_return=report[src]["episode_return"],
            zero_shot_return=summarize_seeds([s["episode_return"] for s in ss]),
            zero_shot_success=summarize_seeds([s["success_rate"] for s in ss]),
            zero_shot_density=summarize_seeds([s["active_link_ratio"] for s in ss]))
    save_json(out / "e9_report.json", dict(per_condition=report, zero_shot_transfer=transfer,
                                           note="reward is normalised per landmark, so returns are comparable across N. peak_memory_mb is the "
                                                "process-lifetime peak: run one N per process (e.g. --ns 50) for clean memory numbers"))

    ys = {}
    for base in ("nocomm", "fullcomm", lam_name("gated", lc, ls)):
        ys[base] = np.array([[summ[f"{base}_N{n}"][k]["episode_return"] for n in a.ns] for k in range(len(a.seeds))])
    plots.lines_by_group(np.arange(len(a.ns)), ys, "number of agents N", "episode return (per-landmark reward)",
                         out / "e9_return_vs_N.png", "E9 return vs N", xticklabels=[str(n) for n in a.ns])
    fps = {base: np.array([[json.loads((runs[f"{base}_N{n}"][k] / "meta.json").read_text())["fps_env_steps"]
                            for n in a.ns] for k in range(len(a.seeds))]) for base in ys}
    plots.lines_by_group(np.arange(len(a.ns)), fps, "number of agents N", "training throughput (env-steps / s)",
                         out / "e9_fps_vs_N.png", "E9 throughput vs N", xticklabels=[str(n) for n in a.ns])
    for n, r in report.items():
        extra = f"  phi {r['phi']['mean']:+.3f}" if "phi" in r else ""
        print(f"{n:34s} return {r['episode_return']['mean']:8.3f}  density {r['active_link_ratio']['mean']:.2f}  "
              f"switch {r['gate_switch_rate']['mean']:.3f}  fps {r['fps_env_steps']['mean']:.0f}  "
              f"mem {r['peak_memory_mb']['mean']:.0f}MB{extra}")
    for b, t in transfer.items():
        print(f"transfer {b}: N={a.transfer_from} {t['in_distribution_return']['mean']:.3f} -> "
              f"N={a.transfer_to} {t['zero_shot_return']['mean']:.3f}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
