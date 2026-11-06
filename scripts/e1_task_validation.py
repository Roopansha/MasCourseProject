"""E1 Task validation: does the task need communication and leave headroom?

Conditions: NoComm (messages * 0, same graph/params), FullComm (all neighbour gates on),
HandRule (gate on when distance < d, d swept).
Pass criteria: FullComm > NoComm, FullComm not at ceiling, best HandRule > NoComm, events in
roughly 10-30% of link-timesteps with variation across distance bins.

    python scripts/e1_task_validation.py --seeds 0 1 2 3 4
    python scripts/e1_task_validation.py --quick
"""
from common import (eval_runs, finalize_args, parser, per_seed, results_dir, save_json, table, train_runs,
                    write_table_csv, fmt_ci)

import numpy as np

from analysis import plots
from analysis.metrics import distance_bins, per_bin_rate
from analysis.stats import bootstrap_ci
from eval import load_logs


def main():
    ap = parser(__doc__)
    ap.add_argument("--hand-d", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="hand-rule distances (world units)")
    ap.add_argument("--ceiling", type=float, default=0.95, help="success rate counted as ceiling")
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e1")

    conds = {"nocomm": ["comm=nocomm"], "fullcomm": ["comm=fullcomm"]}
    for d in a.hand_d:
        conds[f"handrule_d{d:g}"] = ["comm=handrule", f"comm.hand_rule_d={d}"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag="det")
    keys = ["episode_return", "success_rate", "final_coverage", "active_link_ratio", "event_rate"]
    tab = table(summ, keys)
    write_table_csv(out / "e1_table.csv", tab, keys)

    # event frequency per distance bin (FullComm logs: every permitted link is active)
    edges = None
    rates = []
    for d in runs["fullcomm"]:
        logs, meta = load_logs(d / "eval_det.npz")
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(d / "config.yaml")
        edges = list(cfg.eval.distance_bins)
        bins = distance_bins(logs["positions"], edges, cfg.env.obs_radius)
        rates.append(per_bin_rate(logs["E"], logs["mask"], bins, len(edges) - 1))
    rates = np.array(rates)
    bin_labels = [f"{edges[i]:g}-{edges[i + 1]:g}" if edges[i + 1] < 1e6 else f">{edges[i]:g}" for i in range(len(edges) - 1)]

    ret = per_seed(summ, "episode_return")
    succ = per_seed(summ, "success_rate")
    hand = [k for k in ret if k.startswith("handrule")]
    best_hand = max(hand, key=lambda k: np.mean(ret[k]))
    diff = lambda x, y: bootstrap_ci(np.array(x) - np.array(y))
    full_vs_no = diff(ret["fullcomm"], ret["nocomm"])
    hand_vs_no = diff(ret[best_hand], ret["nocomm"])
    ev = np.mean(per_seed(summ, "event_rate")["fullcomm"])
    checks = {
        "fullcomm_beats_nocomm": bool(full_vs_no[1] > 0) if np.isfinite(full_vs_no[1]) else bool(full_vs_no[0] > 0),
        "fullcomm_not_at_ceiling": bool(np.mean(succ["fullcomm"]) < a.ceiling),
        "handrule_beats_nocomm": bool(hand_vs_no[1] > 0) if np.isfinite(hand_vs_no[1]) else bool(hand_vs_no[0] > 0),
        "event_rate_in_10_30pct": bool(0.10 <= ev <= 0.30),
        "event_rate_varies_across_bins": bool(np.nanmax(np.nanmean(rates, 0)) - np.nanmin(np.nanmean(rates, 0)) > 0.05),
    }
    report = dict(table=tab, best_handrule=best_hand, fullcomm_minus_nocomm_return=full_vs_no,
                  besthand_minus_nocomm_return=hand_vs_no, event_rate_fullcomm=ev,
                  event_rate_per_distance_bin=dict(zip(bin_labels, np.nanmean(rates, 0).tolist())),
                  checks=checks, all_pass=all(checks.values()),
                  if_fail="shrink env.obs_radius, hide landmarks more often, or add more landmarks than a local view covers")
    save_json(out / "e1_report.json", report)

    plots.points_with_ci(ret, "episode return (deterministic gates)", out / "e1_return.png", "E1 return per condition")
    plots.points_with_ci(succ, "success rate", out / "e1_success.png", "E1 success per condition")
    plots.lines_by_group(np.arange(len(bin_labels)), {"event rate (FullComm)": rates}, "link distance / R_obs",
                         "P(E=1 | permitted link)", out / "e1_event_rate_by_distance.png",
                         "E1 event frequency per distance bin", xticklabels=bin_labels)
    for name, row in tab.items():
        print(f"{name:18s} return {fmt_ci(row['episode_return'])}  success {fmt_ci(row['success_rate'])}")
    print("checks:", checks)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
