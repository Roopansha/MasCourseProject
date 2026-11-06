"""Shared helpers for the experiment scripts (E1-E9).

Runs are stored as ``runs/<exp_name>/<run_name>/seed_<s>/`` and reused across experiments:
E1-E3 train the core models, E4-E7 analyse them. ``--quick`` switches to a tiny budget
(and a separate ``runs/quick_*`` namespace) to check that a script runs end to end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from analysis.stats import summarize_seeds  # noqa: E402

QUICK = ["algo.total_updates=6", "algo.num_envs=8", "algo.num_minibatches=2", "algo.update_epochs=2",
         "eval.episodes=16", "eval.batch_size=16", "logging.log_every=5"]


def parser(desc: str, default_seeds=(0, 1, 2, 3, 4)) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(default_seeds))
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke run (2 seeds, few updates)")
    ap.add_argument("--exp-name", default="core", help="namespace of the shared training runs")
    ap.add_argument("--set", nargs="*", default=[], help="extra Hydra overrides applied to every run")
    ap.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    return ap


def finalize_args(a):
    if a.quick:
        a.seeds = a.seeds[:2]
        a.exp_name = "quick_" + a.exp_name
    return a


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def base_overrides(a):
    return [f"exp_name={a.exp_name}"] + (QUICK if a.quick else []) + list(a.set)


def train_runs(conditions: dict, a, verbose: bool = True) -> dict:
    """``conditions[name] = list of Hydra overrides``. Trains every (condition, seed) that has
    no checkpoint yet and returns ``{name: [run_dir per seed]}``."""
    from core.utils import has_checkpoint
    from train import run_training

    out = {}
    for name, ov in conditions.items():
        dirs = []
        for s in a.seeds:
            cfg = compose_cfg(base_overrides(a) + [f"run_name={name}"] + list(ov) + [f"seed={s}"])
            cfg.out_dir = str(ROOT / "runs" / a.exp_name / name / f"seed_{s}")  # absolute, cwd-independent
            d = Path(cfg.out_dir)
            if a.retrain or not has_checkpoint(d):
                print(f"=== train {name} seed {s} -> {d}", flush=True)
                run_training(cfg, verbose=verbose)
            dirs.append(d)
        out[name] = dirs
    return out


def eval_runs(runs: dict, tag: str = "det", **kw) -> dict:
    """Evaluate (cached) every run. Returns ``{name: [summary per seed]}``."""
    from eval import evaluate_run

    return {name: [evaluate_run(d, tag=tag, **kw) for d in dirs] for name, dirs in runs.items()}


def per_seed(summaries: dict, key: str) -> dict:
    return {name: [s[key] for s in ss] for name, ss in summaries.items()}


def table(summaries: dict, keys) -> dict:
    return {name: {k: summarize_seeds([s[k] for s in ss]) for k in keys} for name, ss in summaries.items()}


def results_dir(a, exp: str) -> Path:
    d = ROOT / "results" / (("quick_" if a.quick else "") + exp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_np_default))


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_table_csv(path, tab: dict, keys):
    """Rows = conditions; columns = mean / CI / IQM of each key."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + [f"{k}_{s}" for k in keys for s in ("mean", "ci_lo", "ci_hi", "iqm", "n")])
        for name, row in tab.items():
            w.writerow([name] + [row[k][s] for k in keys for s in ("mean", "ci_lo", "ci_hi", "iqm", "n")])


def fmt_ci(s: dict) -> str:
    return f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] (IQM {s['iqm']:.3f}, n={s['n']})"


def event_analysis_run(run_dir, tag: str = "det", n_shifts: int = 1000, overwrite: bool = False, seed: int = 0) -> dict:
    """E5 statistics for one evaluated run (cached as ``events_<tag>.json``): observed phi /
    Cramer's V / lift / AUROC pooled and within distance bins, and their circular-shift nulls."""
    from analysis.events import event_stats
    from analysis.metrics import distance_bins
    from analysis.null import circular_shift_null, compare_to_null
    from eval import load_logs

    run_dir = Path(run_dir)
    cache = run_dir / f"events_{tag}.json"
    if cache.exists() and not overwrite:
        return json.loads(cache.read_text())
    logs, meta = load_logs(run_dir / f"eval_{tag}.npz")
    cfg = compose_cfg([]) if not (run_dir / "config.yaml").exists() else __import__("omegaconf").OmegaConf.load(run_dir / "config.yaml")
    edges = list(cfg.eval.distance_bins)
    nb = len(edges) - 1
    bins = distance_bins(logs["positions"], edges, float(cfg.env.obs_radius))
    obs = event_stats(logs["g"], logs["E"], logs["mask"], bins, nb, alpha=logs["alpha"])
    null = circular_shift_null(logs["g"], logs["E"], logs["mask"], bins, nb, n_shifts=n_shifts, seed=seed)
    out = dict(observed=obs, distance_bin_edges=edges, n_shifts=n_shifts)
    for k in ("phi", "lift", "auroc"):
        out[f"{k}_vs_null"] = compare_to_null(obs[k], null["null_pooled"][k])
        out[f"{k}_stratified_vs_null"] = compare_to_null(obs[f"{k}_stratified"], null["null_stratified"][k])
    out["null_phi_samples"] = np.asarray(null["null_pooled"]["phi"]).tolist()
    save_json(cache, out)
    return out


def lam_name(prefix, lc, ls):
    return f"{prefix}_lc{lc:g}_ls{ls:g}"


def gated_overrides(lc, ls, comm="gated"):
    return [f"comm={comm}", f"penalties.lambda_c={lc}", f"penalties.lambda_s={ls}"]
