"""v2 experiment set: the three research questions of the project proposal, multi-seed.

Differences from the v1 scripts (E1-E9):
  * every condition is trained with several seeds, vmapped into one GPU program per batch;
  * learned gates are trained to a *target link density* (lambda_c adapted by dual ascent),
    so models with and without the switching penalty can be compared at the same link count;
  * 3000 updates per run (v1: 1500, not converged).

Phases (each skips runs that already have a checkpoint):
  baselines  NoComm and FullComm at obs_radius 0.3 and 0.5            -> runs/v2
  pilot      one seed per setting: dual step size, lambda_s range      -> runs/v2_pilot
  pilot2     PI controller + lambda_s switched on after sparsification -> runs/v2_pilot2
  main       target density x lambda_s grid at the chosen radius       -> runs/v2
  controls   random fixed mask at each target density (from scratch)   -> runs/v2

    python scripts/v2_runs.py baselines --radius 0.3
    python scripts/v2_runs.py pilot --radius 0.3
    python scripts/v2_runs.py pilot2 --radius 0.3
    python scripts/v2_runs.py main --radius 0.3 --rho 0.5 0.25 0.1 --ls 0 0.01 0.05
    python scripts/v2_runs.py controls --radius 0.3 --rho 0.5 0.25 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import compose_cfg  # noqa: E402

V2 = ["algo.total_updates=3000", "eval.episodes=200", "eval.batch_size=200", "logging.log_every=20"]
QUICK = ["algo.total_updates=6", "algo.num_envs=8", "algo.num_minibatches=2", "algo.update_epochs=1",
         "eval.episodes=8", "eval.batch_size=8", "logging.log_every=2"]


def gated_name(rho, ls, radius, extra=""):
    return f"gated_rho{rho:g}_ls{ls:g}{extra}_R{radius:g}"


def make_cfg(exp, name, seed, overrides, a):
    base = QUICK if a.quick else V2
    cfg = compose_cfg([f"exp_name={exp}", f"run_name={name}", f"seed={seed}", f"env.obs_radius={a.radius}",
                       *base, *overrides])
    cfg.out_dir = str(ROOT / "runs" / exp / name / f"seed_{seed}")
    return cfg


def conditions(a):
    """``[(exp_name, run_name, overrides, batch_key)]``; runs with the same batch key share a
    jitted program (same static settings)."""
    R = a.radius
    out = []
    if a.phase == "baselines":
        out += [("v2", f"nocomm_R{R:g}", ["comm=nocomm"], "nocomm"),
                ("v2", f"fullcomm_R{R:g}", ["comm=fullcomm"], "fullcomm")]
    elif a.phase == "pilot":
        grid = [(0.25, 0.0, lr) for lr in (1e-4, 3e-4, 1e-3)]
        grid += [(0.25, ls, 3e-4) for ls in (0.003, 0.01, 0.03, 0.1)]
        grid += [(0.1, 0.0, 3e-4), (0.5, 0.0, 3e-4)]
        for rho, ls, lr in grid:
            ov = ["comm=gated", f"penalties.target_density={rho}", f"penalties.lambda_s={ls}", f"penalties.dual_lr={lr}"]
            out.append(("v2_pilot", gated_name(rho, ls, R, f"_lr{lr:g}"), ov, "gated"))
    elif a.phase == "pilot2":
        grid = [(0.25, ls, 0.05) for ls in (0.0, 0.003, 0.01, 0.03, 0.1)]
        grid += [(0.25, 0.01, 0.0), (0.1, 0.0, 0.05), (0.1, 0.03, 0.05), (0.5, 0.03, 0.05)]
        for rho, ls, kp in grid:
            ov = ["comm=gated", f"penalties.target_density={rho}", f"penalties.lambda_s={ls}",
                  f"penalties.dual_lr={a.dual_lr}", f"penalties.dual_kp={kp}",
                  f"penalties.ls_warmup_frac={a.ls_warmup}", f"penalties.ls_anneal_frac={a.ls_anneal}"]
            out.append(("v2_pilot2", gated_name(rho, ls, R, f"_kp{kp:g}"), ov, "gated"))
    elif a.phase == "main":
        for rho in a.rho:
            for ls in a.ls:
                ov = ["comm=gated", f"penalties.target_density={rho}", f"penalties.lambda_s={ls}",
                      f"penalties.dual_lr={a.dual_lr}", f"penalties.dual_kp={a.dual_kp}",
                      f"penalties.ls_warmup_frac={a.ls_warmup}", f"penalties.ls_anneal_frac={a.ls_anneal}"]
                out.append(("v2", gated_name(rho, ls, R), ov, f"gated_rho{rho:g}"))
    elif a.phase == "controls":
        for rho in a.rho:
            out.append(("v2", f"c1fixed_rho{rho:g}_R{R:g}", ["comm=random_fixed", f"comm.density={rho}"], f"c1_{rho:g}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["baselines", "pilot", "pilot2", "main", "controls"])
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    ap.add_argument("--ls", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    ap.add_argument("--dual-lr", type=float, default=1e-3)
    ap.add_argument("--dual-kp", type=float, default=0.05)
    ap.add_argument("--ls-warmup", type=float, default=0.5, help="lambda_s starts after the density target is reached")
    ap.add_argument("--ls-anneal", type=float, default=0.2)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these batch keys")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run in runs/quick_v2*")
    a = ap.parse_args()
    if a.phase in ("pilot", "pilot2"):
        a.seeds = a.seeds[:1]
    if a.quick:
        a.seeds = a.seeds[:2]

    from core.utils import has_checkpoint
    from train import run_training_batch

    batches: dict[str, list] = {}
    for exp, name, ov, bkey in conditions(a):
        exp = ("quick_" + exp) if a.quick else exp
        for s in a.seeds:
            cfg = make_cfg(exp, name, s, ov, a)
            if has_checkpoint(cfg.out_dir):
                print(f"skip {name} seed {s} (checkpoint exists)", flush=True)
                continue
            batches.setdefault(bkey, []).append(cfg)
    for bkey, cfgs in batches.items():
        if a.only and bkey not in a.only:
            continue
        print(f"=== batch {bkey}: {len(cfgs)} runs: " + ", ".join(f"{c.run_name}/s{c.seed}" for c in cfgs), flush=True)
        run_training_batch(cfgs)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
