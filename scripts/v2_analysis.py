"""v2 analysis (see results/v2/ANALYSIS_PLAN.md): evaluation, per-run statistics, RQ1-RQ3.

    python scripts/v2_analysis.py eval  --shard 0 --n-shards 6     # evaluate runs/v2 (stoch + det)
    python scripts/v2_analysis.py stats --shard 0 --n-shards 6     # per-run statistics (cached)
    python scripts/v2_analysis.py report                           # tests, tables, figures -> results/v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import save_json  # noqa: E402

OUT = ROOT / "results" / "v2"
MARGIN = 0.25  # pre-specified: a loss above 25 % of the communication benefit is "substantial"
DIST_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 1.0e9]  # RQ3 distance strata, units of obs_radius (amendment 2)
TAGS = ("stoch", "det")


# ----------------------------------------------------------------------------- discovery
def discover(exp: str = "v2"):
    """``[(run_dir, info)]`` for every trained run under ``runs/<exp>``."""
    out = []
    for d in sorted((ROOT / "runs" / exp).glob("*/seed_*")):
        if not (d / "params.msgpack").exists():
            continue
        cfg = OmegaConf.load(d / "config.yaml")
        name = str(cfg.comm.name)
        info = dict(run=str(cfg.run_name), seed=int(cfg.seed), radius=float(cfg.env.obs_radius), comm=name,
                    rho=None, ls=float(cfg.penalties.lambda_s))
        if name == "gated":
            info["rho"] = float(cfg.penalties.target_density)
            info["kind"] = "gated"
        elif name == "random_fixed":
            info["rho"] = float(cfg.comm.density)
            info["kind"] = "c1"
        else:
            info["kind"] = name  # nocomm | fullcomm
        out.append((d, info))
    return out


def shard(items, i, n):
    return [x for k, x in enumerate(items) if k % n == i]


# ----------------------------------------------------------------------------- evaluation
def do_eval(a):
    from eval import evaluate_run

    for d, info in shard(discover(a.exp), a.shard, a.n_shards):
        for tag in TAGS:
            if info["kind"] in ("nocomm", "fullcomm", "c1") and tag == "det":
                continue  # external gates: deterministic = stochastic
            s = evaluate_run(d, tag=tag, deterministic=(tag == "det"))
            print(f"{info['run']} s{info['seed']} {tag}: return {s['episode_return']:.3f} "
                  f"density {s['active_link_ratio']:.3f}", flush=True)


# ----------------------------------------------------------------------------- per-run stats
def run_stats(run_dir: Path, info: dict, tag: str, n_perm: int = 1000, overwrite: bool = False) -> dict:
    from analysis.events import event_stats, mantel_haenszel, stratum_counts, time_strata
    from analysis.metrics import distance_bins
    from analysis.null import circular_shift_null, compare_to_null, episode_shuffle_null
    from eval import load_logs

    cache = run_dir / f"v2stats_{tag}.json"
    if cache.exists() and not overwrite:
        return json.loads(cache.read_text())
    logs, summary = load_logs(run_dir / f"eval_{tag}.npz")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    g, E, mask = logs["g"], logs["E"], logs["mask"]
    act = (g > 0) & mask
    T = g.shape[1]
    out = dict(info, tag=tag, **{k: summary[k] for k in (
        "episode_return", "success_rate", "final_coverage", "active_link_ratio", "gate_switch_rate",
        "mean_run_length", "event_rate")})
    out["density_by_t"] = (act.sum((0, 2, 3)) / np.maximum(mask.sum((0, 2, 3)), 1)).tolist()
    out["event_by_t"] = ((E & mask).sum((0, 2, 3)) / np.maximum(mask.sum((0, 2, 3)), 1)).tolist()
    out["return_per_episode"] = logs["reward"].sum(-1).tolist()
    dens = out["active_link_ratio"]
    if info["kind"] not in ("gated", "c1") or dens <= 0 or dens >= 1:
        out["events"] = None  # constant gates: association undefined
        save_json(cache, out)
        return out

    edges = DIST_EDGES
    nb = len(edges) - 1
    ts = time_strata(g.shape)
    db = distance_bins(logs["positions"], edges, float(cfg.env.obs_radius))
    st_td = ts * nb + db
    obs = event_stats(g, E, mask, ts, T, alpha=logs["alpha"])
    mh_pooled = mantel_haenszel(stratum_counts(g, E, mask, np.zeros_like(ts), 1))
    mh_t = mantel_haenszel(stratum_counts(g, E, mask, ts, T))
    mh_td = mantel_haenszel(stratum_counts(g, E, mask, st_td, T * nb))
    shuf = episode_shuffle_null(g, E, mask, st_td, T * nb, n_perm=n_perm, seed=0)
    null_mh_td = mantel_haenszel(shuf["counts"])["log_or"]
    null_mh_pooled = mantel_haenszel(shuf["counts"].sum(-1, keepdims=True))["log_or"]
    circ = circular_shift_null(g, E, mask, n_shifts=n_perm, seed=0)
    out["events"] = dict(
        phi=obs["phi"], lift=obs["lift"], auroc_alpha=obs.get("auroc_alpha"),
        phi_time_stratified=obs["phi_stratified"],
        log_or_pooled=float(mh_pooled["log_or"]), log_or_time=float(mh_t["log_or"]),
        log_or_time_dist=float(mh_td["log_or"]), log_or_time_dist_se=mh_td.get("se"), distance_edges=edges,
        shuffle_vs_log_or_time_dist=compare_to_null(float(mh_td["log_or"]), null_mh_td),
        shuffle_vs_log_or_pooled=compare_to_null(float(mh_pooled["log_or"]), null_mh_pooled),
        circular_vs_phi=compare_to_null(obs["phi"], circ["null_pooled"]["phi"]),
        n_perm=n_perm,
    )
    save_json(cache, out)
    return out


def do_stats(a):
    for d, info in shard(discover(a.exp), a.shard, a.n_shards):
        for tag in TAGS:
            if (d / f"eval_{tag}.npz").exists():
                s = run_stats(d, info, tag, overwrite=a.overwrite)
                ev = s.get("events") or {}
                print(f"{info['run']} s{info['seed']} {tag}: logOR(t,d) {ev.get('log_or_time_dist', float('nan')):.3f}", flush=True)


# ----------------------------------------------------------------------------- aggregation helpers
def t_ci(x, ci=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), sd=float("nan"), n=0)
    m = float(x.mean())
    if n == 1:
        return dict(mean=m, lo=float("nan"), hi=float("nan"), sd=float("nan"), n=1)
    sd = float(x.std(ddof=1))
    h = float(sps.t.ppf((1 + ci) / 2, n - 1) * sd / np.sqrt(n))
    return dict(mean=m, lo=m - h, hi=m + h, sd=sd, n=n)


def holm(pvals: dict) -> dict:
    keys = [k for k in pvals if np.isfinite(pvals[k])]
    order = sorted(keys, key=lambda k: pvals[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    for k in pvals:
        out.setdefault(k, float("nan"))
    return out


def ratio_ci(full, nocomm, gated, n_boot=20000, seed=0):
    """Bootstrap CI of (FullComm - Gated) / (FullComm - NoComm), seeds resampled per group."""
    rng = np.random.default_rng(seed)
    f, n, g = (np.asarray(x, float) for x in (full, nocomm, gated))
    bf = f[rng.integers(0, f.size, (n_boot, f.size))].mean(1)
    bn = n[rng.integers(0, n.size, (n_boot, n.size))].mean(1)
    bg = g[rng.integers(0, g.size, (n_boot, g.size))].mean(1)
    r = (bf - bg) / (bf - bn)
    point = (f.mean() - g.mean()) / (f.mean() - n.mean())
    lo, hi = np.percentile(r, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi))


def load_all(exp, tag):
    rows = []
    for d, info in discover(exp):
        use = tag if info["kind"] == "gated" else "stoch"
        p = d / f"v2stats_{use}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    return rows


def group(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["seed"])


def vals(rows, key, sub=None):
    if sub:
        return [((r.get(sub) or {}).get(key, float("nan"))) for r in rows]
    return [r[key] for r in rows]


# ----------------------------------------------------------------------------- report
def do_report(a):
    tag = a.tag
    rows = load_all(a.exp, tag)
    if not rows:
        raise SystemExit("no statistics found; run eval and stats first")
    radius = a.radius
    rows = [r for r in rows if abs(r["radius"] - radius) < 1e-9]
    rhos = sorted({r["rho"] for r in rows if r["kind"] == "gated"}, reverse=True)
    lss = sorted({r["ls"] for r in rows if r["kind"] == "gated"})
    no, full = group(rows, kind="nocomm"), group(rows, kind="fullcomm")
    res = dict(tag=tag, radius=radius, rhos=rhos, lambda_s=lss, conditions={})

    def cond_summary(rs):
        return {k: t_ci(vals(rs, k)) for k in ("episode_return", "success_rate", "final_coverage",
                                               "active_link_ratio", "gate_switch_rate", "mean_run_length",
                                               "event_rate")} | {"seeds": [r["seed"] for r in rs]}

    res["conditions"]["nocomm"] = cond_summary(no)
    res["conditions"]["fullcomm"] = cond_summary(full)
    for rho in rhos:
        res["conditions"][f"c1_rho{rho:g}"] = cond_summary(group(rows, kind="c1", rho=rho))
        for ls in lss:
            res["conditions"][f"gated_rho{rho:g}_ls{ls:g}"] = cond_summary(group(rows, kind="gated", rho=rho, ls=ls))

    # ---- RQ1
    f_ret, n_ret = vals(full, "episode_return"), vals(no, "episode_return")
    gap = t_ci(np.mean(f_ret) - np.asarray(n_ret)) if len(f_ret) else None
    welch = sps.ttest_ind(f_ret, n_ret, equal_var=False)
    pooled_sd = np.sqrt((np.var(f_ret, ddof=1) + np.var(n_ret, ddof=1)) / 2)
    rq1 = dict(benefit=dict(diff=float(np.mean(f_ret) - np.mean(n_ret)), welch_p=float(welch.pvalue),
                            cohens_d=float((np.mean(f_ret) - np.mean(n_ret)) / pooled_sd)), per_rho={})
    p_c1 = {}
    for rho in rhos:
        g0 = group(rows, kind="gated", rho=rho, ls=lss[0])
        c1 = group(rows, kind="c1", rho=rho)
        gr = vals(g0, "episode_return")
        rc = ratio_ci(f_ret, n_ret, gr)
        verdict = ("no substantial loss" if rc["hi"] < MARGIN else
                   "substantial loss" if rc["lo"] > MARGIN else "inconclusive")
        entry = dict(loss_fraction=rc, verdict=verdict, gated_return=t_ci(gr),
                     success_vs_full=dict(gated=t_ci(vals(g0, "success_rate")), full=t_ci(vals(full, "success_rate"))))
        if c1:
            cr = vals(c1, "episode_return")
            w = sps.ttest_ind(gr, cr, equal_var=False)
            entry["vs_c1"] = dict(diff=float(np.mean(gr) - np.mean(cr)), welch_p=float(w.pvalue),
                                  c1_return=t_ci(cr), c1_loss_fraction=ratio_ci(f_ret, n_ret, cr))
            p_c1[rho] = float(w.pvalue)
        rq1["per_rho"][f"{rho:g}"] = entry
    for rho, p in holm(p_c1).items():
        rq1["per_rho"][f"{rho:g}"]["vs_c1"]["holm_p"] = p
    res["rq1"] = rq1

    # ---- RQ2 (paired by seed)
    rq2, p2, p2adj = dict(per=dict(), pooled=dict()), {}, {}
    base_ls = lss[0]
    # density-adjusted secondary (amendment 2): return-density slope from lambda_s = 0 runs + baselines
    xs = [r["active_link_ratio"] for r in rows if r["kind"] == "gated" and r["ls"] == base_ls]
    ys = [r["episode_return"] for r in rows if r["kind"] == "gated" and r["ls"] == base_ls]
    xs += [0.0] * len(no) + [1.0] * len(full)
    ys += vals(no, "episode_return") + vals(full, "episode_return")
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else 0.0
    rq2["density_slope"] = slope
    for ls in lss[1:]:
        pooled_diffs = []
        for rho in rhos:
            b = {r["seed"]: r for r in group(rows, kind="gated", rho=rho, ls=base_ls)}
            t = {r["seed"]: r for r in group(rows, kind="gated", rho=rho, ls=ls)}
            seeds = sorted(set(b) & set(t))
            d = {k: np.array([t[s][k] - b[s][k] for s in seeds]) for k in
                 ("episode_return", "success_rate", "active_link_ratio", "gate_switch_rate", "mean_run_length")}
            p = float(sps.ttest_1samp(d["episode_return"], 0).pvalue) if len(seeds) > 1 else float("nan")
            adj = d["episode_return"] - slope * d["active_link_ratio"]
            p_adj = float(sps.ttest_1samp(adj, 0).pvalue) if len(seeds) > 1 else float("nan")
            matched = bool(abs(np.mean(d["active_link_ratio"])) <= 0.05)
            persist = bool(np.mean(d["gate_switch_rate"]) < 0 and np.mean(d["mean_run_length"]) > 0)
            key = f"rho{rho:g}_ls{ls:g}"
            rq2["per"][key] = dict(rho=rho, ls=ls, seeds=seeds, p=p, density_matched=matched,
                                   persistence_increased=persist, d_return_adjusted=t_ci(adj), p_adjusted=p_adj,
                                   **{f"d_{k}": t_ci(v) for k, v in d.items()})
            p2[key] = p
            p2adj[key] = p_adj
            pooled_diffs += d["episode_return"].tolist()
        rq2["pooled"][f"ls{ls:g}"] = dict(d_return=t_ci(pooled_diffs),
                                          p=float(sps.ttest_1samp(pooled_diffs, 0).pvalue) if len(pooled_diffs) > 1 else float("nan"))
    for k, p in holm(p2).items():
        rq2["per"][k]["holm_p"] = p
    for k, p in holm(p2adj).items():
        rq2["per"][k]["holm_p_adjusted"] = p
    res["rq2"] = rq2

    # ---- RQ3
    rq3, p3, p3b = dict(per_condition={}, persistence={}), {}, {}
    for rho in rhos:
        for ls in lss:
            rs = group(rows, kind="gated", rho=rho, ls=ls)
            lo = vals(rs, "log_or_time_dist", "events")
            key = f"rho{rho:g}_ls{ls:g}"
            x = np.asarray(lo, float)
            x = x[np.isfinite(x)]
            p = float(sps.ttest_1samp(x, 0).pvalue) if x.size > 1 else float("nan")
            rq3["per_condition"][key] = dict(
                log_or_time_dist=t_ci(lo), log_or_time=t_ci(vals(rs, "log_or_time", "events")),
                log_or_pooled=t_ci(vals(rs, "log_or_pooled", "events")), phi=t_ci(vals(rs, "phi", "events")),
                phi_time_stratified=t_ci(vals(rs, "phi_time_stratified", "events")),
                lift=t_ci(vals(rs, "lift", "events")),
                shuffle_p_per_seed=[((r.get("events") or {}).get("shuffle_vs_log_or_time_dist") or {}).get("p_value") for r in rs],
                shuffle_excess_pooled=t_ci([((r.get("events") or {}).get("shuffle_vs_log_or_pooled") or {}).get("excess", np.nan) for r in rs]),
                circular_p_per_seed=[((r.get("events") or {}).get("circular_vs_phi") or {}).get("p_value") for r in rs],
                p=p)
            p3[key] = p
        b = {r["seed"]: r for r in group(rows, kind="gated", rho=rho, ls=lss[0])}
        for ls in lss[1:]:
            t = {r["seed"]: r for r in group(rows, kind="gated", rho=rho, ls=ls)}
            seeds = sorted(set(b) & set(t))
            d = np.array([(t[s].get("events") or {}).get("log_or_time_dist", np.nan) -
                          (b[s].get("events") or {}).get("log_or_time_dist", np.nan) for s in seeds])
            d = d[np.isfinite(d)]
            key = f"rho{rho:g}_ls{ls:g}"
            p = float(sps.ttest_1samp(d, 0).pvalue) if d.size > 1 else float("nan")
            rq3["persistence"][key] = dict(d_log_or=t_ci(d), p=p)
            p3b[key] = p
    for k, p in holm(p3).items():
        rq3["per_condition"][k]["holm_p"] = p
    for k, p in holm(p3b).items():
        rq3["persistence"][k]["holm_p"] = p
    res["rq3"] = rq3

    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / f"v2_results_{tag}.json", res)
    write_tables(res, rows, tag)
    if not a.no_plots:
        from v2_plots import make_all

        make_all(rows, res, OUT, tag)
    print(f"wrote {OUT}/v2_results_{tag}.json", flush=True)


def fmt(s, d=2):
    if s is None or not np.isfinite(s.get("mean", np.nan)):
        return "–"
    if not np.isfinite(s.get("lo", np.nan)):
        return f"{s['mean']:.{d}f}"
    return f"{s['mean']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def write_tables(res, rows, tag):
    import csv

    with open(OUT / f"v2_conditions_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["episode_return", "success_rate", "final_coverage", "active_link_ratio", "gate_switch_rate",
                "mean_run_length", "event_rate"]
        w.writerow(["condition", "n_seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "lo", "hi", "sd")])
        for name, c in res["conditions"].items():
            n = c["episode_return"]["n"]
            if n:
                w.writerow([name, n] + [c[k][s] for k in keys for s in ("mean", "lo", "hi", "sd")])
    with open(OUT / f"v2_runs_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["run", "seed", "kind", "rho", "ls", "episode_return", "success_rate", "active_link_ratio",
                "gate_switch_rate", "mean_run_length", "event_rate"]
        ev = ["phi", "lift", "phi_time_stratified", "log_or_pooled", "log_or_time", "log_or_time_dist"]
        w.writerow(cols + ev)
        for r in sorted(rows, key=lambda r: (r["run"], r["seed"])):
            e = r.get("events") or {}
            w.writerow([r.get(c) for c in cols] + [e.get(k) for k in ev])


def do_pilot(a):
    """Apply the pilot decision rules of ANALYSIS_PLAN.md to runs/v2_pilot (training metrics,
    last 20 % of updates) and to the 5-seed baselines in runs/v2."""
    import pandas as pd

    out = {}
    for R in (0.3, 0.5):
        rows = []
        for d in sorted((ROOT / "runs" / a.pilot_exp).glob(f"*_R{R:g}/seed_0")):
            cfg = OmegaConf.load(d / "config.yaml")
            m = pd.read_csv(d / "train_metrics.csv")
            tail = m[m["update"] >= 0.8 * m["update"].max()]
            P = cfg.penalties
            rows.append(dict(run=str(cfg.run_name), rho=float(P.target_density), ls=float(P.lambda_s),
                             lr=float(P.dual_lr), kp=float(P.get("dual_kp", 0.0)), p_on=float(tail.p_on_mean.mean()), p_on_min=float(tail.p_on_mean.min()),
                             p_on_sd=float(tail.p_on_mean.std()), density=float(tail.active_link_ratio.mean()),
                             switch=float(tail.gate_switch_rate.mean()), ret=float(tail.episode_return.mean()),
                             lambda_c_end=float(m.lambda_c.iloc[-1])))
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["rho", "lr", "ls"])
        print(f"\n=== pilot, obs radius {R:g}\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        lr_ok = df[(df.rho == 0.25) & (df.ls == 0) & ((df.p_on - 0.25).abs() <= 0.03) & (df.p_on_min >= 0.05)]
        lr = float(lr_ok.lr.min()) if len(lr_ok) else None
        base = df[(df.rho == 0.25) & (df.ls == 0) & (df.lr == a.base_lr) & (df.kp == df.kp.max())]
        sw0 = float(base.switch.iloc[0]) if len(base) else float("nan")
        cand = df[(df.rho == 0.25) & (df.ls > 0) & (df.kp == df.kp.max()) & ((df.p_on - 0.25).abs() <= 0.05)].sort_values("ls")
        mid = cand[cand.switch <= sw0 / 2]
        high = cand[cand.switch <= sw0 / 4]
        ls_mid = float(mid.ls.min()) if len(mid) else None
        ls_high = float(high.ls.min()) if len(high) else (float(cand.ls.max()) if len(cand) else None)
        if ls_high is not None and ls_mid is not None and ls_high <= ls_mid:
            later = cand[cand.ls > ls_mid]
            ls_high = float(later.ls.min()) if len(later) else ls_high
        base_rows = {k: [json.loads((d / "meta.json").read_text())["final_metrics"]["episode_return"]
                         for d in sorted((ROOT / "runs" / "v2" / f"{k}_R{R:g}").glob("seed_*")) if (d / "meta.json").exists()]
                     for k in ("nocomm", "fullcomm")}
        out[f"{R:g}"] = dict(table=df.to_dict("records"), dual_lr=lr, switch_ls0=sw0, ls_mid=ls_mid, ls_high=ls_high,
                             baselines_train_return=base_rows)
        print(f"rule 2 dual_lr -> {lr};  rule 3 lambda_s mid -> {ls_mid}, high -> {ls_high}  (switch at ls=0: {sw0:.4f})")
    # rule 1: observation radius from the evaluated 5-seed baselines
    radius = {}
    for R in (0.3, 0.5):
        ev = {}
        for k in ("nocomm", "fullcomm"):
            js = [json.loads((d / "eval_stoch.json").read_text())
                  for d in sorted((ROOT / "runs" / "v2" / f"{k}_R{R:g}").glob("seed_*")) if (d / "eval_stoch.json").exists()]
            ev[k] = js
        if len(ev["nocomm"]) < 2 or len(ev["fullcomm"]) < 2:
            continue
        f = np.array([j["episode_return"] for j in ev["fullcomm"]])
        n = np.array([j["episode_return"] for j in ev["nocomm"]])
        d = (f.mean() - n.mean()) / np.sqrt((f.var(ddof=1) + n.var(ddof=1)) / 2)
        radius[f"{R:g}"] = dict(fullcomm=f.tolist(), nocomm=n.tolist(), gap=float(f.mean() - n.mean()), cohens_d=float(d),
                                fullcomm_success=float(np.mean([j["success_rate"] for j in ev["fullcomm"]])),
                                event_rate=float(np.mean([j["event_rate"] for j in ev["fullcomm"]])))
        print(f"radius {R:g}: FullComm {f.mean():.3f} (sd {f.std(ddof=1):.3f})  NoComm {n.mean():.3f} (sd {n.std(ddof=1):.3f})  "
              f"gap {f.mean() - n.mean():.3f}  d {d:.2f}  FullComm success {radius[f'{R:g}']['fullcomm_success']:.2f}  "
              f"event rate {radius[f'{R:g}']['event_rate']:.3f}")
    ok = {R: v for R, v in radius.items() if v["fullcomm_success"] < 0.9 and 0.10 <= v["event_rate"] <= 0.30}
    if ok:
        best = max(ok, key=lambda R: (round(ok[R]["cohens_d"], 2), R == "0.3"))
        out["radius_rule"] = dict(per_radius=radius, chosen=float(best))
        print(f"rule 1 radius -> {best}")
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / f"{a.pilot_exp}_decisions.json", out)


def do_demo(a):
    """Link raster and animation of one evaluation episode (same layout) for NoComm, FullComm
    and learned gates at one target density with and without the switching penalty."""
    from v2_plots import animate, link_raster

    R = a.radius
    runs = {r["run"]: d for d, r in discover(a.exp) if r["seed"] == a.seed and abs(r["radius"] - R) < 1e-9}
    lss = sorted({float(n.split("_ls")[1].split("_")[0]) for n in runs if n.startswith(f"gated_rho{a.rho:g}_")})
    models = {"NoComm": runs.get(f"nocomm_R{R:g}"), "FullComm": runs.get(f"fullcomm_R{R:g}")}
    for ls in (lss[0], lss[-1]) if lss else ():
        models[f"learned, density {a.rho:g}, λs {ls:g}"] = runs.get(f"gated_rho{a.rho:g}_ls{ls:g}_R{R:g}")
    models = {k: v for k, v in models.items() if v is not None}
    OUT.mkdir(parents=True, exist_ok=True)
    link_raster({k: v for k, v in models.items() if k != "NoComm"}, OUT / "v2_demo_link_raster.png", ep=a.episode)
    animate(models, OUT / "v2_demo_episode.gif", ep=a.episode)
    print(f"wrote demo for {list(models)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["eval", "stats", "report", "demo", "pilot"])
    ap.add_argument("--exp", default="v2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="stoch", help="gate evaluation mode for gated models (stoch | det)")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--pilot-exp", default="v2_pilot", help="pilot: runs/<name> to summarise")
    ap.add_argument("--base-lr", type=float, default=3e-4, help="pilot: dual lr of the lambda_s = 0 reference run")
    ap.add_argument("--rho", type=float, default=0.25, help="demo: target density shown")
    ap.add_argument("--seed", type=int, default=0, help="demo: seed shown")
    ap.add_argument("--episode", type=int, default=0, help="demo: evaluation episode shown")
    a = ap.parse_args()
    {"eval": do_eval, "stats": do_stats, "report": do_report, "demo": do_demo, "pilot": do_pilot}[a.cmd](a)


if __name__ == "__main__":
    main()
