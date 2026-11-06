"""End-to-end smoke tests: a few training updates, evaluation logging, replay consistency."""
import jax
import numpy as np
import pytest

from analysis.faithfulness import consistency_check, f1_zero_messages, make_receiver_fn
from core.rollout import Interv
from eval import evaluate, load_logs, save_logs
from train import run_training, run_training_batch
from core.utils import load_checkpoint

TINY = ["algo.total_updates=2", "algo.num_envs=4", "algo.num_minibatches=2", "algo.update_epochs=1",
        "env.max_steps=10", "eval.episodes=4", "logging.log_every=1"]

LOG_KEYS = ["positions", "actions", "action_logits", "alpha", "g", "g_sampled", "mask", "messages", "attn",
            "vis", "covered", "E", "reward", "hidden", "landmark_pos", "success"]


@pytest.mark.parametrize("comm", ["gated", "gated_reinforce", "language"])
def test_train_eval_roundtrip(make_cfg, tmp_path, comm):
    cfg = make_cfg(f"comm={comm}", "penalties=persist", "penalties.warmup_frac=0.0", f"out_dir={tmp_path.as_posix()}/run", *TINY)
    out = run_training(cfg, verbose=False)
    cfg2, params = load_checkpoint(out)
    logs, summary = evaluate(cfg2, params, n_episodes=4)
    for k in LOG_KEYS:
        assert k in logs, k
    T, N = cfg.env.max_steps, cfg.env.n_agents
    assert logs["g"].shape == (4, T, N, N) and logs["vis"].shape == (4, T, N, N)
    assert np.isfinite(logs["action_logits"]).all()
    assert 0 <= summary["active_link_ratio"] <= 1
    # teacher-forced receiver reproduces logged logits exactly
    f = make_receiver_fn(cfg2, params)
    assert consistency_check(f, logs) < 1e-4
    assert f1_zero_messages(f, logs)["kl_mean_all"] >= 0
    save_logs(tmp_path / "e.npz", logs, summary)
    logs2, meta = load_logs(tmp_path / "e.npz")
    assert meta["n_episodes"] == 4 and np.array_equal(logs2["g"], logs["g"])


def test_eval_interventions_and_controls(make_cfg, tmp_path):
    cfg = make_cfg("comm=fullcomm", f"out_dir={tmp_path.as_posix()}/run", *TINY)
    out = run_training(cfg, verbose=False)
    cfg2, params = load_checkpoint(out)
    logs, s = evaluate(cfg2, params, n_episodes=4, interv=Interv(zero_messages=True))
    assert np.allclose(logs["messages"], 0)
    logs, s = evaluate(cfg2, params, n_episodes=4, comm_overrides={"gate_mode": "random_fixed", "density": 0.0})
    assert s["active_link_ratio"] == 0
    logs, s = evaluate(cfg2, params, n_episodes=4, interv=Interv(ablate="event"))
    assert not ((logs["g"] > 0) & logs["E"]).any()
    # zero-shot transfer to a larger N with the same weights
    logs, s = evaluate(cfg2, params, n_episodes=2, env_overrides={"n_agents": 8})
    assert logs["g"].shape[-1] == 8
    # cross-play
    logs, s = evaluate(cfg2, params, n_episodes=2, sender_params=params["actor"])
    assert np.isfinite(logs["action_logits"]).all()


def test_batched_training_matches_single(make_cfg, tmp_path):
    """vmapped batch training reproduces independent single runs (same seeds, same lambdas)."""
    ov = ["comm=gated", "penalties=persist", "penalties.warmup_frac=0.0", *TINY]
    singles, batch = [], []
    for s in (0, 1):
        singles.append(make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/single_{s}"))
        batch.append(make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/batch_{s}"))
    for c in singles:
        run_training(c, verbose=False)
    run_training_batch(batch, verbose=False)
    for s in (0, 1):
        _, p1 = load_checkpoint(tmp_path / f"single_{s}")
        _, p2 = load_checkpoint(tmp_path / f"batch_{s}")
        diff = jax.tree_util.tree_map(lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), p1, p2)
        assert max(jax.tree_util.tree_leaves(diff)) < 1e-3


def test_batched_training_rejects_mixed_static_settings(make_cfg, tmp_path):
    a = make_cfg("comm=gated", f"out_dir={tmp_path.as_posix()}/a", *TINY)
    b = make_cfg("comm=fullcomm", f"out_dir={tmp_path.as_posix()}/b", *TINY)
    with pytest.raises(ValueError):
        run_training_batch([a, b], verbose=False)


def test_density_target_logs_dual_lambda(make_cfg, tmp_path):
    cfg = make_cfg("comm=gated", "penalties.target_density=0.3", "penalties.warmup_frac=0.0",
                   "penalties.anneal_frac=0.5", "penalties.dual_lr=0.01", f"out_dir={tmp_path.as_posix()}/run",
                   *TINY[:-1], "algo.total_updates=6", "logging.log_every=1")
    out = run_training(cfg, verbose=False)
    import pandas as pd

    m = pd.read_csv(out / "train_metrics.csv")
    assert np.isclose(m["target_density"].iloc[-1], 0.3)
    # gates start near-on (P ~ 0.96) and the target falls below that, so lambda_c must rise
    assert m["lambda_c"].iloc[-1] > 0
