# Learned gated communication in cooperative MARL

This codebase implements `Detailed Experiment Specs, Code-Generation Prompt, and References.md`. It compares no communication, full communication, and learned binary gating with a communication-cost penalty and a switching-cost penalty. For each model it measures:

- performance
- link sparsity
- link persistence
- association between active links and a task event
- whether receivers actually use the messages

Stack: JAX + Flax + optax, Hydra configs, optional Weights & Biases, pytest. It was tested on Windows 11 with Python 3.10.11 on CPU. On native Windows, JAX runs only on the CPU. For GPU, use Linux or WSL2 and install `jax[cuda12]==0.6.2`. No code changes are needed.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows (Linux: .venv/bin/python)
.venv/Scripts/python -m pytest tests -q                       # 35 tests, ~35 s on CPU
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/{env,algo,comm,penalties}/` | Hydra config groups. Every hyperparameter is here. |
| `envs/spread.py` | Cooperative navigation in pure JAX: visibility, coverage, reward, observations, critic state, human-readable label |
| `core/comm.py` | KNN neighbour mask, non-learned gate modes (full, hand rule, random fixed, random Markov), event `E[t,i,j]` |
| `models/gates.py` | Hard-concrete / straight-through / REINFORCE gates, communication cost, switching cost, lambda schedule |
| `models/actor_critic.py` | Shared GRU actor with gated attention over messages and a null key; DeepSets centralized critic (never sees messages) |
| `core/rollout.py` | One env/policy step shared by training and evaluation, including test-time interventions and cross-play |
| `core/mappo.py` | MAPPO update: GAE, clipped PPO, value clipping, penalties, optional gradient accumulation |
| `train.py` | Training entry point (Hydra CLI or `run_training(cfg)`). Logs git hash and config. |
| `eval.py` | Evaluation episodes saved as `.npz` with the full logging schema |
| `analysis/` | Link metrics, event statistics, circular-shift null, bootstrap/IQM, faithfulness (F1–F4), probes, plots |
| `scripts/e1_*.py … e9_*.py` | One script per experiment (E1–E9) |
| `tests/` | Unit and integration tests |

Link convention: arrays indexed `[i, j]` are the directed link from **sender i** to **receiver j**.

## Model summary

- **Environment.** N agents, N landmarks, square arena with side `sqrt(N / density)` (constant density). The local view has radius `obs_radius`. The team reward is `-(sum_l min_i |p_i - l| + collision_penalty * #colliding pairs) / L`. A landmark is covered when an agent is within `cover_threshold` of it. Success means all landmarks are covered at once at some step.
- **Communication.** Each agent sends `m_i = tanh(W h_i)`. A gate `g_ij` comes from a sender-side gate net on `(h_i, (p_j - p_i)/R, |p_j - p_i|/R)`. Links exist only inside the spatial K-nearest-neighbour mask (`jax.lax.top_k`).
- **Receiver.** The receiver attends with query `W_q h_j`, keys `W_k m_i`, and values `W_v m_i` (no bias). Logits are masked where `g = 0`. A learned null key with zero value avoids NaNs when every link is closed. Values are multiplied by `g` so the task loss trains the gate.
- **Hard-concrete gate.** β = 0.5, γ = −0.1, ζ = 1.1. A gate is active when `g > 0`. Evaluation uses deterministic gates. Each eval log also stores `g_sampled`, and `eval_stoch.npz` holds full stochastic-gate rollouts.
- **Loss.** `L = L_MAPPO + λc · mean P(g>0) + λs · mean[p_t(1−p_{t−1}) + p_{t−1}(1−p_t)]`, taken over links permitted at both steps. Both λ are zero for `warmup_frac` of training, then ramp linearly over `anneal_frac`. Gates start near on (`gate_init_logit = 2`, so P(g>0) ≈ 0.96).

**Comm configs** (`comm=...`):

- `gated`, `gated_ste`, `gated_reinforce`, and `ic3net` (REINFORCE with one gate per sender)
- `fullcomm` and `nocomm` (`nocomm` multiplies messages by 0 and keeps the same graph and parameter count)
- `handrule`, `random_fixed`, `random_markov`
- `aecomm`: autoencoder-grounded messages, with the RL gradient blocked
- `language`: a 9-token vocabulary aligned to compass-direction labels

## Evaluation log schema (`runs/.../eval_<tag>.npz`, arrays `[episode, T, ...]`)

| Array | Contents |
| --- | --- |
| `positions` | `[T,N,2]` agent positions |
| `landmark_pos` | `[T,L,2]` landmark positions |
| `actions` | `[T,N]` actions taken |
| `action_logits` | `[T,N,A]` policy logits |
| `alpha` | `[T,i,j]` gate logits |
| `g` | `[T,i,j]` gates actually used |
| `g_sampled` | `[T,i,j]` stochastic gate sample, not fed back into the rollout |
| `p_on` | `[T,i,j]` P(g>0) |
| `mask` | `[T,i,j]` KNN neighbour mask |
| `messages` | `[T,i,:]` messages |
| `attn` | `[T,i,j]` attention weight that receiver j puts on sender i |
| `vis` | `[T,i,L]` landmark visibility per agent |
| `covered` | `[T,L]` landmark coverage |
| `E` | `[T,i,j]` event flags |
| `reward` | `[T]` team reward |
| `hidden` | `[T,N,H]` hidden states, used for the teacher-forced message tests |
| `human_label` | `[T,N]` human-readable label |
| `obs` | `[T,N,D]` observations |
| `success` | `[]` episode success |

## Run order

Every script accepts `--seeds`, `--quick` (tiny budget, separate `runs/quick_*` and `results/quick_*` namespaces), `--set <hydra overrides>` and `--retrain`. Runs are cached in `runs/core/<run_name>/seed_<s>/`, so later experiments reuse the models trained by earlier ones. Results and figures go to `results/<exp>/`.

1. `pytest tests -q`
2. Smoke-test every script: `python scripts/e1_task_validation.py --quick`, and so on for E2–E9 (each takes about 1 min).
3. **E1 task validation.** `python scripts/e1_task_validation.py --seeds 0 1 2 3 4`. Check `results/e1/e1_report.json → checks`. If a check fails, change the task first: shrink `env.obs_radius`, or use more landmarks (`env.n_landmarks`). Before you change the default task, delete `runs/core`.
4. **E2 pilot calibration.** `python scripts/e2_sparsity_sweep.py --pilot --lambdas 0 0.01 0.03 0.1 0.3 1`. Pick about 6 λc values whose densities span roughly 100% down to 10–20%. Then run the full sweep with `--seeds 0 1 2 3 4` (10 seeds if throughput allows).
5. **E3 persistence grid.** `python scripts/e3_persistence_grid.py --lc 0 <small> <large> --ls 0 <small> <large>`
6. **E4 matched controls.** `python scripts/e4_matched_controls.py --lambda-c <λ> --lambda-s 0`
7. **E5 event association.** `python scripts/e5_event_association.py --lc ... --ls ...` (same grid as E3, 1000 circular shifts)
8. **E6 message use.** `python scripts/e6_message_use.py --lc ... --ls ...`
9. **E7 probes.** `python scripts/e7_message_probe.py --lambdas <E2 values>`
10. **E8 method comparison.** `python scripts/e8_method_comparison.py --lambda-c <λ> --with-ic3net`
11. **E9 transfer and scale.** `python scripts/e9_transfer_scale.py --lambda-c <λ> --lambda-s <λ>`. For clean peak-memory numbers, run each N in its own process (`--ns 50`). If memory runs out, lower `--large-num-envs` and raise `--grad-accum`.

Single runs: `python train.py comm=gated penalties.lambda_c=0.1 seed=3`, then `python eval.py --run runs/core/<run_name>/seed_3 [--stochastic]`.

**Budget (CPU, N = 4).** Training runs at about 3,500 env-steps/s. The default 1,500 updates × 64 envs × 50 steps (4.8M env steps) takes about 25 min per run. E1–E3 with 5 seeds is about 95 runs, roughly 40 CPU-hours sequentially. Running several scripts or seeds in parallel processes is safe because each run writes only to its own directory. A 300-update FullComm pilot reached a success rate of about 0.3–0.5.

## Assumptions (stated explicitly)

- **Actions and dynamics.** Discrete actions {noop, ±x, ±y} with MPE-like damped point-mass dynamics. Positions are clipped to the arena.
- **Observations.** The k nearest landmarks and agents (zeroed when outside `obs_radius`), plus own position, velocity and time. Observations have a fixed size, so the same weights run at any N (needed for the N=50→100 transfer).
- **Episodes.** Fixed length, and each MAPPO update collects exactly one full episode per environment, so the GRU always starts from zero.
- **Messages.** One communication round per step. Messages affect only the current action and are not fed back into the GRU, so the E6 tests are one-step tests.
- **Gate inputs.** The gate is computed on the sender side and may use the receiver's relative position: agents know where their KNN neighbours are.
- **Gate noise replay.** Gate noise from the rollout is stored and replayed in the PPO update, so the ratio is computed for the same reparameterised policy. REINFORCE gates store their samples and add their log-probabilities to the PPO ratio.
- **Rewards.** Rewards are multiplied by `algo.reward_scale = 0.1` inside GAE and value targets only. Logged returns are unscaled.
- **Null model.** Each link's gate series is rotated among the timesteps at which that link is permitted. This keeps per-link density and autocorrelation.
- **F4 prediction (pre-registered in `analysis/faithfulness.py`).** The receiver's expected action shift `Δv` should satisfy `Δv · [unit(p_i + d_new − p_j) − unit(p_i + d_orig − p_j)] > 0`. Chance level is 0.5.
- **Bits per step (E8).** Continuous messages are counted at a nominal 32 bits per dimension; language tokens at log2(V).

## Known limitations

- On native Windows, JAX runs only on the CPU. Large-N runs (N=50, and the N=100 transfer) are slow there and should run on Linux or WSL2 with a GPU.
- Peak memory is the process-lifetime peak (Windows working set, or the GPU allocator peak), so it accumulates across runs in one process.
- No λ calibration has been run yet. The default λ values are placeholders, and `--pilot` in E2 is the intended way to choose them.
- Event analysis shows association, not causation. Faithfulness tests measure message influence, not the reasons behind an action.
- `rliable` is used for IQM. A scipy trimmed-mean fallback is used if it is missing.
- The second task in E9 is a second observation radius (`env=spread_small_robs`), not a different environment.
- Code generated with an assistant: review the visibility logic, the event definition and the null model (all unit-tested in `tests/`) before relying on results.
