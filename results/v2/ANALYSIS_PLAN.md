# v2 analysis plan (fixed before the main runs)

Written 2026-11-04 at about 16:10, while only phase A (baselines and pilot) was training. No main-grid results existed when this plan was fixed. The research questions are the three in `Need2Know_ProjectProposal.pdf`.

## Design

| Item | Choice | Reason |
| --- | --- | --- |
| Seeds | 5 per condition (0–4) | The proposal asks for variation across runs; in v1, run-to-run differences of up to 1.5 return appeared between equivalent configurations |
| Training | 3,000 MAPPO updates × 64 envs × 50 steps = 9.6 M env steps per run | v1 runs (1,500 updates) were still improving |
| Link count | Learned gates trained to a **target density** ρ* ∈ {0.5, 0.25, 0.1}: λc is a Lagrange multiplier set by dual ascent so that the mean P(g > 0) tracks a target ramping from 1 to ρ* (warm-up 20 %, ramp 30 %) | RQ2 asks about persistence "independently of link count". With fixed λc, v1 runs were all-on or nearly-off, and no λs > 0 model matched a λs = 0 model in density |
| Persistence | λs ∈ {0, mid, high}, ramped like v1 | mid/high are chosen from the pilot (rule below) |
| Baselines | NoComm and FullComm; random fixed mask at each ρ* trained from scratch (C1) | The proposal's baselines, plus a control for "learned vs. random links at the same count" |
| Evaluation | 200 episodes per run on the same layouts for every model. **Primary: sampled gates** (as in training, and as actions are sampled); deterministic gates reported as secondary | In v1, deterministic thresholding produced densities the policy never saw in training |

## Pilot decision rules (phase A)

1. **Observation radius.** Use whichever of 0.3 and 0.5 gives the larger standardised communication effect, d = (mean FullComm − mean NoComm) / pooled SD over 5 seeds. FullComm must not be at ceiling (success < 0.9), and the event rate must lie in 10–30 %. Ties go to 0.3.
2. **Dual step size.** Use the smallest of {1e-4, 3e-4, 1e-3} whose achieved P(on) over the last 20 % of training is within ±0.03 of the target at ρ* = 0.25, with no collapse below 0.05.
3. **λs.** At ρ* = 0.25, "mid" is the smallest pilot λs that at least halves the per-step gate switch rate relative to λs = 0. "High" is the smallest that cuts it by at least 4×. Both must keep density within ±0.05 of the target. If no λs reaches 4×, use the largest that still holds density.

## Primary analyses

Unit of analysis: **one number per run** (seed). Link-timesteps are never pooled across seeds.

**RQ1: fewer links without substantial loss?**
- Primary measure: evaluation return.
- Communication benefit Δ = FullComm − NoComm.
- For each ρ* (λs = 0), the loss is L = FullComm − Gated(ρ*).
- **Pre-specified margin: a "substantial" loss is more than 25 % of Δ.**
- Verdict from the 95 % CI of L / Δ (bootstrap over seeds, resampled independently per condition):
  - upper bound < 0.25 → **no substantial loss**
  - lower bound > 0.25 → **substantial loss**
  - otherwise → inconclusive
- Secondary: success rate; learned vs. C1 random mask at the same ρ* (Welch t, Holm over the 3 densities).

**RQ2: does persistence affect performance independently of link count?**
- At each ρ*, compare λs ∈ {mid, high} with λs = 0.
- Runs with the same seed share initialisation and the warm-up phase, so differences are **paired by seed**.
- Paired t-test on return, Holm-corrected over the 6 comparisons.
- Manipulation checks, reported with the result:
  - switch rate and mean link run length must change;
  - achieved density must stay matched (|Δ density| ≤ 0.05).
- If a manipulation check fails, the comparison is reported as not interpretable.
- An overall λs effect (all densities pooled, 15 paired differences per λs level) is also reported.

**RQ3: are active links associated with the task event beyond chance, and does persistence change it?**
- Event: the sender sees an uncovered landmark that the receiver cannot see (unchanged from v1).
- Primary statistic per run: the **Mantel–Haenszel log odds ratio** between gate-on and event, stratified by **time step × sender–receiver distance bin**. This controls the time-in-episode confound found in v1 and the dependence of the event on distance.
- Shuffled baseline (the proposal's wording): whole gate sequences are re-paired with a different episode's events (same link, same time steps; 1,000 permutations), which keeps each gate series' time profile and persistence.
- Across seeds: mean log OR with a 95 % t-CI; one-sample t-test against 0, Holm over the 9 gated conditions.
- Persistence comparison: paired difference in log OR (λs > 0 vs. λs = 0) at each ρ*, Holm over 6.
- Reported alongside, for comparability with v1: pooled φ, lift, the circular-shift null.
- Interpretation is association only, as the proposal states.

## What will be reported regardless of outcome

All conditions, all seeds (per-seed points in every figure), failed or collapsed runs, and any deviation from this plan, with the reason.

## Amendment 1 (16:40, after pilot 1, before any main-grid run)

Pilot 1 (`runs/v2_pilot`, decisions in `v2_pilot_decisions.json`) showed that the pure dual-ascent controller cannot hold a matched density.
- **Collapse:** at obs radius 0.3, two of the four λs = 0 runs collapsed to P(on) < 0.01. A collapsed hard-concrete gate has almost no gradient, so a negative λc cannot reopen it.
- **λs > 0 still blocks matched density:** λs = 0.1 kept every gate open. λs = 0.01–0.03 collapsed to about 0. This is the v1 problem: the expected-flip cost has a barrier at P = 0.5 that all gates must cross together while sparsifying.

Rules 2–3 could not be met for λs > 0 at radius 0.3. Changes:
1. **Controller:** PI with anti-windup. λc = clip(I + Kp·e), I = clip(I + Ki·e), with Ki = 1e-3 (pilot 1's best) and Kp = 0.05.
2. **Schedule:** **λs ramps in only after the density target is reached** (λs warm-up 50 %, ramp to 70 %). The density target still ramps over 20–50 %. Persistence is therefore imposed on an already-sparse gating policy, at a density the controller keeps matched.
3. **Pilot 2** (`runs/v2_pilot2`, seed 0, ρ* = 0.25, λs ∈ {0, 0.003, 0.01, 0.03, 0.1}, plus Kp = 0 and ρ* ∈ {0.1, 0.5} checks) applies rule 3 unchanged to pick λs mid/high. If the PI controller holds density within ±0.03 of the target at λs = 0, it is used for the main grid.

The analyses, tests, margin and verdict rules above are unchanged.

## Pilot 2 outcome and final main-grid settings (17:15, before any main-grid result)

At obs radius 0.3 and ρ* = 0.25 (`v2_pilot2_decisions.json`), the PI controller held P(on) at 0.247 with λs = 0 and at 0.24–0.26 for every λs > 0. With Kp = 0 it reached only 0.20 ± 0.10, so the proportional term is needed. Switch rates were 0.180 (λs = 0), 0.097 (0.003), 0.053 (0.01), 0.035 (0.03) and 0.014 (0.1).

Rule 3 gives **λs mid = 0.01** (3.4× fewer switches) and **λs high = 0.03** (5.1×).

Main grid:
- ρ* ∈ {0.5, 0.25, 0.1} × λs ∈ {0, 0.01, 0.03} × seeds 0–4
- Ki = 1e-3, Kp = 0.05; λs ramp 50 % → 70 %
- obs radius 0.3 (rule 1: d = 8.2 vs. 4.3 at 0.5)
- C1 random fixed-mask controls at the same ρ*, same radius, 5 seeds

The radius-0.5 copy of pilot 2 was stopped once rule 1 had chosen 0.3.

## Amendment 2 (about 17:35, from pilot-2 evaluations only; no main-grid run had finished)

Evaluating the pilot-2 checkpoints showed two things.

**1. Final-checkpoint density can miss the target.** The controller oscillates around the target, and training stops at an arbitrary point. Evaluated densities were 0.23–0.33 at ρ* = 0.25, and 0.26 for ρ* = 0.1 with λs = 0.03.
- RQ2's manipulation check (mean |Δ density| ≤ 0.05) therefore stays as written.
- New **secondary, density-adjusted RQ2 comparison**:
  - Fit the return–density slope b by least squares on every λs = 0 gated run (all ρ*), including NoComm (density 0) and FullComm (density 1).
  - Adjust each paired difference: Δreturn − b · Δdensity.
  - Test the adjusted differences in the same way: paired t, Holm over 6.
- If the manipulation check fails for a comparison, the adjusted result is the one interpreted.

**2. Finer distance strata.** At obs radius 0.3 the default distance bins (0, 0.5, 1, 1.5, 2, ∞ × R) put most links in the last bin. The RQ3 strata use edges **0, 0.5, 1, 1.5, 2, 3, 4, 6, ∞ × R** instead.
