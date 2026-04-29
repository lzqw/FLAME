# Safe-Pullback MeanFlow / Flow Control: Training Pipeline

This document codifies the intended implementation flow for the Safe-Pullback RF2/SAC-Entropy extension, based on `rf2_sac_ent.py` as the algorithmic backbone.

## Key separation

- Raw action: `raw_action ~ pi_theta(.|s)`
- Executed action: `exec_action = S_x(raw_action)`
- Environment transitions and reward critic use **executed action**.
- Entropy/log-likelihood remains on **raw action distribution**.
- Projection/intervention critic uses **raw action**.

## Environment-step contract

Each transition should expose:

- `raw_action`
- `exec_action`
- `projection_residual = ||raw - exec||`
- `projection_cost = projection_residual^2`
- `safe_violation`, `filter_active`, `state_violation`, `is_success`

## Replay tuple contract

`(obs, raw_action, action, reward, done, next_obs, projection_residual, projection_cost, ...)`

where `action` is the executed action.

## Critic updates

### Reward critic

- Input: `(s, exec_action)`
- Target next action generation:
  1) sample `raw_next ~ pi_theta(.|s')`
  2) project `exec_next = S_{x'}(raw_next)`
  3) backup with entropy term from `log pi(raw_next|s')`

### Intervention critic Q_S

- Input: `(s, raw_action)`
- TD target:
  `projection_cost + gamma_s * (1-done) * V_S_target(next_obs)`

### Intervention value V_S

- Fit toward policy expectation of Q_S:
  sample multiple raw actions from current policy and regress `V_S(s)` to average `Q_S(s, raw)`.

## Actor update (RF2 weighted loss)

For RF2 candidate raw actions:

1) project each candidate to executed action for reward scoring,
2) compute reward score from reward critic,
3) compute intervention penalty from `Q_S(s, raw)` (optionally immediate projection penalty),
4) build safe-pullback score and softmax weights,
5) reuse existing RF2 weighted flow loss with these weights.

## Training phases

1) Shielded exploration (`lambda_s ~ 0`)
2) Safe-pullback shaping (`lambda_s` warmup)
3) Autonomy refinement (lower FAR/APR, higher feasible raw-action ratio)

## Metrics to always log

- `q1_loss`, `q2_loss`, `qp_loss`, `vp_loss`, `policy_loss`
- `alpha`, `lambda_s_current`
- `projection_cost_batch`, `projection_residual_batch`
- `safe_violation_batch`, `filter_active_batch`
- `feasible_raw_action_ratio_batch`

## Evaluation artifacts

- `rollouts.npz`
- `summary.json`
- trajectory, occupancy, and prefilter-risk figures

## Current status in repository

`safe_pullback_*` files are intended as extension points from `rf2_sac_ent` and should be brought into full parity with this contract.
