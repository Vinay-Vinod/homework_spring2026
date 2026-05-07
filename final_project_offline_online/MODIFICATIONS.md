# Modification log

Tracks changes made during the CS 185 offline-to-online final project (file path + short description).

| Date       | File | Change |
|------------|------|--------|
| 2026-04-08 | `problem/src/infrastructure/pytorch_util.py` | Part 0: Inserted `nn.LayerNorm(size)` after each hidden `Linear` and before the activation in `build_mlp` and in `build_ensemble_mlp` (`_build_single`). |
| 2026-04-09 | `problem/src/agents/sacbc_agent.py` | Part I: Implemented `update_q` (twin-Q Bellman with mean of target Qs, no entropy in backup), `update_actor` (Q-max + BC + entropy via reparameterization), `update_target_critic` (Polyak averaging). |
| 2026-04-09 | `problem/src/agents/fql_agent.py` | Part I: Implemented `get_action` (one-step policy with random noise), `get_bc_action` (Euler ODE integration of flow policy), `update_q` (Bellman with one-step actor for next actions), `update_bc_actor` (flow-matching loss), `update_onestep_actor` (distillation from BC flow + Q-maximization), `update_target_critic` (Polyak averaging). |
| 2026-04-09 | `problem/src/scripts/train_offline_online.py` | Part I: Implemented `run_offline_training_loop` (dataset sampling, agent creation, training, eval), `run_online_training_loop` (env rollout, replay buffer, online updates, eval), updated `main` to chain offline→online and added `--offline_data` / `--wsrl_steps` CLI args for Part II. |
