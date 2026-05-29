# Traffic Rule Compliant Autonomous Driving Using Reward Machines

Code repository for the manuscript:

> **Traffic Rule Compliant Autonomous Driving Using Reward Machines and Reinforcement Learning**

---

## Structure

```
├── reward_machine/                     # Reward Machine implementation (from https://github.com/RodrigoToroIcarte/reward_machines.git)
├── rm_examples/                        # RM specification files (.txt)
├── train_hwyenv_stopandgo_rm.py        # RM-PPO training — Toy Case
├── train_hwyenv_stopandgo_lstmppo.py   # LSTM-PPO baseline — Toy Case
├── train_sumo_intersection_fcfs_rm.py  # RM-PPO training — SUMO unsignalized intersection
├── train_sumo_intersection_fcfs_lstmppo.py  # LSTM-PPO baseline — SUMO unsignalized intersection
├── eval_multiseed_s2g.py               # Multi-seed eval — Toy Case RM-PPO
├── eval_multiseed_s2g_lstmppo.py       # Multi-seed eval — Toy Case LSTM-PPO
├── eval_multiseed_fcfs.py              # Multi-seed eval — SUMO unsignalized intersection RM-PPO
├── eval_multiseed_fcfs_lstmppo.py      # Multi-seed eval — SUMO unsignalized intersection LSTM-PPO
├── train_callback.py                   # SB3 training callback (shared utility)
└── requirements.txt
```

---

## Installation

### 1. Python Environment

Python 3.8 or later is required.

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. PyTorch

Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your platform and CUDA version. For CPU-only:

```bash
pip install torch
```

### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. SUMO (required for FCFS intersection experiments)

SUMO is **not** pip-installable and must be built or installed separately.
Follow the official build/install instructions at:

> https://sumo.dlr.de/docs/Developer/index.html#build_instructions

After installing SUMO, set the environment variable before running any SUMO script:

```bash
export SUMO_HOME=/path/to/your/sumo    # macOS/Linux
# set SUMO_HOME=C:\path\to\sumo        # Windows
```

The SUMO scripts detect `$SUMO_HOME` automatically and add the `tools/` directory to `sys.path` so that `traci` and `sumolib` resolve correctly.

---

## Usage

### Toy case: Stopping at a stop sign

#### `train_hwyenv_stopandgo_rm.py` — RM-PPO (proposed method)

Trains a PPO agent on the toy case using a Reward Machine product-MDP wrapper (`HighwayStopEnvRM`). All hyperparameters are configured via constants at the top of the `if __name__ == "__main__"` block:

| Variable | Default | Description |
|---|---|---|
| `pre_train` | `True` | `True` = run training; `False` = load saved model and evaluate |
| `isPPO` | `True` | `True` = use PPO; `False` = use DQN |
| `learn_itrs` | `2048 * 60` | Total training timesteps (≈ 122 880) |

```bash
python train_hwyenv_stopandgo_rm.py
```

Outputs: trained model `ppo_hwy_sg_rm_rss_<steps>.zip`, TensorBoard logs in `./ppo_hwy_tensorboard/`.

---

#### `train_hwyenv_stopandgo_lstmppo.py` — LSTM-PPO baseline

Trains either a RecurrentPPO (LSTM) baseline (`HighwayStopEnvLSTM`) or the RM-PPO agent, selected by the `USE_LSTM` flag in the script:

| Variable | Default | Description |
|---|---|---|
| `USE_LSTM` | `True` | `True` = LSTM-PPO baseline; `False` = RM-PPO |
| `pre_train` | `True` | `True` = run training; `False` = load and evaluate |
| `learn_itrs` | `2048 * 60` | Total training timesteps |

```bash
python train_hwyenv_stopandgo_lstmppo.py
```

Outputs: `lstm_ppo_hwy_sg_dense_<steps>.zip` (LSTM) or `ppo_hwy_sg_rm_rss_<steps>.zip` (RM), TensorBoard logs in `./lstm_ppo_hwy_tensorboard/`.

---

### SUMO: Unsignalized Intersection

> **Note:** `$SUMO_HOME` must be set and SUMO must be installed before running these scripts.

#### `train_sumo_intersection_fcfs_rm.py` — RM-PPO (proposed method)

Trains or evaluates a PPO/DQN agent on the SUMO unsignalized intersection with a full FCFS Reward Machine wrapper (`AllWayStopEnvRM`).

```
python train_sumo_intersection_fcfs_rm.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | choice | `train` | `train`, `eval`, `demo`, `debug`, or `sim` |
| `--algorithm` | choice | `DQN` | RL algorithm: `DQN` or `PPO` |
| `--timesteps` | int | `50000` | Total training timesteps |
| `--model` | str | `None` | Path to a saved model for evaluation |
| `--episodes` | int | `10` | Number of evaluation episodes |
| `--sumo` | flag | — | Use SUMO/TraCI backend (needed) |
| `--render` | flag | — | Render during evaluation |
| `--seed` | int | `0` | Random seed |
| `--log-fail-only` | flag | — | Only log failed episodes (mutually exclusive with `--no-log`) |
| `--no-log` | flag | — | Suppress all evaluation logging (mutually exclusive with `--log-fail-only`) |
| `--trajectories` | int | `10000` | (sim mode) Number of trajectories to sample |
| `--max-len` | int | `500` | (sim mode) Max trajectory length in steps |
| `--output` | str | `None` | (sim mode) Output pickle path |
| `--rho` | float | `0.2` | Policy update period in seconds |

**Example — train with SUMO backend:**
```bash
python train_sumo_intersection_fcfs_rm.py --mode train --algorithm PPO --timesteps 122880 --sumo --seed 0
```

**Example — evaluate a saved model:**
```bash
python train_sumo_intersection_fcfs_rm.py --mode eval --model ppo_allway_stop_122880 --episodes 100 --sumo --no-log
```

---

#### `train_sumo_intersection_fcfs_lstmppo.py` — LSTM-PPO baseline

Trains or evaluates the LSTM-PPO baseline (`AllWayStopEnvLSTM` + `RecurrentPPO`) on the SUMO unsignalized intersection.

```
python train_sumo_intersection_fcfs_lstmppo.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | choice | `train` | `train` or `eval` |
| `--timesteps` | int | `122880` | Training timesteps |
| `--episodes` | int | `100` | Number of evaluation episodes |
| `--model` | str | `None` | Model path for eval (auto-derived from `--timesteps` if omitted) |
| `--sumo` | flag | `True` | Use SUMO/TraCI backend (needed) |
| `--render` | flag | `False` | Render with SUMO-GUI during evaluation |
| `--seed` | int | `None` | Random seed (random each run if omitted) |

**Example — train:**
```bash
python train_sumo_intersection_fcfs_lstmppo.py --mode train --timesteps 122880 --seed 0
```

**Example — evaluate:**
```bash
python train_sumo_intersection_fcfs_lstmppo.py --mode eval --model lstm_ppo_allway_stop_122880 --episodes 100
```

---

### Multi-Seed Evaluation

All four `eval_multiseed_*.py` scripts share the same seed-selection and training-loop interface. They train from scratch across multiple independent seeds and log four (highway-env) or five (SUMO) metrics at every evaluation checkpoint. Results are written to `.npy` arrays and an incremental `.csv` file (crash-safe).

**Shared arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--seeds` | int | `10` | Number of seeds (ignored when `--seed`, `--seed-start/--seed-end`, or `--seed-list` is used) |
| `--seed` | int | `None` | Run a single seed index |
| `--seed-start` | int | `None` | Start of inclusive seed range (requires `--seed-end`) |
| `--seed-end` | int | `None` | End of inclusive seed range (requires `--seed-start`) |
| `--seed-list` | str | `None` | Explicit seed list, e.g. `'1,3,5,6'` or `'[1,3,5,6]'` |
| `--steps` | int | per-script | Training steps per seed |
| `--eval-freq` | int | `4096` | Evaluate every N environment steps |
| `--eval-eps` | int | `100` | Evaluation episodes per checkpoint |
| `--plot-only` | flag | — | Skip training; reload saved `.npy` files and replot |
| `--plot-only-with-txt` | flag | — | Skip training; reload `*_results.txt` and replot |

> `--plot-only` and `--plot-only-with-txt` are mutually exclusive.

---

#### `eval_multiseed_s2g.py` — Toy case with RM-PPO, multi-seed

Multi-seed training + evaluation for the toy case: stopping at a stop sign, using RM-PPO (`HighwayStopEnvRM`). Default: 10 seeds * 122 880 steps, evaluated every 4 096 steps over 100 episodes. Metrics: M_goal, M_safe, M_stop, avg_d_stop.

```bash
# Run all 10 seeds
python eval_multiseed_s2g.py

# Run a single seed
python eval_multiseed_s2g.py --seed 3

# Run a range of seeds
python eval_multiseed_s2g.py --seed-start 0 --seed-end 4

# Replot from previously saved .npy files
python eval_multiseed_s2g.py --plot-only
```

---

#### `eval_multiseed_s2g_lstmppo.py` — Toy case with LSTM-PPO, multi-seed

Identical protocol to `eval_multiseed_s2g.py`, but trains the LSTM-PPO baseline (`HighwayStopEnvLSTM` + `RecurrentPPO`). The recurrent hidden state `(h_t, c_t)` is preserved across every evaluation step.

```bash
python eval_multiseed_s2g_lstmppo.py

python eval_multiseed_s2g_lstmppo.py --seed-list '0,2,5,7,9'

python eval_multiseed_s2g_lstmppo.py --plot-only
```

---

#### `eval_multiseed_fcfs.py` — SUMO unsignalized intersection RM-PPO, multi-seed

Multi-seed training + evaluation for the SUMO unsignalized intersection using RM-PPO (`AllWayStopEnvRM`). Each seed creates two separate SUMO/TraCI processes (`'train'` and `'eval'` labels) that coexist in the same Python process without conflict. Default: 10 seeds * 122 880 steps. Metrics: M_goal, M_safe, M_stop, M_yield, avg_d_stop.

> **Requires `$SUMO_HOME` to be set.**

```bash
python eval_multiseed_fcfs.py

python eval_multiseed_fcfs.py --seed 0 --steps 122880

python eval_multiseed_fcfs.py --plot-only-with-txt
```

---

#### `eval_multiseed_fcfs_lstmppo.py` — SUMO unsignalized intersection LSTM-PPO, multi-seed

Identical to `eval_multiseed_fcfs.py`, but uses the LSTM-PPO baseline (`AllWayStopEnvLSTM` + `RecurrentPPO`). FCFS priority is still tracked internally via `env.fcfs_queue` (hidden from the policy), so M_yield is evaluated identically.

> **Requires `$SUMO_HOME` to be set.**

```bash
python eval_multiseed_fcfs_lstmppo.py

python eval_multiseed_fcfs_lstmppo.py --seed-start 5 --seed-end 9

python eval_multiseed_fcfs_lstmppo.py --plot-only
```

---

## Metrics

| Metric | Description |
|---|---|
| **M_goal** | Fraction of episodes where ego cleared the intersection after a valid stop |
| **M_safe** | Fraction of collision-free episodes |
| **M_stop** | Fraction of episodes where ego fully stopped at the stop sign |
| **M_yield** | (SUMO only) Fraction of episodes where ego correctly respected FCFS priority |
| **avg_d_stop** | Mean distance-to-stop-line at the moment of registered full stop at the intersection (m) |

---

<!-- ## Citation -->

<!-- If you use this code, please cite:

```bibtex
@article{guo2025traffic,
  title   = {Traffic Rule Compliant Autonomous Driving Using Reward Machines and Reinforcement Learning},
  author  = {Guo, Steven and ...},
  year    = {2025}
}
``` -->
