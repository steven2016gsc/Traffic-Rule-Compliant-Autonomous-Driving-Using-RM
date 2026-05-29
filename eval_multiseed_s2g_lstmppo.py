"""
eval_multiseed_s2g_lstmppo.py

Multi-seed evaluation for the highway-env Stop-and-Go LSTM-PPO baseline.

Mirrors eval_multiseed_s2g.py exactly, with three LSTM-specific changes:

1.  Policy        RecurrentPPO("MlpLstmPolicy") replaces PPO("MlpPolicy").
                  Architecture identical to HighwayStopEnvLSTM training:
                    DenseRecurrentFeatureExtractor: obs→Dense(256,ReLU)→Dense(128,ReLU)
                    lstm_hidden_size=128, n_lstm_layers=1, net_arch=[] (pre-LSTM),
                    enable_critic_lstm=True, batch_size=64 (BPTT T=64 steps).

2.  Evaluate      lstm_state and episode_start are threaded through every
                  model.predict() call so the recurrent hidden state (h_t, c_t)
                  is preserved across steps within each episode, allowing the
                  policy to maintain its implicit belief about task phase.

3.  Environment   HighwayStopEnvLSTM (POMDP: raw kinematics only, no RM
                  one-hot) replaces HighwayStopEnvRM.  It does not expose
                  enable_episode_logging; that attribute is simply not set.
                  Physical state is read via env._st4obs() (no obs argument).

Training protocol — identical to eval_multiseed_s2g.py:
  - N_SEEDS       = 10 independent RecurrentPPO runs, each with a different seed
  - TOTAL_STEPS   = 122880 steps per seed  (= 2048 × 60)
  - EVAL_FREQ     = 4096 steps per checkpoint (= 2 × PPO n_steps)
  - N_CHECKPOINTS = 30  (= 122880 // 4096)
  - EVAL_EPISODES = 100 episodes per checkpoint
  - Metrics: M_goal, M_safe, M_stop, avg_d_stop  (no M_yield — no FCFS queue)
"""

from __future__ import annotations

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'HighwayEnv'))

import gc
import logging
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc

import gymnasium as gym
import highway_env  # noqa: F401  — registers highway-v0

from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    raise ImportError("sb3_contrib required. Install with: pip install sb3-contrib")

from train_hwyenv_stopandgo_lstmppo_single import (
    HighwayStopEnvLSTM,
    DenseRecurrentFeatureExtractor,
)

# ── Suppress loggers ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
for _noisy in ('train_hwyenv_stopandgo_lstmppo_single',
               'stable_baselines3', 'sb3_contrib', 'gymnasium', 'highway_env'):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# =============================================================================
# Hyper-parameters
# =============================================================================
N_SEEDS       = 10
TOTAL_STEPS   = 2048 * 60          # 122880 — same as RM multiseed (2048*60)
EVAL_FREQ     = 4096
EVAL_EPISODES = 100
N_CHECKPOINTS = TOTAL_STEPS // EVAL_FREQ

# RecurrentPPO kwargs — matches train_hwyenv_stopandgo_lstmppo_single.py exactly.
#
# Architecture: obs → Dense(256,ReLU) → Dense(128,ReLU)  [DenseRecurrentFeatureExtractor]
#                   → LSTM(128)                           [lstm_hidden_size=128]
#                   → actor / critic heads                [net_arch=[]]
#
# CRITICAL: in sb3_contrib, net_arch layers are inserted AFTER the LSTM
# (MlpExtractor receives lstm_output_dim as its input).  Using net_arch=[256,128]
# would place the Dense layers post-LSTM and feed raw 21-dim obs into the LSTM —
# exactly the opposite of the intended pre-LSTM feature-extraction design.
# The correct approach is DenseRecurrentFeatureExtractor (pre-LSTM) + net_arch=[].
#
# batch_size=64 → BPTT truncation horizon T=64 policy steps ≈ 12.8 s at 5 Hz.
_RPPO_KWARGS = dict(
    policy_kwargs = dict(
        features_extractor_class  = DenseRecurrentFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = [],          # no post-LSTM MLP; heads from h_t directly
        activation_fn             = nn.ReLU,    # matches hardcoded ReLU in extractor
        lstm_hidden_size          = 128,
        n_lstm_layers             = 1,
        enable_critic_lstm        = True,        # separate LSTM for V(h_t)
    ),
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,      # BPTT T=64 steps ≈ 12.8 s at 5 Hz
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 0,
    device        = "cpu",
)

# highway-v0 config — exact copy from train_hwyenv_stopandgo_lstmppo_single.py
_HWY_CONFIG = {
    "lanes_count":        1,
    "vehicles_count":     2,
    "controlled_vehicles": 1,
    "initial_speed":      20,
    "initial_spacing":    1,
    "duration":           20,
    "policy_frequency":   5,
    "simulation_frequency": 15,
    "observation": {
        "type":           "Kinematics",
        "vehicles_count": 5,
        "features":       ["x", "y", "vx", "vy"],
        "normalize":      True,
        "absolute":       True,
    },
    "action": {
        "type":          "DiscreteMetaAction",
        "longitudinal":  True,
        "lateral":       False,
        "target_speeds": [0, 15, 30],
    },
    "reward_speed_range": [0, 30],
    "collision_reward":   -1,
}

# Output paths (distinct from RM multiseed to avoid overwriting)
_RESULTS_NPY   = os.path.join(_SCRIPT_DIR, "s2g_lstmppo_multiseed_results.npy")
_TIMESTEPS_NPY = os.path.join(_SCRIPT_DIR, "s2g_lstmppo_multiseed_timesteps.npy")
_PLOT_PDF      = os.path.join(_SCRIPT_DIR, "s2g_lstmppo_multiseed_curves.pdf")
_CSV_TXT       = os.path.join(_SCRIPT_DIR, "s2g_lstmppo_multiseed_results.txt")


# =============================================================================
# Environment factory
# =============================================================================
def create_lstm_hwy_environment() -> HighwayStopEnvLSTM:
    """Create a fresh highway-v0 + HighwayStopEnvLSTM instance."""
    base_env = gym.make("highway-v0", render_mode=None, config=_HWY_CONFIG)
    base_env.reset()
    return HighwayStopEnvLSTM(base_env)


# =============================================================================
# In-process evaluation
# =============================================================================
def _evaluate_inline(
    model:      RecurrentPPO,
    env:        HighwayStopEnvLSTM,
    n_episodes: int,
) -> tuple[float, float, float, float]:
    """
    Run `n_episodes` deterministic rollouts and return (M_goal, M_safe, M_stop,
    avg_d_stop).

    LSTM-specific:
      · lstm_state = None is reset at the start of every episode.
      · ep_start = ones((1,), bool) signals the model to reset h_t at episode
        boundaries; it is set to array([done or trunc]) after each step.
      · model.predict(obs, state=lstm_state, episode_start=ep_start, ...) carries
        the hidden state (h_t, c_t) forward across the full episode, so the policy
        can distinguish "has_stopped=True" from "has_stopped=False" trajectories.
    """
    n_goal = n_col = n_pass = n_stopped = 0
    sum_d_stop = 0.0

    for _ in range(n_episodes):
        obs, _info   = env.reset()
        lstm_state   = None                         # reset LSTM hidden state
        ep_start     = np.ones((1,), dtype=bool)   # signal episode start
        done = trunc = False

        ep_col = ep_stopped = ep_passed = ep_goal = False
        # Distance-at-first-stop tracking
        stop_counter  = False   # latch: first full stop already recorded
        curr_d_stop = None
        cor_d_stop = None

        while not (done or trunc):
            # Thread lstm_state and ep_start so h_t persists within the episode
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start, deterministic=True
            )
            obs, _reward, done, trunc, info = env.step(action)
            ep_start = np.array([done or trunc])

            # Use info['event'] set by HighwayStopEnvLSTM.step().
            # Physical state via env._st4obs() (no obs argument — reads directly
            # from the simulator; same pattern as AllWayStopEnvLSTM.st4obs).
            state = env._st4obs()
            event = info.get('event', '')

            if not ep_col and (event == 'x' or info.get('crashed', False)):
                ep_col = True
            if not ep_stopped and event == 's':
                ep_stopped  = True
                #curr_d_stop = state['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
                cor_d_stop = state['dist_to_stop_line']

            if abs(state['v_e']) < 0.1 and not stop_counter:
                curr_d_stop = state['dist_to_stop_line'] # Stopping precision for ordinary stops
                stop_counter = True

            if not ep_passed and event == 'p':
                ep_passed = True
            if not ep_goal and event == 'g':
                ep_goal = True

        if ep_col:
            n_col += 1
        if ep_stopped:
            n_stopped  += 1
            sum_d_stop += cor_d_stop
        # if curr_d_stop is not None:
        #     sum_d_stop += curr_d_stop
        if ep_passed and not ep_stopped:
            n_pass += 1
        if ep_goal and ep_stopped and not ep_col:
            n_goal += 1

    N          = n_episodes
    M_goal     = 100.0 * n_goal    / N
    M_safe     = 100.0 * (1 - n_col  / N)
    M_stop     = 100.0 * n_stopped / N
    avg_d_stop = sum_d_stop / n_stopped if n_stopped > 0 else float('nan') # Strict avg_ds
    # avg_d_stop = sum_d_stop / (N-n_pass) if n_stopped > 0 else float('nan') # Relaxed avg_ds
    return M_goal, M_safe, M_stop, avg_d_stop


# =============================================================================
# Main training + evaluation loop
# =============================================================================
def run_multi_seed(
    n_seeds:      int              = N_SEEDS,
    total_steps:  int              = TOTAL_STEPS,
    eval_freq:    int              = EVAL_FREQ,
    n_eval_ep:    int              = EVAL_EPISODES,
    csv_path:     str              = _CSV_TXT,
    seed_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each seed: train RecurrentPPO for `total_steps` on a dedicated training
    env, evaluating every `eval_freq` steps on a separate eval env.

    Returns
    -------
    timesteps   : ndarray (n_checkpoints,)
    all_results : ndarray (len(seeds), n_checkpoints, 4)
                  axis-2: [M_goal, M_safe, M_stop, avg_d_stop]
    """
    seeds  = seed_indices if seed_indices is not None else list(range(n_seeds))
    n_ckpt = total_steps // eval_freq
    timesteps   = np.arange(1, n_ckpt + 1) * eval_freq
    all_results = np.full((len(seeds), n_ckpt, 4), np.nan, dtype=np.float64)

    _csv_init(csv_path)
    wall_start = time.time()

    for i, s in enumerate(seeds):
        seed_start = time.time()
        print(f"\n{'='*64}")
        print(f"  Seed {i + 1}/{len(seeds)}   (global seed = {s})  [LSTM-PPO S2G]")
        print(f"{'='*64}")

        train_env = create_lstm_hwy_environment()
        vec_env   = DummyVecEnv([lambda e=train_env: e])
        model     = RecurrentPPO("MlpLstmPolicy", vec_env, seed=s, **_RPPO_KWARGS)

        eval_env  = create_lstm_hwy_environment()

        pbar = tqdm(
            total=n_ckpt,
            desc=f"Seed {s:2d} ({i + 1}/{len(seeds)})",
            unit="ckpt",
            dynamic_ncols=True,
            leave=True,
        )

        for ckpt in range(n_ckpt):
            model.learn(
                total_timesteps     = eval_freq,
                reset_num_timesteps = (ckpt == 0),
                progress_bar        = False,
            )

            metrics = _evaluate_inline(model, eval_env, n_eval_ep)
            all_results[i, ckpt] = metrics
            _csv_append(csv_path, s, int(timesteps[ckpt]), metrics)

            mg, ms, mst, ds = metrics
            pbar.set_postfix({
                'M_goal': f"{mg:.1f}%",
                'M_safe': f"{ms:.1f}%",
                'M_stop': f"{mst:.1f}%",
                'avg_ds': f"{ds:.2f}m" if not np.isnan(ds) else "nan",
            })
            pbar.update(1)

        pbar.close()

        partial_path = os.path.join(_SCRIPT_DIR, f"s2g_lstmppo_multiseed_seed{s:02d}.npy")
        np.save(partial_path, all_results[i])

        seed_elapsed = time.time() - seed_start
        print(f"  Seed {s} done in {seed_elapsed / 60:.1f} min  -> {partial_path}")

        eval_env.close()
        vec_env.close()
        del model, vec_env, train_env, eval_env
        gc.collect()

    wall_elapsed = time.time() - wall_start
    print(f"\nAll {len(seeds)} seeds done in {wall_elapsed / 60:.1f} min total.")
    return timesteps, all_results


# =============================================================================
# Plotting  (identical to eval_multiseed_s2g.py)
# =============================================================================
_IEEE_W = 7.16
_IEEE_H = 2.5


def plot_results(
    timesteps:   np.ndarray,
    all_results: np.ndarray,
    output_file: str  = _PLOT_PDF,
    log_xaxis:   bool = True,
) -> None:
    """IEEE-formatted 1×2 convergence plot (identical format to eval_multiseed_s2g.py)."""
    rc('text', usetex=True)
    rc('font', family='serif', size=12)

    base, ext = os.path.splitext(output_file)
    output_file = f"{base}_{'log' if log_xaxis else 'linear'}{ext}"

    learn_itrs_k = timesteps / 1000.0
    _mstop_mean  = np.nanmean(all_results[:, :, 2], axis=0)
    _nonzero_idx = np.where(_mstop_mean > 0)[0]
    x_min_k = learn_itrs_k[max(0, _nonzero_idx[0] - 1)] if len(_nonzero_idx) > 0 \
              else learn_itrs_k[0]
    x_max_k = learn_itrs_k[-1]

    colors = {
        'M_goal': '#0072B2',
        'M_safe': '#009E73',
        'M_stop': '#D55E00',
        'D_stop': '#CC79A7',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_IEEE_W, _IEEE_H))

    for metric, m_idx in [('M_goal', 0), ('M_safe', 1), ('M_stop', 2)]:
        data  = all_results[:, :, m_idx]
        mean  = np.nanmean(data, axis=0)
        valid = np.sum(~np.isnan(data), axis=0)
        ci    = np.where(valid > 1,
                         2 * np.nanstd(data, axis=0, ddof=1) / np.sqrt(valid),
                         np.nan)
        ax1.fill_between(learn_itrs_k,
                         np.maximum(mean - ci, 0), np.minimum(mean + ci, 100),
                         color=colors[metric], alpha=0.2, linewidth=0)
        ax1.plot(learn_itrs_k, mean,
                 label=r'$M_{\mathrm{' + metric.split('_')[1] + r'}}$',
                 color=colors[metric], linewidth=1.5)

    if log_xaxis:
        ax1.set_xscale('log')
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, which='both')
        ax1.set_xlim(learn_itrs_k[0], x_max_k * 1.05)
    else:
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.set_xlim(0, x_max_k * 1.05)
    ax1.set_xlabel(r'Training Timesteps ($\times 10^3$)')
    ax1.set_ylabel(r'(\%)')
    ax1.legend(loc='best', framealpha=0.9, fontsize=10)
    ax1.set_title(r'(a) Task Completion Metrics')

    data_d  = all_results[:, :, 3]
    mean_d  = np.nanmean(data_d, axis=0)
    valid_d = np.sum(~np.isnan(data_d), axis=0)
    ci_d    = np.where(valid_d > 1,
                       2 * np.nanstd(data_d, axis=0, ddof=1) / np.sqrt(valid_d),
                       np.nan)
    ax2.fill_between(learn_itrs_k,
                     np.maximum(mean_d - ci_d, 0), mean_d + ci_d,
                     color=colors['D_stop'], alpha=0.2, linewidth=0)
    ax2.plot(learn_itrs_k, mean_d,
             label=r'$D_{\mathrm{stop}}$', color=colors['D_stop'], linewidth=1.5)

    if log_xaxis:
        ax2.set_xscale('log')
        ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, which='both')
        ax2.set_xlim(x_min_k, x_max_k * 1.05)
    else:
        ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        ax2.set_xlim(0, x_max_k * 1.05)
    ax2.set_xlabel(r'Training Timesteps ($\times 10^3$)')
    ax2.set_ylabel(r'Stopping Precision (m)')
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=10)
    ax2.set_title(r'(b) Stopping Precision')

    upper_d = np.nanmax(mean_d + ci_d) if not np.all(np.isnan(mean_d)) else 10.0
    ax2.set_ylim(bottom=ax1.get_ylim()[0], top=upper_d * 1.1)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Figure saved -> {output_file}")
    png_file = output_file.replace('.pdf', '.png')
    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"PNG preview saved -> {png_file}")
    plt.close(fig)


# =============================================================================
# Incremental CSV helpers
# =============================================================================
_CSV_HEADER = 'seed,learn_itrs,M_goal,M_safe,M_stop,D_stop'


def _csv_init(path: str) -> None:
    with open(path, 'w') as f:
        f.write(_CSV_HEADER + '\n')
    print(f"CSV initialised -> {path}")


def _csv_append(path: str, seed: int, learn_itrs: int,
                metrics: tuple[float, ...]) -> None:
    mg, ms, mst, ds = metrics
    ds_str = f"{ds:.4f}" if not np.isnan(ds) else "nan"
    row = f"{seed},{learn_itrs},{mg:.4f},{ms:.4f},{mst:.4f},{ds_str}"
    with open(path, 'a') as f:
        f.write(row + '\n')


def _load_from_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (timesteps, all_results) from the incremental CSV file."""
    rows: list[tuple[int, int, float, float, float, float]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('seed'):
                continue
            parts = line.split(',')
            seed_val   = int(parts[0])
            learn_itrs = int(parts[1])
            mg   = float(parts[2])
            ms   = float(parts[3])
            mst  = float(parts[4])
            ds   = float('nan') if parts[5].strip().lower() == 'nan' \
                   else float(parts[5])
            rows.append((seed_val, learn_itrs, mg, ms, mst, ds))
    if not rows:
        raise ValueError(f"No data rows found in {path!r}")
    seeds_sorted = sorted(set(r[0] for r in rows))
    ckpts_sorted = sorted(set(r[1] for r in rows))
    seed_to_idx  = {s: i for i, s in enumerate(seeds_sorted)}
    ckpt_to_idx  = {c: j for j, c in enumerate(ckpts_sorted)}
    timesteps    = np.array(ckpts_sorted, dtype=np.int64)
    all_results  = np.full((len(seeds_sorted), len(ckpts_sorted), 4),
                           np.nan, dtype=np.float64)
    for seed_val, learn_itrs, mg, ms, mst, ds in rows:
        all_results[seed_to_idx[seed_val], ckpt_to_idx[learn_itrs], :] = \
            [mg, ms, mst, ds]
    print(f"Loaded from CSV: {len(seeds_sorted)} seed(s), {len(ckpts_sorted)} "
          f"checkpoint(s)  (seeds present: {seeds_sorted})")
    return timesteps, all_results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-seed RecurrentPPO (LSTM) training + eval — highway-env S2G"
    )

    def _parse_seed_list(s: str) -> list[int]:
        try:
            return [int(x.strip()) for x in s.strip().strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--seed-list: cannot parse {s!r}. Expected: '[1,3,5]' or '1,3,5'."
            )

    parser.add_argument('--seeds',      type=int, default=N_SEEDS)
    parser.add_argument('--seed',       type=int, default=None)
    parser.add_argument('--seed-start', type=int, default=None)
    parser.add_argument('--seed-end',   type=int, default=None)
    parser.add_argument('--seed-list',  type=_parse_seed_list, default=None)
    parser.add_argument('--steps',      type=int, default=TOTAL_STEPS)
    parser.add_argument('--eval-freq',  type=int, default=EVAL_FREQ)
    parser.add_argument('--eval-eps',   type=int, default=EVAL_EPISODES)
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plot-only',          action='store_true')
    plot_group.add_argument('--plot-only-with-txt', action='store_true')
    args = parser.parse_args()

    _range_given = (args.seed_start is not None or args.seed_end is not None)
    _list_given  = args.seed_list is not None
    if args.seed is not None and (_range_given or _list_given):
        parser.error("--seed cannot be combined with --seed-start/--seed-end or --seed-list.")
    if _range_given and _list_given:
        parser.error("--seed-list cannot be combined with --seed-start/--seed-end.")
    if (args.seed_start is None) != (args.seed_end is None):
        parser.error("--seed-start and --seed-end must be used together.")
    if args.seed_start is not None and args.seed_start > args.seed_end:
        parser.error("--seed-start must be <= --seed-end.")

    if args.seed is not None:
        seed_indices = [args.seed]
    elif _range_given:
        seed_indices = list(range(args.seed_start, args.seed_end + 1))
    elif _list_given:
        seed_indices = args.seed_list
    else:
        meta_rng = np.random.RandomState(0)          # fixed meta-seed
        seed_indices = sorted(meta_rng.randint(0, 10000, N_SEEDS).tolist()) # Pineau et al. 2021
        # seed_indices = None

    if args.plot_only:
        if not (os.path.exists(_RESULTS_NPY) and os.path.exists(_TIMESTEPS_NPY)):
            parser.error("--plot-only requires the .npy files to exist.")
        timesteps   = np.load(_TIMESTEPS_NPY)
        all_results = np.load(_RESULTS_NPY)
        print(f"Loaded: {all_results.shape}")
    elif args.plot_only_with_txt:
        if not os.path.exists(_CSV_TXT):
            parser.error(f"--plot-only-with-txt requires {_CSV_TXT} to exist.")
        timesteps, all_results = _load_from_csv(_CSV_TXT)
    else:
        timesteps, all_results = run_multi_seed(
            n_seeds      = args.seeds,
            total_steps  = args.steps,
            eval_freq    = args.eval_freq,
            n_eval_ep    = args.eval_eps,
            csv_path     = _CSV_TXT,
            seed_indices = seed_indices,
        )
        np.save(_RESULTS_NPY,   all_results)
        np.save(_TIMESTEPS_NPY, timesteps)
        print(f"\nRaw arrays saved:\n  {_RESULTS_NPY}\n  {_TIMESTEPS_NPY}")
        print(f"Incremental CSV: {_CSV_TXT}")

    plot_results(timesteps, all_results)
