"""
eval_multiseed_fcfs_lstmppo.py

Multi-seed evaluation for the SUMO FCFS all-way stop LSTM-PPO baseline.

Mirrors eval_multiseed_fcfs.py exactly, with three LSTM-specific changes:

1.  Policy        RecurrentPPO("MlpLstmPolicy") replaces PPO("MlpPolicy").
                  Architecture identical to AllWayStopEnvLSTM training:
                    net_arch=[256, 128], lstm_hidden_size=128, n_lstm_layers=1,
                    enable_critic_lstm=True, batch_size=64 (BPTT T=64 steps).

2.  Evaluate      lstm_state and episode_start are threaded through every
                  model.predict() call so the recurrent hidden state (h_t, c_t)
                  is preserved across steps within each episode, allowing the
                  policy to maintain its implicit belief about task phase
                  (has_stopped, FCFS priority).

3.  Environment   AllWayStopEnvLSTM (POMDP: raw kinematics only, no RM one-hot)
                  replaces AllWayStopEnvRM.  FCFS priority is still tracked
                  internally via env.fcfs_queue (hidden from the policy), so
                  M_yield can be evaluated identically to the RM version.
                  No enable_episode_logging attribute; logging is suppressed by
                  the module-level basicConfig(CRITICAL) at the top of this file.

Training protocol — identical to eval_multiseed_fcfs.py:
  - N_SEEDS      = 10 independent RecurrentPPO runs
  - TOTAL_STEPS  = 65536 steps per seed
  - EVAL_FREQ    = 4096 steps per checkpoint
  - EVAL_EPISODES = 100 episodes per checkpoint
  - Metrics: M_goal, M_safe, M_stop, M_yield, avg_d_stop
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

from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    raise ImportError("sb3_contrib required. Install with: pip install sb3-contrib")

from train_sumo_intersection_fcfs_lstmppo import (
    AllWayStopEnvLSTM,
    create_lstm_environment,
    DenseRecurrentFeatureExtractor,
)

# ── Suppress loggers ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
for _noisy in ('train_sumo_intersection_fcfs_lstmppo',
               'train_sumo_intersection_fcfs_rm',
               'stable_baselines3', 'sb3_contrib', 'gymnasium', 'traci', 'sumolib'):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# =============================================================================
# Hyper-parameters
# =============================================================================
N_SEEDS       = 10
TOTAL_STEPS   = 2048 * 60
EVAL_FREQ     = 4096
EVAL_EPISODES = 100
N_CHECKPOINTS = TOTAL_STEPS // EVAL_FREQ

# RecurrentPPO kwargs — matches train_sumo_intersection_fcfs_lstmppo.train_lstm() exactly.
#
# Architecture: obs(120) → Dense(256,ReLU) → Dense(128,ReLU)  [DenseRecurrentFeatureExtractor]
#                        → LSTM(128)                           [lstm_hidden_size=128]
#                        → actor / critic heads                [net_arch=[]]
#
# CRITICAL: in sb3_contrib, net_arch layers are placed AFTER the LSTM
# (MlpExtractor receives lstm_output_dim as its input).  net_arch=[256,128]
# would produce obs(120)→Flatten→LSTM(120→128)→Dense(Tanh)→Dense(Tanh)→heads,
# feeding raw 120-dim kinematics into the LSTM instead of 128-dim pre-processed
# features.  DenseRecurrentFeatureExtractor + net_arch=[] gives the correct
# pre-LSTM ordering that matches the paper specification and the single-seed file.
#
# batch_size=64 → BPTT truncation horizon T=64 steps.
_RPPO_KWARGS = dict(
    policy_kwargs = dict(
        features_extractor_class  = DenseRecurrentFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = [],          # no post-LSTM MLP; heads from h_t
        activation_fn             = nn.ReLU,    # matches hardcoded ReLU in extractor
        lstm_hidden_size          = 128,
        n_lstm_layers             = 1,
        enable_critic_lstm        = True,        # separate LSTM for V(h_t)
    ),
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,      # BPTT T=64 steps
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

# Output paths (distinct from RM multiseed to avoid overwriting)
_RESULTS_NPY   = os.path.join(_SCRIPT_DIR, "fcfs_lstmppo_multiseed_results.npy")
_TIMESTEPS_NPY = os.path.join(_SCRIPT_DIR, "fcfs_lstmppo_multiseed_timesteps.npy")
_PLOT_PDF      = os.path.join(_SCRIPT_DIR, "fcfs_lstmppo_multiseed_curves.pdf")
_CSV_TXT       = os.path.join(_SCRIPT_DIR, "fcfs_lstmppo_multiseed_results.txt")


# =============================================================================
# In-process evaluation
# =============================================================================
def _evaluate_inline(
    model:      RecurrentPPO,
    env:        AllWayStopEnvLSTM,
    n_episodes: int,
) -> tuple[float, float, float, float, float]:
    """
    Run `n_episodes` deterministic rollouts and return
    (M_goal, M_safe, M_stop, M_yield, avg_d_stop).

    LSTM-specific:
      · lstm_state = None is reset at the start of every episode.
      · ep_start = ones((1,), bool) signals the model to reset h_t at episode
        boundaries; it becomes array([done or trunc]) after each step.
      · model.predict(..., state=lstm_state, episode_start=ep_start) carries the
        hidden state (h_t, c_t) forward so the policy can distinguish
        "has_stopped=True" trajectories from "has_stopped=False" trajectories
        and can track FCFS priority history — both unobservable in a single frame.

    FCFS yield check — identical to eval_multiseed_fcfs.py:
      On the first step where ego crosses the stop line (dist_to_stop_line < 0),
      env.fcfs_queue.has_priority('ego') is queried.  env.fcfs_queue is the
      internal FCFSQueue object of AllWayStopEnvLSTM; it is updated every step
      by _update_fcfs_queue() and _update_ego_fcfs_queue() inside env.step().
    """
    n_goal = n_col = n_pass = n_yv = n_stopped = 0
    sum_d_stop = 0.0

    for _ in range(n_episodes):
        obs, _info   = env.reset()
        lstm_state   = None                         # reset LSTM hidden state
        ep_start     = np.ones((1,), dtype=bool)   # signal episode start
        done = trunc = False

        ep_col = ep_stopped = ep_passed = ep_goal = ep_entered = ep_yv = False
        # Distance-at-first-stop tracking
        stop_counter  = False   # latch: first full stop already recorded
        curr_d_stop = None

        while not (done or trunc):
            # Thread lstm_state and ep_start so h_t persists within the episode
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start, deterministic=True
            )
            obs, _reward, done, trunc, info = env.step(action)
            ep_start = np.array([done or trunc])

            # Physical state via env.st4obs(obs) — AllWayStopEnvLSTM exposes this
            # method with the obs argument (reads from the live SUMO/TraCI state).
            state = env.st4obs(obs)
            event = info.get('event', '')

            # 1. Collision — exclude connection_lost (ego exited network normally)
            if not ep_col and (event == 'x' or info.get('crashed', False)) \
                    and not info.get('connection_lost', False):
                ep_col = True

            # 2. Proper stop at stop line
            if not ep_stopped and event == 's':
                ep_stopped  = True
                # curr_d_stop = state['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
                cor_d_stop = state['dist_to_stop_line']

            if abs(state['v_e']) < 0.1 and not stop_counter:
                curr_d_stop = state['dist_to_stop_line'] # Stopping precision for ordinary stops
                stop_counter = True

            # 3. Passed stop line without stopping
            if not ep_passed and event == 'p':
                ep_passed = True

            # 4. Yield violation — query FCFS queue once on first crossing
            if not ep_entered and state['dist_to_stop_line'] < 0:
                ep_entered = True
                if not env.fcfs_queue.has_priority('ego'):
                    ep_yv = True

            # 5. Goal
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
        if ep_yv:
            n_yv += 1
        if ep_goal and ep_stopped and not ep_yv and not ep_col:
            n_goal += 1

    N          = n_episodes
    M_goal     = 100.0 * n_goal    / N
    M_safe     = 100.0 * (1 - n_col  / N)
    M_stop     = 100.0 * n_stopped / N
    M_yield    = 100.0 * (1 - n_yv   / N)
    avg_d_stop = sum_d_stop / n_stopped if n_stopped > 0 else float('nan') # Strict avg_ds
    # avg_d_stop = sum_d_stop / (N-n_pass) if n_stopped > 0 else float('nan') # Relaxed avg_ds

    return M_goal, M_safe, M_stop, M_yield, avg_d_stop


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
    For each seed: train RecurrentPPO for `total_steps` on a dedicated SUMO
    training env ('lstm_train' TraCI label), evaluating every `eval_freq` steps
    on a separate SUMO eval env ('lstm_eval' TraCI label).

    The two SUMO processes coexist in the same Python process via TraCI's
    labeled-connection API.  Only one is active at any time (single-threaded),
    so there is no concurrent-access conflict.

    Returns
    -------
    timesteps   : ndarray (n_checkpoints,)
    all_results : ndarray (len(seeds), n_checkpoints, 5)
                  axis-2: [M_goal, M_safe, M_stop, M_yield, avg_d_stop]
    """
    seeds  = seed_indices if seed_indices is not None else list(range(n_seeds))
    n_ckpt = total_steps // eval_freq
    timesteps   = np.arange(1, n_ckpt + 1) * eval_freq
    all_results = np.full((len(seeds), n_ckpt, 5), np.nan, dtype=np.float64)

    _csv_init(csv_path)
    wall_start = time.time()

    for i, s in enumerate(seeds):
        seed_start = time.time()
        print(f"\n{'='*64}")
        print(f"  Seed {i + 1}/{len(seeds)}   (global seed = {s})  [LSTM-PPO FCFS]")
        print(f"{'='*64}")

        train_env = create_lstm_environment(
            use_sumo=True, traci_label='lstm_train'
        )
        vec_env = DummyVecEnv([lambda e=train_env: e])
        model   = RecurrentPPO("MlpLstmPolicy", vec_env, seed=s, **_RPPO_KWARGS)

        eval_env = create_lstm_environment(
            use_sumo=True, traci_label='lstm_eval'
        )

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

            mg, ms, mst, my, ds = metrics
            pbar.set_postfix({
                'M_goal':  f"{mg:.1f}%",
                'M_safe':  f"{ms:.1f}%",
                'M_stop':  f"{mst:.1f}%",
                'M_yield': f"{my:.1f}%",
                'avg_ds':  f"{ds:.2f}m" if not np.isnan(ds) else "nan",
            })
            pbar.update(1)

        pbar.close()

        partial_path = os.path.join(_SCRIPT_DIR,
                                    f"fcfs_lstmppo_multiseed_seed{s:02d}.npy")
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
# Plotting  (identical format to eval_multiseed_fcfs.py)
# =============================================================================
_IEEE_W = 7.16
_IEEE_H = 2.5


def plot_results(
    timesteps:   np.ndarray,
    all_results: np.ndarray,
    output_file: str  = _PLOT_PDF,
    log_xaxis:   bool = True,
) -> None:
    """IEEE-formatted 1×2 convergence plot (identical format to eval_multiseed_fcfs.py)."""
    rc('text', usetex=True)
    rc('font', family='serif', size=12)

    base, ext = os.path.splitext(output_file)
    output_file = f"{base}_{'log' if log_xaxis else 'linear'}{ext}"

    learn_itrs_k = timesteps / 1000.0
    x_min_k = learn_itrs_k[0]
    x_max_k = learn_itrs_k[-1]

    colors = {
        'M_goal':  '#0072B2',
        'M_safe':  '#009E73',
        'M_stop':  '#D55E00',
        'M_yield': '#56B4E9',
        'D_stop':  '#CC79A7',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_IEEE_W, _IEEE_H))

    for metric, m_idx in [('M_goal', 0), ('M_safe', 1), ('M_stop', 2), ('M_yield', 3)]:
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
        ax1.set_xlim(x_min_k, x_max_k * 1.05)
    else:
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.set_xlim(0, x_max_k * 1.05)
    ax1.set_xlabel(r'Training Timesteps ($\times 10^3$)')
    ax1.set_ylabel(r'(\%)')
    ax1.legend(loc='lower right', bbox_to_anchor=(1.28, 0.0), framealpha=0.9, fontsize=10)
    ax1.set_title(r'(a) Task Completion Metrics')

    data_d  = all_results[:, :, 4]
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
_CSV_HEADER = 'seed,learn_itrs,M_goal,M_safe,M_stop,M_yield,D_stop'


def _csv_init(path: str) -> None:
    with open(path, 'w') as f:
        f.write(_CSV_HEADER + '\n')
    print(f"CSV initialised -> {path}")


def _csv_append(path: str, seed: int, learn_itrs: int,
                metrics: tuple[float, ...]) -> None:
    mg, ms, mst, my, ds = metrics
    ds_str = f"{ds:.4f}" if not np.isnan(ds) else "nan"
    row = f"{seed},{learn_itrs},{mg:.4f},{ms:.4f},{mst:.4f},{my:.4f},{ds_str}"
    with open(path, 'a') as f:
        f.write(row + '\n')


def _load_from_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (timesteps, all_results) from the incremental CSV file."""
    rows: list[tuple[int, int, float, float, float, float, float]] = []
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
            my   = float(parts[5])
            ds   = float('nan') if parts[6].strip().lower() == 'nan' \
                   else float(parts[6])
            rows.append((seed_val, learn_itrs, mg, ms, mst, my, ds))
    if not rows:
        raise ValueError(f"No data rows found in {path!r}")
    seeds_sorted = sorted(set(r[0] for r in rows))
    ckpts_sorted = sorted(set(r[1] for r in rows))
    seed_to_idx  = {s: i for i, s in enumerate(seeds_sorted)}
    ckpt_to_idx  = {c: j for j, c in enumerate(ckpts_sorted)}
    timesteps    = np.array(ckpts_sorted, dtype=np.int64)
    all_results  = np.full((len(seeds_sorted), len(ckpts_sorted), 5),
                           np.nan, dtype=np.float64)
    for seed_val, learn_itrs, mg, ms, mst, my, ds in rows:
        all_results[seed_to_idx[seed_val], ckpt_to_idx[learn_itrs], :] = \
            [mg, ms, mst, my, ds]
    print(f"Loaded from CSV: {len(seeds_sorted)} seed(s), {len(ckpts_sorted)} "
          f"checkpoint(s)  (seeds present: {seeds_sorted})")
    return timesteps, all_results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-seed RecurrentPPO (LSTM) training + eval — SUMO FCFS"
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
