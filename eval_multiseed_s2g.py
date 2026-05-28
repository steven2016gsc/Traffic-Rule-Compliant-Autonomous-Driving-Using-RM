"""
eval_multiseed_s2g.py

Multi-seed evaluation for the highway-env Stop-and-Go RL task.

Training protocol (Icarte et al. 2018 / Figure 1 style):
  - N_SEEDS     = 10 independent PPO runs, each with a different random seed
  - TOTAL_STEPS = 102400 steps per seed  (= 2048 * 50, matching learn_itrs in
                  train_hwyenv_stopandgo_rm_single_sas_rss_28.py)
  - One deterministic evaluation rollout every EVAL_FREQ = 2048 environment
    steps (= one complete PPO rollout buffer), using EVAL_EPISODES = 100
    episodes.
  - Four metrics: M_goal, M_safe, M_stop, avg_d_stop
    (no M_yield — highway-env has no explicit FCFS priority queue)

Design notes:
  - Training is strictly sequential (one seed at a time).
  - Each seed creates TWO highway-env instances: one for training (wrapped in
    DummyVecEnv for PPO) and one for evaluation.  Unlike SUMO, highway-env is
    a pure-Python simulator with no TraCI labels or external processes — both
    envs coexist as ordinary Python objects in the same process.
  - The training env is never reset or interrupted between checkpoints;
    model._last_obs (held in the DummyVecEnv) remains valid across eval calls.
  - device="cpu" — small MLP policies are faster on CPU than MPS on Apple
    Silicon due to kernel-launch overhead.
  - Intermediate numpy results are saved after each completed seed so a crash
    does not lose previously finished work.
  - A CSV file is created with its header before training begins; one row is
    appended and flushed after every checkpoint evaluation (crash-safe).
  - No intermediate model (.zip) files are saved during training.
"""

from __future__ import annotations

import sys
import os

# ── Path: mirror what the training script does ────────────────────────────────
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the wrapper and RM from the training module.
# The module-level `if __name__ == "__main__":` block is guarded, so importing
# is safe and does not trigger training or logging setup.
from train_hwyenv_stopandgo_rm_single_sas_rss_28 import HighwayStopEnvRM
from reward_machine.reward_machine import RewardMachine

# ── Suppress all loggers so eval I/O stays clean ──────────────────────────────
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
for _noisy in ('train_hwyenv_stopandgo_rm_single_sas_rss_28',
               'stable_baselines3', 'gymnasium', 'highway_env'):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# =============================================================================
# Hyper-parameters
# =============================================================================
N_SEEDS       = 10
TOTAL_STEPS   = 2048 * 60          # 102400 — matches learn_itrs in training script
EVAL_FREQ     = 4096                # evaluate every N env steps (= PPO n_steps)
EVAL_EPISODES = 100                 # eval episodes per checkpoint
N_CHECKPOINTS = TOTAL_STEPS // EVAL_FREQ   # 50

# PPO kwargs — matches train script exactly; verbose=0 suppresses SB3 tables.
_PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 512,
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

# highway-v0 config — exact copy from train_hwyenv_stopandgo_rm_single_sas_rss_28.py
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

# Output paths
_RESULTS_NPY   = os.path.join(_SCRIPT_DIR, "s2g_multiseed_results.npy")
_TIMESTEPS_NPY = os.path.join(_SCRIPT_DIR, "s2g_multiseed_timesteps.npy")
_PLOT_PDF      = os.path.join(_SCRIPT_DIR, "s2g_multiseed_curves.pdf")
_CSV_TXT       = os.path.join(_SCRIPT_DIR, "s2g_multiseed_results.txt")


# =============================================================================
# Environment factory
# =============================================================================
def create_hwy_environment() -> HighwayStopEnvRM:
    """
    Create a fresh highway-v0 + HighwayStopEnvRM instance.

    Mirrors the environment setup in the training script's __main__ block:
    the base gym env is reset once before wrapping so that the observation
    space is finalised, then HighwayStopEnvRM wraps it (which resets again
    internally to measure the actual obs shape).
    """
    base_env = gym.make("highway-v0", render_mode=None, config=_HWY_CONFIG)
    base_env.reset()   # apply obs config before wrapping (see training script note)
    rm = RewardMachine(os.path.join(_SCRIPT_DIR, "rm_examples/stop_fcfs_rss_3.txt"))
    return HighwayStopEnvRM(base_env, rm)


# =============================================================================
# In-process evaluation
# =============================================================================
def _evaluate_inline(
    model:      PPO,
    env:        HighwayStopEnvRM,
    n_episodes: int,
) -> tuple[float, float, float, float]:
    """
    Run `n_episodes` deterministic rollouts and compute evaluation metrics.

    Mirrors the testing loop in train_hwyenv_stopandgo_rm_single_sas_rss_28.py
    but uses info['event'] (set by HighwayStopEnvRM.step()) instead of calling
    L() a second time, which would corrupt ego_stopped_time and miss 's'.

    Returns
    -------
    (M_goal, M_safe, M_stop, avg_d_stop)
    Rate metrics are in [0, 100] (percent).
    avg_d_stop is in metres; NaN when no episode recorded a valid stop.
    """
    # Suppress per-step file logging for the duration of this eval block.
    env.enable_episode_logging = False

    n_goal = n_col = n_pass = n_stopped = 0
    sum_d_stop = 0.0

    for _ in range(n_episodes):
        obs, _info = env.reset()
        done = trunc = False

        # Episode-level latches
        ep_col = ep_stopped = ep_passed = ep_goal = False
        # Distance-at-first-stop tracking
        stop_counter  = False   # latch: first full stop already recorded
        curr_d_stop = None   # None = no valid stop recorded this episode
        cor_d_stop = None

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, trunc, info = env.step(action)

            # Use info['event'] set by HighwayStopEnvRM.step() — do NOT call
            # L() again here.  A second call would see stale ego_stopped_time
            # and prev_dist_s, corrupting 's' and dropping 'p'.
            # st4obs() is still called for the dist_to_stop_line field.
            state = env.st4obs(obs)
            event = info.get('event', '')

            # 1. Collision (M_safe) — latch for the episode.
            # highway-env removes vehicles on collision (no --collision.action
            # warn), so info['crashed'] is a reliable collision indicator with
            # no connection_lost ambiguity.
            if not ep_col and (event == 'x' or info.get('crashed', False)):
                ep_col = True

            # 2. Proper stop at stop line (M_stop numerator).
            # 's' fires the first time ego is in the stop zone and is_stopped.
            if not ep_stopped and event == 's':
                ep_stopped  = True
                cor_d_stop = state['dist_to_stop_line'] # Stopping precision for correct stops at the stop sign
            
            if abs(state['v_e']) < 0.1 and not stop_counter:
                curr_d_stop = state['dist_to_stop_line'] # Stopping precision for ordinary stops
                stop_counter = True

            # 3. Ran stop sign without stopping first.
            if not ep_passed and event == 'p':
                ep_passed = True

            # 4. Goal reached — ego has cleared 50 m past the stop line.
            if not ep_goal and event == 'g':
                ep_goal = True

        # Accumulate episode counters
        if ep_col:
            n_col += 1
        if ep_stopped:
            n_stopped += 1
            sum_d_stop += cor_d_stop
        # if curr_d_stop is not None:
        #     sum_d_stop += curr_d_stop
        if ep_passed and not ep_stopped:
            n_pass += 1
        # Goal: must have stopped AND reached goal AND no collision.
        # (No FCFS yield-violation check — highway-env has no priority queue.)
        if ep_goal and ep_stopped and not ep_col:
            n_goal += 1

    N = n_episodes
    M_goal     = 100.0 * n_goal    / N
    M_safe     = 100.0 * (1 - n_col  / N)
    # M_stop = fraction of episodes where ego *actually stopped* at the stop
    # line (event 's' fired).  Using (1 - n_pass/N) was wrong: an agent that
    # never reaches the stop zone has n_pass=0 → M_stop=100%, while
    # n_stopped=0 → D_stop=nan.  Using n_stopped/N makes both metrics
    # consistent: M_stop=0% ↔ D_stop=nan when the stop zone is never reached.
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
    For each seed: train PPO for `total_steps` on a dedicated training env,
    evaluating every `eval_freq` steps on a separate eval env.  The training
    env is never reset or interrupted between checkpoints.

    Parameters
    ----------
    seed_indices : list of seed values to run, or None to run 0..n_seeds-1.
        Use a single-element list (e.g. [3]) for a targeted single-seed trial.

    Returns
    -------
    timesteps   : ndarray (n_checkpoints,)
    all_results : ndarray (len(seeds), n_checkpoints, 4)
                  axis-2 order: [M_goal, M_safe, M_stop, avg_d_stop]
    """
    seeds  = seed_indices if seed_indices is not None else list(range(n_seeds))
    n_ckpt = total_steps // eval_freq
    timesteps   = np.arange(1, n_ckpt + 1) * eval_freq   # x-axis: env steps
    # Pre-fill with NaN so incomplete checkpoints don't silently skew means.
    all_results = np.full((len(seeds), n_ckpt, 4), np.nan, dtype=np.float64)

    # ── Create CSV file with header before any training begins ─────────────
    _csv_init(csv_path)

    wall_start = time.time()

    for i, s in enumerate(seeds):
        seed_start = time.time()
        print(f"\n{'='*64}")
        print(f"  Seed {i + 1}/{len(seeds)}   (global seed = {s})")
        print(f"{'='*64}")

        # ── Create separate training and evaluation environments ───────────
        # Two HighwayStopEnvRM instances are independent Python objects;
        # unlike SUMO, no TraCI labels or external processes are involved.
        train_env = create_hwy_environment()
        train_env.enable_episode_logging = False

        # SB3 expects a VecEnv; DummyVecEnv wraps the single training env.
        vec_env = DummyVecEnv([lambda e=train_env: e])

        model = PPO("MlpPolicy", vec_env, seed=s, **_PPO_KWARGS)

        # Separate eval env — training env is never touched during evaluation.
        eval_env = create_hwy_environment()
        eval_env.enable_episode_logging = False

        pbar = tqdm(
            total=n_ckpt,
            desc=f"Seed {s:2d} ({i + 1}/{len(seeds)})",
            unit="ckpt",
            dynamic_ncols=True,
            leave=True,
        )

        for ckpt in range(n_ckpt):
            # ── Train for exactly eval_freq steps ──────────────────────────
            # reset_num_timesteps=False keeps the global step counter running
            # across chunks so SB3's internal logging stays coherent.
            model.learn(
                total_timesteps     = eval_freq,
                reset_num_timesteps = (ckpt == 0),
                progress_bar        = False,
            )

            # ── Evaluate on the separate eval env ──────────────────────────
            # eval_env is an independent Python object, so training env state
            # (model._last_obs in the DummyVecEnv) is unaffected.
            metrics = _evaluate_inline(model, eval_env, n_eval_ep)
            all_results[i, ckpt] = metrics

            # ── Persist this checkpoint to CSV immediately (crash-safe) ────
            _csv_append(csv_path, s, int(timesteps[ckpt]), metrics)

            # ── Progress reporting ─────────────────────────────────────────
            mg, ms, mst, ds = metrics
            pbar.set_postfix({
                'M_goal': f"{mg:.1f}%",
                'M_safe': f"{ms:.1f}%",
                'M_stop': f"{mst:.1f}%",
                'avg_ds': f"{ds:.2f}m" if not np.isnan(ds) else "nan",
            })
            pbar.update(1)

        pbar.close()

        # ── Persist intermediate results (crash-safe) ──────────────────────
        partial_path = os.path.join(_SCRIPT_DIR,
                                    f"s2g_multiseed_seed{s:02d}.npy")
        np.save(partial_path, all_results[i])

        seed_elapsed = time.time() - seed_start
        print(f"  Seed {s} done in {seed_elapsed / 60:.1f} min  "
              f"-> partial results: {partial_path}")

        eval_env.close()
        vec_env.close()   # also closes the underlying train_env

        # ── Explicit memory release between seeds ──────────────────────────
        del model, vec_env, train_env, eval_env
        gc.collect()

    wall_elapsed = time.time() - wall_start
    print(f"\nAll {len(seeds)} seeds done in {wall_elapsed / 60:.1f} min total.")
    return timesteps, all_results


# =============================================================================
# Plotting  (mirrors conv_plot.py / eval_multiseed_fcfs.py formatting exactly)
# =============================================================================

_IEEE_W = 7.16   # inches — full double-column width
_IEEE_H = 2.5    # inches — column height


def plot_results(
    timesteps:   np.ndarray,
    all_results: np.ndarray,
    output_file: str  = _PLOT_PDF,
    log_xaxis:   bool = True,
) -> None:
    """
    IEEE-formatted 1×2 convergence plot.

    Subplot (a): M_goal, M_safe, M_stop (percentage metrics).
    Subplot (b): avg_d_stop / D_stop (stopping distance in metres).

    Solid line = mean over seeds; shaded band = ±2 SE (95 % CI).
    Saves both a PDF (publication-ready) and a PNG (quick preview).
    """
    rc('text', usetex=True)
    rc('font', family='serif', size=12) #10

    base, ext = os.path.splitext(output_file)
    output_file = f"{base}_{'log' if log_xaxis else 'linear'}{ext}"

    learn_itrs_k = timesteps / 1000.0
    # Set x_min_k to one checkpoint before M_stop first becomes > 0, so the
    # plot starts just before the stopping behaviour emerges.  Fall back to the
    # very first checkpoint if M_stop is always zero.
    _mstop_mean = np.nanmean(all_results[:, :, 2], axis=0)  # shape (n_ckpts,)
    _nonzero_idx = np.where(_mstop_mean > 0)[0]
    if len(_nonzero_idx) > 0:
        x_min_k = learn_itrs_k[max(0, _nonzero_idx[0] - 1)]
    else:
        x_min_k = learn_itrs_k[0]
    x_max_k = learn_itrs_k[-1]

    # Colorblind-friendly palette (Wong 2011)
    colors = {
        'M_goal': '#0072B2',   # Blue
        'M_safe': '#009E73',   # Bluish green
        'M_stop': '#D55E00',   # Vermillion
        'D_stop': '#CC79A7',   # Reddish purple
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_IEEE_W, _IEEE_H))

    # =========================================================================
    # Subplot (a): percentage metrics — M_goal, M_safe, M_stop
    # =========================================================================
    for metric, m_idx in [('M_goal', 0), ('M_safe', 1), ('M_stop', 2)]:
        data  = all_results[:, :, m_idx]        # (n_seeds, n_checkpoints)
        mean  = np.nanmean(data, axis=0)
        valid = np.sum(~np.isnan(data), axis=0)
        ci    = np.where(
            valid > 1,
            2 * np.nanstd(data, axis=0, ddof=1) / np.sqrt(valid),
            np.nan,
        )

        ax1.fill_between(
            learn_itrs_k,
            np.maximum(mean - ci, 0),
            np.minimum(mean + ci, 100),
            color=colors[metric],
            alpha=0.2,
            linewidth=0,
        )
        ax1.plot(
            learn_itrs_k,
            mean,
            label=r'$M_{\mathrm{' + metric.split('_')[1] + r'}}$',
            color=colors[metric],
            linewidth=1.5,
        )

    if log_xaxis:
        ax1.set_xscale('log')
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, which='both')
        ax1.set_xlim(learn_itrs_k[0], x_max_k * 1.05) # x_min_k
    else:
        ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.set_xlim(0, x_max_k * 1.05)

    ax1.set_xlabel(r'Training Timesteps ($\times 10^3$)')
    ax1.set_ylabel(r'(\%)')
    ax1.legend(loc='center left', framealpha=0.9, fontsize=10) #9; 'best'
    #loc='best', framealpha=0.9; loc='lower right', bbox_to_anchor=(1.3, 0.0), framealpha=0.9
    ax1.set_title(r'(a) Task Completion Metrics')

    # =========================================================================
    # Subplot (b): stopping distance — avg_d_stop (= D_stop)
    # =========================================================================
    data_d  = all_results[:, :, 3]              # index 3 = avg_d_stop
    mean_d  = np.nanmean(data_d, axis=0)
    valid_d = np.sum(~np.isnan(data_d), axis=0)
    ci_d    = np.where(
        valid_d > 1,
        2 * np.nanstd(data_d, axis=0, ddof=1) / np.sqrt(valid_d),
        np.nan,
    )

    ax2.fill_between(
        learn_itrs_k,
        np.maximum(mean_d - ci_d, 0),
        mean_d + ci_d,
        color=colors['D_stop'],
        alpha=0.2,
        linewidth=0,
    )
    ax2.plot(
        learn_itrs_k,
        mean_d,
        label=r'$D_{\mathrm{stop}}$',
        color=colors['D_stop'],
        linewidth=1.5,
    )

    if log_xaxis:
        ax2.set_xscale('log')
        ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, which='both')
        ax2.set_xlim(x_min_k, x_max_k * 1.05)
    else:
        ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
        ax2.set_xlim(0, x_max_k * 1.05)

    ax2.set_xlabel(r'Training Timesteps ($\times 10^3$)')
    ax2.set_ylabel(r'Stopping Precision (m)')
    ax2.legend(loc='best', framealpha=0.9, fontsize=10) #9, loc='lower left'
    ax2.set_title(r'(b) Stopping Precision')

    upper_d = np.nanmax(mean_d + ci_d) if not np.all(np.isnan(mean_d)) else 10.0
    # 5% autoscale margin: -0.05*data_range
    ax2.set_ylim(bottom=-0.05*upper_d*1.1, top=upper_d * 1.1) #ax1.get_ylim()[0]

    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Figure saved -> {output_file}")
    png_file = output_file.replace('.pdf', '.png')
    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"PNG preview saved -> {png_file}")
    plt.close(fig)


# =============================================================================
# Incremental CSV helpers  (crash-safe, one row per checkpoint)
# =============================================================================
_CSV_HEADER = 'seed,learn_itrs,M_goal,M_safe,M_stop,D_stop'


def _csv_init(path: str) -> None:
    """Create (or overwrite) the CSV file with just the header line."""
    with open(path, 'w') as f:
        f.write(_CSV_HEADER + '\n')
    print(f"CSV initialised -> {path}")


def _csv_append(path: str, seed: int, learn_itrs: int,
                metrics: tuple[float, ...]) -> None:
    """
    Append one raw result row and flush immediately (crash-safe).

    Parameters
    ----------
    seed       : seed value used for this run
    learn_itrs : environment steps completed at this checkpoint
    metrics    : (M_goal, M_safe, M_stop, avg_d_stop)
    """
    mg, ms, mst, ds = metrics
    ds_str = f"{ds:.4f}" if not np.isnan(ds) else "nan"
    row = f"{seed},{learn_itrs},{mg:.4f},{ms:.4f},{mst:.4f},{ds_str}"
    with open(path, 'a') as f:
        f.write(row + '\n')


def _load_from_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct (timesteps, all_results) from the incremental CSV file.

    Reads rows with the header ``seed,learn_itrs,M_goal,M_safe,M_stop,D_stop``
    and builds arrays in the same shape as the .npy outputs so that the result
    can be passed directly to plot_results():

        timesteps   : ndarray (n_checkpoints,)
        all_results : ndarray (n_seeds, n_checkpoints, 4)
                      axis-2 order: [M_goal, M_safe, M_stop, avg_d_stop]

    Seeds are matched by their value in the first column.  Non-contiguous seed
    IDs (e.g. only seeds 0, 3, 7 were run) are sorted and mapped to row
    indices.  Checkpoints are sorted by learn_itrs.  Any (seed, checkpoint)
    pair absent from the file remains NaN so that nanmean / nanstd in
    plot_results() naturally ignores it.
    """
    rows: list[tuple[int, int, float, float, float, float]] = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('seed'):
                continue                      # skip header and blank lines
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

    # Discover all unique seeds and checkpoint positions
    seeds_sorted = sorted(set(r[0] for r in rows))
    ckpts_sorted = sorted(set(r[1] for r in rows))

    seed_to_idx = {s: i for i, s in enumerate(seeds_sorted)}
    ckpt_to_idx = {c: j for j, c in enumerate(ckpts_sorted)}

    n_seeds = len(seeds_sorted)
    n_ckpt  = len(ckpts_sorted)

    timesteps   = np.array(ckpts_sorted, dtype=np.int64)
    all_results = np.full((n_seeds, n_ckpt, 4), np.nan, dtype=np.float64)

    for seed_val, learn_itrs, mg, ms, mst, ds in rows:
        i = seed_to_idx[seed_val]
        j = ckpt_to_idx[learn_itrs]
        all_results[i, j, :] = [mg, ms, mst, ds]

    print(f"Loaded from CSV: {n_seeds} seed(s), {n_ckpt} checkpoint(s)  "
          f"(seeds present: {seeds_sorted})")
    return timesteps, all_results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed PPO training + evaluation on the highway-env "
            "Stop-and-Go task."
        )
    )
    def _parse_seed_list(s: str) -> list[int]:
        """Accept '[1,3,5,6]' or '1,3,5,6' and return [1, 3, 5, 6]."""
        try:
            return [int(x.strip()) for x in s.strip().strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--seed-list: cannot parse {s!r}. "
                "Expected format: '[1,3,5,6]' or '1,3,5,6'."
            )

    parser.add_argument('--seeds',      type=int, default=N_SEEDS,
                        help=f"Number of random seeds (default {N_SEEDS}); "
                             "ignored when --seed, --seed-start/--seed-end, "
                             "or --seed-list is used.")
    parser.add_argument('--seed',       type=int, default=None,
                        help="Run a single specific seed index (e.g. --seed 3). "
                             "Mutually exclusive with --seed-start/--seed-end "
                             "and --seed-list.")
    parser.add_argument('--seed-start', type=int, default=None,
                        help="Start of an inclusive seed range (e.g. --seed-start 1 "
                             "--seed-end 6 runs seeds [1,2,3,4,5,6]). "
                             "Must be paired with --seed-end. "
                             "Mutually exclusive with --seed and --seed-list.")
    parser.add_argument('--seed-end',   type=int, default=None,
                        help="End of an inclusive seed range (inclusive). "
                             "Must be paired with --seed-start. "
                             "Mutually exclusive with --seed and --seed-list.")
    parser.add_argument('--seed-list',  type=_parse_seed_list, default=None,
                        help="Explicit set of seed indices, e.g. '[1,3,5,6]' or "
                             "'1,3,5,6'. Brackets are optional. "
                             "Mutually exclusive with --seed and "
                             "--seed-start/--seed-end.")
    parser.add_argument('--steps',     type=int, default=TOTAL_STEPS,
                        help=f"Training steps per seed (default {TOTAL_STEPS})")
    parser.add_argument('--eval-freq', type=int, default=EVAL_FREQ,
                        help=f"Eval every N env steps (default {EVAL_FREQ})")
    parser.add_argument('--eval-eps',  type=int, default=EVAL_EPISODES,
                        help=f"Eval episodes per checkpoint (default {EVAL_EPISODES})")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plot-only', action='store_true',
                            help="Skip training; load saved .npy files and replot.")
    plot_group.add_argument('--plot-only-with-txt', action='store_true',
                            help="Skip training; load s2g_multiseed_results.txt "
                                 "and replot using the same format as --plot-only.")
    args = parser.parse_args()

    # ── Seed-selection validation ──────────────────────────────────────────────
    _range_given = (args.seed_start is not None or args.seed_end is not None)
    _list_given  = args.seed_list is not None

    if args.seed is not None and (_range_given or _list_given):
        parser.error("--seed cannot be combined with --seed-start/--seed-end "
                     "or --seed-list.")
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
        #seed_indices = None   # run 0 .. args.seeds-1

    if args.plot_only:
        if not (os.path.exists(_RESULTS_NPY) and os.path.exists(_TIMESTEPS_NPY)):
            parser.error(
                "--plot-only requires s2g_multiseed_results.npy and "
                "s2g_multiseed_timesteps.npy to exist in the script directory."
            )
        timesteps   = np.load(_TIMESTEPS_NPY)
        all_results = np.load(_RESULTS_NPY)
        print(f"Loaded results: {all_results.shape}  "
              f"seeds={all_results.shape[0]}, "
              f"checkpoints={all_results.shape[1]}")
    elif args.plot_only_with_txt:
        if not os.path.exists(_CSV_TXT):
            parser.error(
                f"--plot-only-with-txt requires {_CSV_TXT} to exist."
            )
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
