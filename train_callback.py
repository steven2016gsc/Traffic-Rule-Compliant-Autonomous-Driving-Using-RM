from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

class TrainStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_rates = []
        self.episode_rewards_history = []
        self.episode_lengths_history = []

    def _on_training_start(self):
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.exploration_rates.clear()

    def _on_step(self):
        # Log exploration rate
        if hasattr(self.model, "exploration_rate"):
            self.exploration_rates.append(self.model.exploration_rate)
        # Log reward and length if an episode finished
        done_array = np.array(self.locals['dones'])
        infos = self.locals['infos']
        for idx, done in enumerate(done_array):
            if done:
                if "episode" in infos[idx]:
                    self.episode_rewards.append(infos[idx]["episode"]["r"])
                    self.episode_lengths.append(infos[idx]["episode"]["l"])
        return True

    def _on_rollout_end(self):
        # At the end of each rollout, keep history for plotting
        if self.episode_rewards:
            self.episode_rewards_history.append(np.mean(self.episode_rewards[-10:]))  # Last 10
            self.episode_lengths_history.append(np.mean(self.episode_lengths[-10:]))  # Last 10

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Info for each env in the vectorized environment
        for info in self.locals["infos"]:
            if "episode" in info.keys():
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                print(f"Episode {len(self.episode_rewards)} - Reward: {reward}")
        return True