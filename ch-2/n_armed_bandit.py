# %%
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from action_selection_rule import (
    ActionSelectionRule,
    Greedy,
    EpsilonGreedy,
    UpperConfidenceBound,
    Softmax,
)
from value_estimator import (
    ExponentialRecencyWeightedAverage,
    ValueEstimator,
    SampleAverage,
    GradientBandit,
)


class NArmedBandit:

    def __init__(
        self,
        action_selector: ActionSelectionRule,
        value_estimator: ValueEstimator,
        label: str = None,
    ):
        self.action_selector = action_selector
        self.value_estimator = value_estimator
        self.label = label

    def select_action(self) -> int:
        return self.action_selector.select_action(
            self.value_estimator.get_value_estimates()
        )

    def update_value_estimates(self, action_taken: int, reward_signal: float):
        self.value_estimator.update_value_estimates(action_taken, reward_signal)

    def reset(self):
        self.value_estimator.reset()
        self.action_selector.reset()


class ArmedBanditTask:

    def __init__(self, n_arms: int, stationary: bool = True, seed: int = None):
        self.n_arms = n_arms
        self.stationary = stationary
        if seed:
            np.random.seed(seed)
        self.bandit_values = np.random.normal(loc=0.0, scale=1.0, size=n_arms)
        self.optimal_action = np.argmax(self.bandit_values)

    def get_reward_signal(self, arm_id: int) -> float:
        return self.bandit_values[arm_id] + np.random.normal(loc=0.0, scale=1.0)

    def simulate_bandit_nsteps(self, bandit: NArmedBandit, n_steps: int = 1000):
        rewards_observed = []
        optimal_actions = []
        for i in range(n_steps):
            bandit, rewards_observed, optimal_actions = self.simulate_bandit_one_step(
                bandit=bandit,
                rewards_observed=rewards_observed,
                optimal_actions=optimal_actions,
            )
            if not self.stationary:
                self.random_walk_update()

        return np.array(rewards_observed), np.array(optimal_actions).astype(int)

    def simulate_bandit_one_step(
        self,
        bandit: NArmedBandit,
        rewards_observed: List[float],
        optimal_actions: List[int],
        debug=False,
    ):
        arm_id = bandit.select_action()
        if debug:
            print(
                f"Selected action {arm_id} with bandit state {bandit.value_estimator.get_value_estimates()}"
            )
        reward = self.get_reward_signal(arm_id)
        bandit.update_value_estimates(arm_id, reward)
        if debug:
            print(
                f"Observed reward {reward}, bandit value estimates updated to {bandit.value_estimator.get_value_estimates()}"
            )
            print("")
        rewards_observed.append(reward)
        optimal_actions.append(arm_id == self.optimal_action)
        return bandit, rewards_observed, optimal_actions

    def random_walk_update(self):
        # For non-stationary environment simulation
        self.bandit_values += np.random.normal(loc=0, scale=0.01, size=self.n_arms)
        self.optimal_action = np.argmax(self.bandit_values)


class ArmedBanditTestBed:

    def simulate_bandits(
        self,
        bandits: List[NArmedBandit],
        n_arms: int,
        n_tasks: int,
        n_steps: int,
        stationary=True,
    ):
        n_bandits = len(bandits)
        sum_rewards = np.zeros((n_bandits, n_steps))
        sum_optimal = np.zeros((n_bandits, n_steps))
        for i in range(n_tasks):
            task = ArmedBanditTask(n_arms, stationary)
            for j, bandit in enumerate(bandits):
                bandit.reset()
                rewards, optimal = task.simulate_bandit_nsteps(bandit, n_steps=n_steps)
                sum_rewards[j] += rewards
                sum_optimal[j] += optimal
        avg_rewards = sum_rewards / n_tasks
        avg_optimal = sum_optimal / n_tasks
        self.plot_helper(avg_rewards, "Average reward", bandits)
        self.plot_helper(avg_optimal, "% Optimal Action", bandits)

    def plot_helper(
        self, series: np.ndarray[float], title: str, bandits: List[NArmedBandit]
    ):
        for i in range(len(series)):
            plt.plot(series[i], label=bandits[i].label)
        plt.title(title)
        plt.legend(loc="upper left")
        plt.xlabel("Steps")
        plt.savefig(f"{title.replace(' ', '_')}_{'_'.join([bandit.label for bandit in bandits])}.png")
        plt.show()


# def plot_optimal_actions(optimal_actions: np.ndarray[int]):


def main():

    n_arms = 10
    n_steps = 10000
    n_tasks = 2000
    stationary = False
    # Exercise 2.5
    bandits = [
        NArmedBandit(
            action_selector=EpsilonGreedy(epsilon=0.1),
            value_estimator=SampleAverage(
                initial_estimates=np.zeros(n_arms),
            ),
            label="Sample Average",
        ),
        NArmedBandit(
            action_selector=EpsilonGreedy(epsilon=0.1),
            value_estimator=ExponentialRecencyWeightedAverage(
                initial_estimates=np.zeros(n_arms), alpha=0.1
            ),
            label="ERWA",
        ),
    ]

    test_bed = ArmedBanditTestBed()
    test_bed.simulate_bandits(bandits, n_arms, n_tasks, n_steps, stationary)


# %%
if __name__ == "__main__":
    main()

# %%
