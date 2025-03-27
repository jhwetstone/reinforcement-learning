from abc import abstractmethod

from copy import copy
import numpy as np


class ValueEstimator:

    def __init__(self, initial_estimates: np.ndarray[float]):
        self.value_estimates = copy(initial_estimates)
        self.initial_estimates = initial_estimates
        self.n_arms = len(initial_estimates)

    @abstractmethod
    def update_value_estimates(self, action_taken: int, reward_signal: float):
        pass

    def reset(self):
        self.value_estimates = copy(self.initial_estimates)

    def get_value_estimates(self):
        return self.value_estimates

    def __repr__(self):
        return str(self.get_value_estimates())


class SampleAverage(ValueEstimator):

    def __init__(self, initial_estimates: np.ndarray[float]):
        self.action_counter = np.zeros(len(initial_estimates))
        super().__init__(initial_estimates)

    def reset(self):
        self.action_counter = np.zeros(len(self.value_estimates))
        super().reset()

    def update_value_estimates(self, action_taken: int, reward_signal: float):
        if self.action_counter[action_taken] >= 1:
            self.value_estimates[action_taken] += (
                1
                / self.action_counter[action_taken]
                * (reward_signal - self.value_estimates[action_taken])
            )
        else:
            self.value_estimates[action_taken] = reward_signal

        self.action_counter[action_taken] += 1


class ExponentialRecencyWeightedAverage(ValueEstimator):

    def __init__(self, alpha: float, initial_estimates: np.ndarray[float]):
        self.alpha = alpha
        super().__init__(initial_estimates)

    def update_value_estimates(self, action_taken: int, reward_signal: float):
        self.value_estimates[action_taken] += self.alpha * (
            reward_signal - self.value_estimates[action_taken]
        )


class GradientBandit(ValueEstimator):
    def __init__(self, alpha: float, initial_estimates: np.ndarray[float]):
        self.alpha = alpha
        self.reward_avg = 0
        self.reward_counter = 0
        super().__init__(initial_estimates)

    @staticmethod
    def get_softmax_probs(value_estimates):
        # numerically stable softmax
        offset_vals = value_estimates - np.max(value_estimates)
        return np.exp(offset_vals) / np.sum(np.exp(offset_vals))

    def update_value_estimates(self, action_taken: int, reward_signal: float):
        probabilities = self.get_softmax_probs(self.value_estimates)
        for i in range(self.n_arms):
            if i == action_taken:
                self.value_estimates[action_taken] += (
                    self.alpha
                    * (reward_signal - self.reward_avg)
                    * (1 - probabilities[action_taken])
                )
            else:
                self.value_estimates[i] -= (
                    self.alpha * (reward_signal - self.reward_avg) * probabilities[i]
                )

        if self.reward_counter >= 1:
            self.reward_avg += (
                1 / self.reward_counter * (reward_signal - self.reward_avg)
            )
        else:
            self.reward_avg = reward_signal
        self.reward_counter += 1

    def reset(self):
        self.reward_avg = 0
        self.reward_counter = 0
        super().reset()
