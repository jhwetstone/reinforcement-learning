from abc import abstractmethod

import numpy as np
from value_estimator import GradientBandit


class ActionSelectionRule:

    @abstractmethod
    def select_action(value_estimates: np.ndarray[float]) -> int:
        pass

    def reset(self):
        return

    @staticmethod
    def ids_of_max_value(input_arr: np.ndarray[float]) -> int:
        max_val = np.max(input_arr)
        return np.where(input_arr == max_val)[0]


class Greedy(ActionSelectionRule):

    def select_action(self, value_estimates: np.ndarray[float]) -> int:
        return np.random.choice(self.ids_of_max_value(value_estimates))


class Softmax(ActionSelectionRule):

    def select_action(self, value_estimates: np.ndarray[float]) -> int:
        probabilities = GradientBandit.get_softmax_probs(value_estimates)
        return np.random.choice(range(len(value_estimates)), p=probabilities)


class EpsilonGreedy(ActionSelectionRule):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, value_estimates: np.ndarray[float]) -> int:
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            return np.random.choice(len(value_estimates))
        return np.random.choice(self.ids_of_max_value(value_estimates))


class UpperConfidenceBound(ActionSelectionRule):

    def __init__(self, c: float, n_arms: int):
        self.time_steps = 0
        self.c = c
        self.n_arms = n_arms
        self.action_counter = np.zeros(n_arms)

    def select_action(self, value_estimates: np.ndarray[float]) -> int:

        if min(self.action_counter) == 0:
            next_action = np.argmin(self.action_counter)
        else:
            next_action = np.random.choice(self.ids_of_max_value(
                np.array(value_estimates)
                + self.c * np.sqrt(np.log(self.time_steps) / self.action_counter)
            ))

        self.action_counter[next_action] += 1
        self.time_steps += 1
        return next_action

    def reset(self):
        self.action_counter = np.zeros(self.n_arms)
        self.time_steps = 0
