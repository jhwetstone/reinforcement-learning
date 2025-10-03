from abc import abstractmethod
import numpy as np

from ch_2.action_selection_rule import Greedy, ActionSelectionRule

class ValueUpdateRule:

    def __init__(self, alpha: float, gamma: float):
        '''
        params
          - alpha: the learning rate
          - gamma: the discount factor
        '''
        self.alpha = alpha
        self.gamma = gamma

    def update_value(self, old_value: float, target_value: float, reward: float) -> float:
        return old_value + self.alpha * (
          reward + self.gamma * target_value - old_value
        )

    @abstractmethod
    def target_value(self, value_estimates: np.ndarray[float], next_state: tuple) -> float:
        pass

class SARSA(ValueUpdateRule):
    ''' Implements the SARSA value update rule for Q(s,a).
        - on-policy rule
        - target is the next (state, action) pair based on the current values and policy
    '''

    def __init__(self, alpha: float, gamma: float, target_policy: ActionSelectionRule):
        self.target_policy = target_policy
        super().__init__(alpha=alpha, gamma=gamma)

    def target_value(self,
                     value_estimates: np.ndarray[float],
                     next_state: tuple[float]
                     ) -> float:

       next_action = self.target_policy.select_action(value_estimates[next_state[0], next_state[1], :])
       return value_estimates[next_state[0], next_state[1], next_action]

class QLearning(ValueUpdateRule):
    ''' Implements the Q-learning value update rule for Q(s,a).
        - off-policy rule
        - target is the best action given the next state (the action chosen by a greedy policy)
    '''


    def __init__(self, alpha: float, gamma: float):
        self.target_policy = Greedy()
        super().__init__(alpha=alpha, gamma=gamma)

    def target_value(self,
                     value_estimates: np.ndarray[float],
                     next_state: tuple[float]
    ) -> float:
       next_action = self.target_policy.select_action(value_estimates[next_state[0], next_state[1], :])
       return value_estimates[next_state[0], next_state[1], next_action]

class ExpectedSarsa(ValueUpdateRule):
    ''' Implements the Expected Sarsa value update rule for Q(s,a).
        - off-policy or on-policy rule, depending on whether the target action selection rule is the current policy
        - target is the expected value of the next state/action pair, given the target policy
    '''

    def __init__(self, alpha: float, gamma: float, target_policy: ActionSelectionRule):
        self.target_policy = target_policy
        super().__init__(alpha=alpha, gamma=gamma)

    def target_value(self,
                     value_estimates: np.ndarray[float],
                     next_state: tuple[float]
    ) -> float:
        return np.sum(
            self.target_policy.get_action_probabilities(value_estimates[next_state[0], next_state[1], :]) *
            value_estimates[next_state[0], next_state[1], :]
        )