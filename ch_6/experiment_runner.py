from enum import Enum
import numpy as np

from ch_6.grid import Grid
from ch_2.action_selection_rule import ActionSelectionRule, Greedy, EpsilonGreedy
from ch_6.value_update_rule import ValueUpdateRule, SARSA, QLearning
from ch_6.action import Action, KingAction, KingActionWithStayOption

class ExperimentRunner:

  def __init__(self, grid: Grid, policy: ActionSelectionRule, value_update_rule: ValueUpdateRule, avail_actions: Enum):
    self.grid = grid
    self.policy = policy
    self.value_update_rule = value_update_rule
    self.max_policy = Greedy()
    self.avail_actions = avail_actions
    self.values = np.zeros((self.grid.dim_x, self.grid.dim_y, len(self.avail_actions)))

  def single_episode(self):
    curr_state = self.grid.start
    a = self.policy.select_action(self.values[curr_state[0], curr_state[1], :])
    path = [curr_state]
    while curr_state != self.grid.goal and len(path) < self.grid.max_path_length:
      s_prime = self.grid.take_step(curr_state, self.avail_actions(a))
      r = -1 if s_prime != self.grid.goal else 0
      a_prime = self.policy.select_action(self.values[s_prime[0], s_prime[1], :])
      target_value = self.value_update_rule.target_value(self.values, s_prime)
      self.values[curr_state[0], curr_state[1], a] = self.value_update_rule.update_value(self.values[curr_state[0], curr_state[1], a], target_value, r)
      curr_state = s_prime
      a = a_prime
      path += [curr_state]
    return len(path) - 1

  def train_n_steps(self, n_step_lim: int):
    n_steps = 0
    n_eps = 0
    while n_steps < n_step_lim:
      n_steps += self.single_episode()
      n_eps += 1
      if n_eps % 100 == 0:
        print(f"Episode {n_eps}")
        curr_state = self.grid.start
        path = [curr_state]
        while curr_state != self.grid.goal and len(path) < self.grid.max_path_length:
          curr_state = self.grid.take_step(curr_state, self.avail_actions(self.max_policy.select_action(self.values[curr_state[0], curr_state[1], :])))
          path += [curr_state]
        print(path)
        self.grid.draw_grid(path=path)
# %%
def main():
    DIM_Y = 7
    DIM_X = 10
    WIND = [0,0,0,1,1,1,2,2,1,0]
    START = (0,3)
    GOAL = (7,3)
    
    g = Grid(
        dim_x=DIM_X,
        dim_y=DIM_Y,
        start=START,
        goal=GOAL,
        wind=WIND,
        stochastic=False
    )
    policy = EpsilonGreedy(epsilon=0.1)
    value_update_rule = SARSA(alpha=0.5, gamma=1, target_policy=policy)
    er = ExperimentRunner(grid=g, policy=policy, value_update_rule=value_update_rule, avail_actions=Action)
    er.train_n_steps(10000)

    g = Grid(
        dim_x=DIM_X,
        dim_y=DIM_Y,
        start=START,
        goal=GOAL,
        wind=WIND,
        stochastic=False
    )
    policy = EpsilonGreedy(epsilon=0.1)
    value_update_rule = QLearning(alpha=0.5, gamma=1)
    er = ExperimentRunner(grid=g, policy=policy, value_update_rule=value_update_rule, avail_actions=KingAction)
    er.train_n_steps(10000)

    g = Grid(
        dim_x=DIM_X,
        dim_y=DIM_Y,
        start=START,
        goal=GOAL,
        wind=WIND,
        stochastic=False
    )
    policy = EpsilonGreedy(epsilon=0.1)
    value_update_rule = QLearning(alpha=0.5, gamma=1)
    er = ExperimentRunner(grid=g, policy=policy, value_update_rule=value_update_rule, avail_actions=KingActionWithStayOption)
    er.train_n_steps(10000)

#%%
if __name__ == "__main__":
  main()