import matplotlib.pyplot as plt
import numpy as np

class Grid:
  '''
  Grid class for implementing 'Windy Gridworld' from Example 6.5
  '''
  def __init__(self, dim_x, dim_y, start, goal, wind=None, max_path_length=500, stochastic = False):
    self.dim_x = dim_x
    self.dim_y = dim_y
    self.start = start
    self.goal = goal
    self.wind = wind
    self.max_path_length = max_path_length
    self.stochastic = stochastic

  def take_step(self, pos, action):

    ## If wind exists, find the wind adjustment to the y dimension
    if self.wind:
      if self.stochastic:
        rand_elem = np.random.choice([-1,0,1])
      else:
        rand_elem = 0
      w_adj = self.wind[pos[0]] + rand_elem
    else:
      w_adj = 0

    # Update x value
    new_x = max(0, min(self.dim_x - 1, pos[0] + action.x))

    # Update y value
    new_y = max(0, min(self.dim_y - 1, pos[1] + action.y + w_adj))

    return (new_x, new_y)

  def draw_grid(self, path=None):
    x_coords = np.arange(0, self.dim_x)
    y_coords = np.arange(0, self.dim_y)

    plt.figure(figsize=(self.dim_x + 1, self.dim_y + 1))

    X, Y = np.meshgrid(x_coords, y_coords)

    # Plot the points
    # 'o' specifies circle markers, 'k' specifies black color
    plt.plot(X, Y, 'ok')

    # Set plot limits to ensure all points are visible and centered
    plt.xlim(-0.5, self.dim_x - 0.5)
    plt.ylim(-0.5, self.dim_y - 0.5)

    # Add grid lines for visual reference of cells
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot the path
    if path:
      plt.plot([p[0] for p in path], [p[1] for p in path], 'b-')

    # Plot start and end points
    plt.plot(self.start[0], self.start[1], 'ro')
    plt.annotate("START", self.start,  # The text and the data point coordinates
              textcoords="offset points",  # Position the text relative to the point
              xytext=(0, 10),  # Offset the text by 10 points vertically
              ha='center',
              color='red')
    plt.plot(self.goal[0], self.goal[1], 'go')
    plt.annotate("GOAL", self.goal,  # The text and the data point coordinates
          textcoords="offset points",  # Position the text relative to the point
          xytext=(0, 10),  # Offset the text by 10 points vertically
          ha='center',
          color='green')

    if self.wind:
      plt.xticks(ticks=np.arange(0, self.dim_x), labels=self.wind)
      plt.yticks([])

    # Set axis labels and title
    title = "Grid World"
    if path:
      title += f": {len(path) - 1} steps"
    plt.title(title)

    # Ensure equal aspect ratio for square cells
    plt.gca().set_aspect('equal', adjustable='box')

    # Display the plot
    _ = plt.show()
# %%
def main():
    DIM_Y = 7
    DIM_X = 10
    WIND = [0,0,0,1,1,1,2,2,1,0]
    START = (0,3)
    GOAL = (7,3)
    PATH = [(0,3),(1,3),(1,4),(2,4),(2,5),(3,5),(4,5),(5,5),(5,4),(5,3),(6,3),(7,3)]
    g = Grid(DIM_X, DIM_Y, START, GOAL, WIND)
    g.draw_grid(path=PATH)

# %%
if __name__ == "__main__":
  main()