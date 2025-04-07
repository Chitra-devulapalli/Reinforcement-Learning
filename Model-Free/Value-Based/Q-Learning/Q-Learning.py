import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Grid:
    """ Class to create a grid world (5x5) environment for Q-learning """
    def __init__(self):
        self.grid = np.zeros((5, 5))  # 5x5 grid world
        self.grid[0][3] = 1         # Goal state
        self.grid[1][2] = -1        # Obstacle state

        self.start_state = (3, 0)   # Start state (row, column)
        self.state = self.start_state

    def reset(self):
        """
        Reset the environment to the start state
        :return: Start state
        """
        self.state = self.start_state
        return self.state
    
    def is_terminal(self, state):
        """
        Check if the state is a terminal state
        :param state: Current state (row, column)
        :return: True if terminal, False otherwise
        """
        row, col = state
        return self.grid[row][col] == 1 or self.grid[row][col] == -1
    
    def get_next_action(self, state, action):
        next_state = list(state)
        if action == 0:  # Move Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Move Right
            next_state[1] = min(4, state[1] + 1)  # Column shouldn't exceed 4
        elif action == 2:  # Move Down
            next_state[0] = min(4, state[0] + 1)  # Row shouldn't exceed 4
        elif action == 3:  # Move Left
            next_state[1] = max(0, state[1] - 1)  # Column shouldn't be less than 0  

        return tuple(next_state)
    
    def step(self, action):
        """
        Take a step in the environment based on the action taken
        :param action: Action to take (0: Up, 1: Right, 2: Down, 3: Left)
        :return: (next_state, reward, done)
        """
        next_state = self.get_next_action(self.state, action)

        # Get the reward for the next state
        if self.grid[next_state[0]][next_state[1]] == 1:
            reward = 10
        elif self.grid[next_state[0]][next_state[1]] == -1:
            reward = -10
        else:
            reward = 0
        
        done = self.is_terminal(next_state)  # Check if the next state is terminal
        self.state = next_state

        return next_state, reward, done
    
class QLearning:
    def __init__(self, learning_rate=0.01, discount_factor=0.9, epsilon=0.1):
        """
        Initialize Q-learning parameters
        :param learning_rate: Learning rate (alpha)
        :param discount_factor: Discount factor (gamma)
        :param epsilon: Exploration probability
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros((5, 5, 4))

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy
        :param state: Current state (row, column)
        :return: Action to take (0: Up, 1: Right, 2: Down, 3: Left)
        """
        if np.random.rand() < self.exploration_prob:
            # Explore: choose a random action
            return np.random.choice(4)
        else:
            # Exploit: choose the action with the highest Q-value
            row, col = state
            return np.argmax(self.q_table[row][col])
        
    def update_q(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state[0]][next_state[1]])
        current_q = self.q_table[state[0]][state[1]][action]
        # Update the Q-value using the Q-learning formula
        self.q_table[state[0]][state[1]][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

# Training the agent 
env = Grid()
agent = QLearning()

num_episodes = 1000  # Number of episodes to train the agent 
# Dictionary to store snapshots of Q-values for action 0 at specific episodes
snapshots = {}
accumulated_rewards = []  # List to store accumulated rewards for each episode

for episode in range(num_episodes):
    state = env.reset()  # Reset the environment to start state
    done = False
    episode_reward = 0  # Initialize episode reward

    while not done:
        action = agent.choose_action(state)  # Choose an action based on epsilon-greedy policy
        next_state, reward, done = env.step(action)  # Take a step in the environment

        # Update the Q-value based on the action taken
        agent.update_q(state, action, reward, next_state)

        state = next_state  # Move to the next state
        episode_reward += reward  # Accumulate the reward for the episode

    # After each episode, if it's one of the desired episodes, store a snapshot of Q-values for action 0
    if episode in [0, 500, 999]:
        snapshots[episode] = agent.q_table[:, :, 0].copy()

    accumulated_rewards.append(episode_reward)  # Store the accumulated reward for the episode

# Extracting the learned policy from the Q-table
policy = np.chararray((5, 5), itemsize=1)  # Initialize a policy array
for row in range(5):
    for col in range(5):
        if env.grid[row][col] == 1:
            policy[row][col] = 'G'  # Goal state
        elif env.grid[row][col] == -1:
            policy[row][col] = 'X'  # Obstacle state
        elif row == 3 and col == 0:
            policy[row][col] = 'S'  # Start state
        else:
            best_action = np.argmax(agent.q_table[row][col])
            if best_action == 0:
                policy[row][col] = 'U'  # Up
            elif best_action == 1:
                policy[row][col] = 'R'  # Right
            elif best_action == 2:
                policy[row][col] = 'D'  # Down
            elif best_action == 3:
                policy[row][col] = 'L'  # Left

# Print the learned policy
print("Learned Policy:")
for row in range(5):
    for col in range(5):
        # Decode byte string to normal string for printing
        print(policy[row][col].decode('utf-8'), end=' ' if col < 4 else '\n')

# Plotting heatmaps as subplots after printing the policy
episodes_to_plot = [0, 500, 999]

# Create a figure with 2 rows: first row for heatmaps, second row for the policy
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5])  # 2 rows, 3 columns; second row is shorter

# First row: heatmaps for each specified episode
for idx, ep in enumerate(episodes_to_plot):
    ax = fig.add_subplot(gs[0, idx])
    if ep in snapshots:
        im = ax.imshow(snapshots[ep], cmap='hot', interpolation='nearest')
        ax.set_title(f'Episode {ep}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        fig.colorbar(im, ax=ax, label='Q-value')

# Second row: policy subplot spanning all 3 columns
ax_policy = fig.add_subplot(gs[1, :])
# Create a blank background for the policy grid
ax_policy.imshow(np.zeros((5, 5)), cmap='gray', alpha=0.3)
ax_policy.set_xticks(np.arange(5))
ax_policy.set_yticks(np.arange(5))
ax_policy.set_title("Learned Policy")
ax_policy.set_xlabel("Column")
ax_policy.set_ylabel("Row")

# Annotate each cell with the corresponding policy letter
for i in range(5):
    for j in range(5):
        # Decode the byte string to a normal string for printing
        letter = policy[i, j].decode('utf-8')
        ax_policy.text(j, i, letter, ha='center', va='center', fontsize=18)

plt.tight_layout()
plt.savefig('q_learning_heatmaps_and_policy.png')  # Save the figure to a file
plt.show()

# Plotting the accumulated rewards over episodes
plt.figure(figsize=(10, 5))
plt.plot(accumulated_rewards)
plt.title('Accumulated Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Accumulated Reward')
plt.grid()
plt.show()
