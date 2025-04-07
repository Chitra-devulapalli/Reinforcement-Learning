import numpy as np 
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.gridspec as gridspec  # Added for subplot layout

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridEnv:
    """
    A grid environment for DQN.
    States are (row, col).
    """
    def __init__(self, size=10):
        self.size = size
        # 0 = empty, 1 = goal, -1 = obstacle
        self.grid = np.zeros((size, size))

        # set multiple goals and obstacles
        self.grid[1][8] = 1    # Goal
        self.grid[3][4] = -1   # Obstacle
        self.grid[1][5] = -1   # Obstacle
        self.grid[7][2] = -1   # Obstacle

        # Start in bottom-left corner
        self.start_state = (size - 1, 0)
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
        """
        row, col = state
        return self.grid[row][col] == 1 or self.grid[row][col] == -1

    def step(self, action):
        """
        action: 0=Up, 1=Right, 2=Down, 3=Left
        Returns: next_state, reward, done
        """
        row, col = self.state
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Right
            col = min(col + 1, self.size - 1)
        elif action == 2:  # Down
            row = min(row + 1, self.size - 1)
        elif action == 3:  # Left
            col = max(col - 1, 0)

        next_state = (row, col)

        # Rewards
        if self.grid[row][col] == 1:
            reward = 10
        elif self.grid[row][col] == -1:
            reward = -10
        else:
            reward = 0

        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

class ReplayBuffer:
    """
    We need a replay buffer to store transitions (s,a,r,s',done)
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done_sample = zip(*batch)
        return states, actions, rewards, next_states, done_sample

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    A 2 layer MLP that takes vector of states and actions [s,a] and returns the Q values Q(s,a)
    """
    def __init__(self, input_dim = 2, hidden_dim = 64, output_dim = 4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(
    num_episodes=100,
    batch_size=32,
    buffer_capacity=10000,
    gamma=0.9,
    learning_rate=0.001,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=500,
    target_update_freq=10
):
    """
    Train the DQN agent on the grid environment
    """
    env = GridEnv()
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Creating 2 networks, one for training and one for target
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # we have to set the target network to eval mode

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate) # parameters are w, b of policy_net

    def epsilon(step):
        return max(epsilon_end, epsilon_start*(0.99**(step/epsilon_decay))) # exponential decay
    
    all_rewards = []
    step_count = 0

    # For capturing snapshots of Q-values at episodes 0, 50, and 99
    episodes_to_snapshot = [0, 50, 99]
    snapshots = {}

    for episode in range(num_episodes):
        state = env.reset() # putting the agent in the start state
        done = False
        episode_reward = 0

        while not done:
            step_count += 1
            epsilon_value = epsilon(step_count)
            if np.random.rand() < epsilon_value:
                action = np.random.randint(0, 4) # random action
            else:
                # use the policy network to get the action
                state_tensor = torch.FloatTensor([state[0], state[1]]).to(device) # state 0 and 1 are row and column
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action) # take the action
            replay_buffer.push(state, action, reward, next_state, done) # store the transition in the replay buffer
            episode_reward += reward
            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size:
                # sample minibatch from the replay buffer
                states, actions, rewards, next_states, done_sample = replay_buffer.sample(batch_size)
                # converting to tensors
                states = torch.FloatTensor([[s[0], s[1]] for s in states]).to(device)
                actions = torch.LongTensor(actions).unsqueeze(-1).to(device) # shape = (batch_size,1)
                rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
                next_states = torch.FloatTensor([[ns[0], ns[1]] for ns in next_states]).to(device)
                done_sample = torch.Tensor(done_sample).unsqueeze(-1).to(device)

                # current q-values (before training)
                q_values = policy_net(states)
                # Gather the Q-value corresponding to each action in the batch
                q_values = q_values.gather(1, actions) # gather is a pytorch method

                # getting next q-values from target net
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(dim=1, keepdim = True)[0] # 0 coz max returns values and indices, we just want the values

                # calculating target to be used in loss
                target = rewards + gamma * max_next_q_values * (1 - done_sample)

                # loss using mse
                loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # update target net after training, but not always (depending on target_update_freq)
            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(episode_reward)

        # Capture snapshot of Q-values (for action 0) over the entire grid at specific episodes
        if episode in episodes_to_snapshot:
            snapshot = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    state_tensor = torch.FloatTensor([i, j]).to(device)
                    with torch.no_grad():
                        q_vals = policy_net(state_tensor)
                    snapshot[i, j] = q_vals[0].item()  # storing Q-value for action 0
            snapshots[episode] = snapshot

        print(f"Episode {episode+1}/{num_episodes}, Epsilon: {epsilon_value}, Episode Reward: {episode_reward}")

    return policy_net, target_net, all_rewards, snapshots

if __name__ == "__main__":
    policy_net, target_net, rewards, snapshots = train_dqn(num_episodes=100)
    # Plotting the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.show()

    # -------------------------------
    # Plotting heatmaps as subplots after printing the policy
    # -------------------------------
    # We will plot snapshots (heatmaps) at episodes 0, 50, and 99.
    episodes_to_plot = [0, 50, 99]

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
            ax.set_xticks(np.arange(0, 11, 2))
            ax.set_yticks(np.arange(0, 11, 2))
            fig.colorbar(im, ax=ax, label='Q-value')

    # Second row: policy subplot spanning all 3 columns
    # Now that the grid is 10x10, we plot the learned policy directly.
    env = GridEnv()  # Create an environment instance to access grid information
    policy_vis = np.chararray((10, 10), itemsize=1)
    for i in range(10):
        for j in range(10):
            if env.grid[i][j] == 1:
                policy_vis[i, j] = 'G'  # Goal state
            elif env.grid[i][j] == -1:
                policy_vis[i, j] = 'X'  # Obstacle state
            elif (i, j) == env.start_state:
                policy_vis[i, j] = 'S'  # Start state
            else:
                state_tensor = torch.FloatTensor([i, j]).to(device)
                with torch.no_grad():
                    q_vals = policy_net(state_tensor)
                best_action = torch.argmax(q_vals).item()
                if best_action == 0:
                    policy_vis[i, j] = 'U'  # Up
                elif best_action == 1:
                    policy_vis[i, j] = 'R'  # Right
                elif best_action == 2:
                    policy_vis[i, j] = 'D'  # Down
                elif best_action == 3:
                    policy_vis[i, j] = 'L'  # Left

    ax_policy = fig.add_subplot(gs[1, :])
    # Create a blank background for the policy grid (10x10)
    ax_policy.imshow(np.zeros((10, 10)), cmap='gray', alpha=0.3)
    ax_policy.set_xticks(np.arange(10))
    ax_policy.set_yticks(np.arange(10))
    ax_policy.set_title("Learned Policy (10x10)")
    ax_policy.set_xlabel("Column")
    ax_policy.set_ylabel("Row")

    # Annotate each cell with the corresponding policy letter
    for i in range(10):
        for j in range(10):
            letter = policy_vis[i, j].decode('utf-8')
            ax_policy.text(j, i, letter, ha='center', va='center', fontsize=18)

    plt.tight_layout()
    plt.savefig('dqn_heatmaps_and_policy.png')  # Save the figure to a file
    plt.show()

    # Plotting the accumulated rewards over episodes
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Accumulated Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.grid()
    plt.show()

