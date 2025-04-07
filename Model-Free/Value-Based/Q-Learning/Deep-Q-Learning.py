import numpy as np 
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random

class GridEnv:
    """
    A grid environment for DQN.
    States are (row, col).
    """
    def __init__(self, size=100):
        self.size = size
        # 0 = empty, 1 = goal, -1 = obstacle
        self.grid = np.zeros((size, size))

        # set multiple goals and obstacles
        self.grid[1][80] = 1    # Goal
        self.grid[34][43] = -1   # Obstacle
        self.grid[15][50] = -1   # Obstacle
        self.grid[70][25] = -1   # Obstacle

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
    num_episodes=1000,
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
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # we have to set the target network to eval mode
    # policy_net.train()  # we have to set the policy network to train mode

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate) # parameters are w, b of policy_net

    def epsilon(step):
        return max(epsilon_end, epsilon_start*(0.99**(step/epsilon_decay))) # exponential decay
    
    all_rewards = []
    step_count = 0

    for episode in range(num_episodes):
        state = env.reset() # putting the agent in the start state
        done = False
        episode_reward = 0

        while not done:
            step_count += 1
            epsilon_value = epsilon(step_count)
            if np.random.rand() < epsilon_value:
                action = np.random.randint(0, 4) # random action
                # print("Exploring")
            else:
                # use the policy network to get the action
                state_tensor = torch.FloatTensor([state[0], state[1]]) # state 0 and 1 are row and column
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
                # print("Exploiting")
            
            next_state, reward, done = env.step(action) # take the action
            replay_buffer.push(state, action, reward, next_state, done) # store the transition in the replay buffer
            episode_reward += reward
            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size:
                # sample minibatch from the replay buffer
                states, actions, rewards, next_states, done_sample = replay_buffer.sample(batch_size)
                # converting to tensors
                states = torch.FloatTensor([[s[0], s[1]] for s in states])
                actions = torch.LongTensor(actions).unsqueeze(-1) # shape = (batch_size,1)
                rewards = torch.FloatTensor(rewards).unsqueeze(-1)
                next_states = torch.FloatTensor([[ns[0], ns[1]] for ns in next_states])
                done_sample = torch.LongTensor(done_sample).unsqueeze(-1)

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
            if step_count%target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(episode_reward)

        # if(episode + 1)%50==0:
        print(f"Episode {episode+1}/{num_episodes}, Epsilon: {epsilon}, Episode Reward: {episode_reward}")

    return policy_net, target_net, all_rewards

if __name__ == "__main__":
    policy_net, target_net, rewards = train_dqn(num_episodes=1000)
    # Plotting the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.show()


                
        
