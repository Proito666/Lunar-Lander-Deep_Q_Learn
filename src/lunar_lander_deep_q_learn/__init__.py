import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Initialise the environment
env = gym.make("LunarLander-v3")

episode = 50000
learning_rate=0.001
reward_decay=0.99
e_greedy=0.99
e_greedy_increment=0.0005
epsilon=0.
replace_target_iter=1000
memory_size=10000
batch_size=64
train_freq=5
model_path = "model/dqn_eval_model.pth"
render_threshold = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        self.cost_his = []

        # consist of [target_net, evaluate_net]
        self.eval_net = DQNNet(n_features, n_actions).to(device)
        self.target_net = DQNNet(n_features, n_actions).to(device)
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)
        print(f"Eval model saved to {path}")

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.eval_net.state_dict())

        print(f"Eval model loaded from {path} and target_model synced")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(obs_tensor)
            action = torch.argmax(actions_value).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 更新 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("\ntarget_params_replaced\n")
        # sample batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s_batch = torch.FloatTensor(batch_memory[:, :self.n_features]).to(device)
        a_batch = torch.LongTensor(batch_memory[:, self.n_features]).unsqueeze(1).to(device)
        r_batch = torch.FloatTensor(batch_memory[:, self.n_features + 1]).to(device)
        s_next_batch = torch.FloatTensor(batch_memory[:, -self.n_features:]).to(device)

        q_eval = self.eval_net(s_batch).gather(1, a_batch).squeeze()
        with torch.no_grad():
            q_next = self.target_net(s_next_batch)
            q_target = r_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())
        self.learn_step_counter += 1

    def update_epsilon(self):
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
      

if __name__ == "__main__":
    RL = DeepQNetwork(env.action_space.n, env.observation_space.shape[0])
    render_enabled = False
    # 确保 model 目录存在
    os.makedirs("model", exist_ok=True)
    
    # 尝试加载已有模型
    if os.path.exists(model_path):
        RL.load_model(model_path)
        print("Loaded existing model, continuing training...")
    else:
        print("No existing model found, starting training from scratch.")

    all_ep_r = []

    for ep in range(episode):
        # Reset the environment to generate the first observation
        observation, info = env.reset()
        step = 0
        ep_r = 0
        while True:
            # this is where you would insert your policy
            action = RL.choose_action(observation)

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            observation_, reward, terminated, truncated, info = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # If the episode has ended then we can reset to start a new episode

            if RL.memory_counter > RL.batch_size and (step % train_freq  == 0):
                RL.learn()

            # swap observation
            observation = observation_
            ep_r += reward
            
            if terminated or truncated:
                break
            step += 1
         # 计算滑动平均奖励
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            f"Ep: {ep}",
            f"| Ep_r: {ep_r:.2f}",
            f"| Avg_r: {all_ep_r[-1]:.2f}",
            f"| epsilon: {RL.epsilon:.4f}"
        )
        if not render_enabled and all_ep_r[-1] >= render_threshold:
            print(f"Avg_r >= {render_threshold}, enabling rendering...")
            env.close()
            env = gym.make("LunarLander-v3", render_mode="human")
            render_enabled = True

        RL.update_epsilon()
        # 每 N 个 episode 保存一次模型，或者根据需要保存
        if ep % 10 == 0:
            RL.save_model(model_path)
    env.close()
    RL.plot_cost()
    # 最终保存模型
    RL.save_model(model_path)