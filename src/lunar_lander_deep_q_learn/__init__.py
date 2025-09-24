import os

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")


episode = 1000
learning_rate=0.01
reward_decay=0.9
e_greedy=0.99
e_greedy_increment=0.001
replace_target_iter=300
memory_size=500
batch_size=32
train_freq=1
model_path = "model/dqn_eval_model.keras"

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
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        self.cost_his = []

        # consist of [target_net, evaluate_net]
        self._build_net()
        
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

    def _build_net(self):
        # 初始化器
        w_initializer = tf.keras.initializers.RandomNormal(0., 0.3)
        b_initializer = tf.keras.initializers.Constant(0.1)

        # 输入保持原名
        self.s = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s')
        self.s_ = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s_')

        # evaluate_net
        e1 = Dense(20, activation='relu', kernel_initializer=w_initializer,
                bias_initializer=b_initializer, name='e1')(self.s)
        self.q_eval = Dense(self.n_actions, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='q')(e1)
        self.eval_model = tf.keras.Model(inputs=self.s, outputs=self.q_eval)

        # target_net
        t1 = Dense(20, activation='relu', kernel_initializer=w_initializer,
                bias_initializer=b_initializer, name='t1')(self.s_)
        self.q_next = Dense(self.n_actions, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='t2')(t1)
        self.target_model = tf.keras.Model(inputs=self.s_, outputs=self.q_next)

    def save_model(self, path="dqn_eval_model.h5"):
        """保存 eval_model 模型，包括权重和结构"""
        self.eval_model.save(path)
        print(f"Eval model saved to {path}")

    def load_model(self, path="dqn_eval_model.h5"):
        """加载 eval_model 模型"""
        self.eval_model = load_model(path)
        # 同步 target_model
        self.target_model.set_weights(self.eval_model.get_weights())
        print(f"Eval model loaded from {path} and target_model synced")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :].astype(np.float32)
        if np.random.uniform() < self.epsilon:
            q_eval_val = self.eval_model(observation).numpy()
            action = np.argmax(q_eval_val)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 更新 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
            print("\ntarget_params_replaced\n")
        # sample batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s_batch = batch_memory[:, :self.n_features].astype(np.float32)
        a_batch = batch_memory[:, self.n_features].astype(np.int32)
        r_batch = batch_memory[:, self.n_features + 1].astype(np.float32)
        s_next_batch = batch_memory[:, -self.n_features:].astype(np.float32)

        with tf.GradientTape() as tape:
            q_eval_val = self.eval_model(s_batch)
            q_next_val = self.target_model(s_next_batch)

            batch_indices = tf.range(q_eval_val.shape[0], dtype=tf.int32)
            indices = tf.stack([batch_indices, a_batch], axis=1)
            q_eval_wrt_a_val = tf.gather_nd(q_eval_val, indices)
            q_target_val = r_batch + self.gamma * tf.reduce_max(q_next_val, axis=1)
            loss_val = tf.reduce_mean(tf.square(q_target_val - q_eval_wrt_a_val))

        grads = tape.gradient(loss_val, self.eval_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.eval_model.trainable_variables))

        self.cost_his.append(loss_val.numpy())

        # update epsilon
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
      

if __name__ == "__main__":

    RL = DeepQNetwork(env.action_space.n, env.observation_space.shape[0])

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
        observation, info = env.reset(seed=42)
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
            f"| Avg_r: {all_ep_r[-1]:.2f}"
        )
        # 每 N 个 episode 保存一次模型，或者根据需要保存
        if ep % 10 == 0:
            RL.save_model(model_path)
    env.close()
    RL.plot_cost()
    # 最终保存模型
    RL.save_model(model_path)