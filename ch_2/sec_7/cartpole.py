"""
class Sample
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras import initializers
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

RENDER = True

class Sample:
    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net
        self.gamma = 0.98
    
    def sample_episodes(self, num_episodes):
        # 产生 num_episodes 条轨迹
        batch_obs = []
        batch_actions = []
        batch_rs = []
        for i in range(num_episodes):
            observation = self.env.reset()
            # 将一个 episode 的回报存储起来
            reward_episode = []
            while True:
                if RENDER:
                    self.env.render()
                # 根据策略网络产生一个动作
                state = np.reshape(observation, [1, 4])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                batch_obs.append(observation)
                batch_actions.append(action)
                reward_episode.append(reward)
                # 一个 episode 结束
                if done:
                    # 处理回报函数
                    reward_sum = 0
                    discounted_sum_reward = np.zeros_like(reward_episode)
                    for t in reversed(range(0, len(reward_episode))):
                        reward_sum = reward_sum * self.gamma + reward_episode[t]
                        discounted_sum_reward[t] = reward_sum
                    # 归一化处理
                    discounted_sum_reward -= np.mean(discounted_sum_reward)
                    discounted_sum_reward /= np.std(discounted_sum_reward)
                    # 将归一化的数据存储到批回报中
                    for t in range(len(reward_episode)):
                        batch_rs.append(discounted_sum_reward[t])
                    break
                # 智能体往前推进一步
                observation = observation_
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions, [len(batch_actions), ])
        batch_rs = np.reshape(batch_rs, [len(batch_rs), ])
        return batch_obs, batch_actions, batch_rs


class Policy_Net:
    def __init__(self, env, model_file=None):
        self.learning_rate = 0.01
        # 输入特征的维数
        self.n_features = env.observation_space.shape[0]
        # 输出动作空间的维数
        self.n_actions = env.action_space.n
        self.model = self._build_q_net()
    
    def _build_q_net(self):
        model = keras.Sequential()
        self.f1 = kl.Dense(
            units=20,
            input_shape=(self.n_features, ),
            activation='relu',
            kernel_initializer=initializers.random_normal(mean=0, stddev=0.1),
            bias_initializer=initializers.constant_initializer(0.1)
        )
        self.all_act = kl.Dense(
            units=self.n_actions,
            activation=None,
            kernel_initializer=initializers.random_normal(mean=0, stddev=0.1),
            bias_initializer=initializers.constant_initializer(0.1)
        )
        self.all_act_prob = kl.Activation('softmax')
        model.add(self.f1)
        model.add(self.all_act)
        model.add(self.all_act_prob)

        def _sparse_softmax_cross_entropy_with_logits(self, target):
            pass

'''
书上的代码可读性很差，参考：
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
'''