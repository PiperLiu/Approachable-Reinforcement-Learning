import numpy as np
import matplotlib.pyplot as plt

class KB_Game:
    def __init__(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.current_cumulative_rewards = 0.0
        self.actions = [1, 2, 3]
        self.counts = 0
        self.counts_history = []
        self.cumulative_rewards_history = []
        self.average_rewards_history = []
        self.a = 1
        self.reward = 0
    
    def step(self, a):
        r = 0
        if a == 1:
            r = np.random.normal(1, 1)
        elif a == 2:
            r = np.random.normal(2, 1)
        elif a == 3:
            r = np.random.normal(1.5, 1)
        return r
    
    def choose_action(self, policy, **kwargs):
        action = 0
        if policy == 'e_greedy':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.randint(1, 4)
            else:
                action = np.argmax(self.q) + 1
        if policy == 'ucb':
            c_ratio = kwargs['c_ratio']
            if 0 in self.action_counts:
                action = np.where(self.action_counts==0)[0][0] + 1
            else:
                value = self.q + c_ratio * np.sqrt(np.log(self.counts) /\
                    self.action_counts)
                action = np.argmax(value) + 1
        if policy == 'boltzmann':
            tau = kwargs['temperature']
            p = np.exp(self.q / tau) / (np.sum(np.exp(self.q / tau)))
            action = np.random.choice([1, 2, 3], p=p.ravel())
        return action
    
    def train(self, play_total, policy, **kwargs):
        reward_1 = []
        reward_2 = []
        reward_3 = []
        for i in range(play_total):
            action = 0
            if policy == 'e_greedy':
                action = \
                    self.choose_action(policy, epsilon=kwargs['epsilon'])
            if policy == 'ucb':
                action = \
                    self.choose_action(policy, c_ratio=kwargs['c_ratio'])
            if policy == 'boltzmann':
                action = \
                    self.choose_action(policy, temperature=kwargs['temperature'])
            self.a = action

            self.r = self.step(self.a)
            self.counts +=1

            self.q[self.a-1] = (self.q[self.a-1] * self.action_counts[self.a-1] + self.r) /\
                (self.action_counts[self.a-1] + 1)
            self.action_counts[self.a-1] += 1

            reward_1.append(self.q[0])
            reward_2.append(self.q[1])
            reward_3.append(self.q[2])
            self.current_cumulative_rewards += self.r
            self.cumulative_rewards_history.append(self.current_cumulative_rewards)
            self.average_rewards_history.append(self.current_cumulative_rewards /\
                (len(self.average_rewards_history) + 1))
            self.counts_history.append(i)

    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.current_cumulative_rewards = 0.0
        self.actions = [1, 2, 3]
        self.counts = 0
        self.counts_history = []
        self.cumulative_rewards_history = []
        self.average_rewards_history = []
        self.a = 1
        self.reward = 0
    
    def plot(self, colors, policy, style):
        plt.figure(1)
        plt.plot(self.counts_history, self.average_rewards_history, colors+style, label=policy)
        plt.legend()
        plt.xlabel('n', fontsize=18)
        plt.ylabel('average rewards', fontsize=18)
