import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuan_yang_env_fa import YuanYangEnv


class LFA_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        # 基于特征表示的参数
        self.theta_tr = np.zeros((400, 1)) * 0.1
        # 基于固定稀疏所对应的参数
        self.theta_fsr = np.zeros((80, 1)) * 0.1
    
    def feature_tr(self, s, a):
        phi_s_a = np.zeros((1, 400))
        phi_s_a[0, 100 * a + s] = 1
        return phi_s_a
    
    # 定义贪婪策略
    def greedy_policy_tr(self, state):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        return self.yuanyang.actions[amax]
    
    # 定义 epsilon 贪婪策略
    def epsilon_greedy_policy_tr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[
                int(random.random() * len(self.yuanyang.actions))
            ]

    # 找到动作所对应的序号
    def find_anum(self, a):
        for i in range(len(self.yuanyang.actions)):
            if a == self.yuanyang.actions[i]:
                return i
    
    def qlearning_lfa_tr(self, num_iter, alpha, epsilon):
        iter_num = []
        self.theta_tr = np.zeros((400, 1)) * 0.1
        # 第1个大循环，产生了多少实验
        for iter in range(num_iter):
            # 随机初始化状态
            epsilon = epsilon * 0.99
            s_sample = []
            # 初始状态
            s = 0
            flag = self.greedy_test_tr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('sarsa 第 1 次完成任务需要的迭代次数为 {}'.format(iter_num[0]))
            if flag == 2:
                print('sarsa 第 1 次实现最短路径需要的迭代次数为 {}'.format(iter))
                break
            # 利用 epsilon-greedy 策略选初始动作
            a = self.epsilon_greedy_policy_tr(s, epsilon)
            t = False
            count = 0

            # 第 2 个循环， 1 个实验， s0-s1-s2-s1-s2-s_terminate
            while t == False and count < 30:
                # 与环境交互得到下一状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 本轨迹中已有，给出负回报
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                # 判断是否是终止状态
                if t == True:
                    q_target = r
                else:
                    # 下一状态处的最大动作，体现同轨策略
                    a1 = self.greedy_policy_tr(s_next)
                    a1_num = self.find_anum(a1)
                    # Q-learning 得到时间差分目标
                    q_target = r + self.gamma * np.dot(self.feature_tr(s_next, a1_num), self.theta_tr)
                # 利用梯度下降的方法对参数进行学习
                self.theta_tr = self.theta_tr + alpha * (q_target - np.dot(self.feature_tr(s, a_num), self.theta_tr))[0, 0] * np.transpose(self.feature_tr(s, a_num))
                # 转到下一状态
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy_tr(s, epsilon)
                count += 1
        
        return self.theta_tr

    def greedy_test_tr(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while done == False and step_num < 30:
            a = self.greedy_policy_tr(s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        
        if s == 9:
            flag = 1
        if s == 9 and step_num < 21:
            flag = 2
        
        return flag

    def feature_fsr(self, s, a):
        phi_s_a = np.zeros((1, 80))
        y = int(s / 10)
        x = s - 10 * y
        phi_s_a[0, 20 * a + x] = 1
        phi_s_a[0, 20 * a + 10 + y] = 1
        return phi_s_a
    
    def greedy_policy_fsr(self, state):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数
        for i in range(4):
            qfun[i] = np.dot(self.feature_fsr(state, i), self.theta_fsr)
        amax = qfun.argmax()
        return self.yuanyang.actions[amax]
    
    # 定义 epsilon 贪婪策略
    def epsilon_greedy_policy_fsr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数
        for i in range(4):
            qfun[i] = np.dot(self.feature_fsr(state, i), self.theta_fsr)
        amax = qfun.argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[
                int(random.random() * len(self.yuanyang.actions))
            ]
    
    def greedy_test_fsr(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while done == False and step_num < 30:
            a = self.greedy_policy_fsr(s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        
        if s == 9:
            flag = 1
        if s == 9 and step_num < 21:
            flag = 2
        
        return flag
    
    def qlearning_lfa_fsr(self, num_iter, alpha, epsilon):
        iter_num = []
        self.theta_tr = np.zeros((80, 1)) * 0.1
        # 第1个大循环，产生了多少实验
        for iter in range(num_iter):
            # 随机初始化状态
            epsilon = epsilon * 0.99
            s_sample = []
            # 初始状态
            s = 0
            flag = self.greedy_test_fsr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('sarsa 第 1 次完成任务需要的迭代次数为 {}'.format(iter_num[0]))
            if flag == 2:
                print('sarsa 第 1 次实现最短路径需要的迭代次数为 {}'.format(iter))
                break
            # 利用 epsilon-greedy 策略选初始动作
            a = self.epsilon_greedy_policy_fsr(s, epsilon)
            t = False
            count = 0

            # 第 2 个循环， 1 个实验， s0-s1-s2-s1-s2-s_terminate
            while t == False and count < 30:
                # 与环境交互得到下一状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 本轨迹中已有，给出负回报
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                # 判断是否是终止状态
                if t == True:
                    q_target = r
                else:
                    # 下一状态处的最大动作，体现同轨策略
                    a1 = self.greedy_policy_fsr(s_next)
                    a1_num = self.find_anum(a1)
                    # Q-learning 得到时间差分目标
                    q_target = r + self.gamma * np.dot(self.feature_fsr(s_next, a1_num), self.theta_fsr)
                # 利用梯度下降的方法对参数进行学习
                self.theta_fsr = self.theta_fsr + alpha * (q_target - np.dot(self.feature_fsr(s, a_num), self.theta_fsr))[0, 0] * np.transpose(self.feature_fsr(s, a_num))
                # 转到下一状态
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy_fsr(s, epsilon)
                count += 1
        
        return self.theta_fsr
    

def qlearning_lfa_tr():
    yuanyang = YuanYangEnv()
    brain = LFA_RL(yuanyang)
    brain.qlearning_lfa_tr(num_iter=5000, alpha=0.1, epsilon=0.8)

    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    qvalue1 = np.zeros((100, 4))
    for i in range(400):
        y = int(i / 100)
        x = i - 100 * y
        qvalue1[x, y] = np.dot(brain.feature_tr(x, y), brain.theta_tr)
    yuanyang.action_value = qvalue1
    step_num = 0

    # 将最优路径打印出来
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy_tr(s)
        print('%d->%s\t'%(s, a), qvalue1[s, 0], qvalue1[s, 1], qvalue1[s, 2], qvalue1[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.1)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()

def qlearning_lfa_fsr():
    yuanyang = YuanYangEnv()
    brain = LFA_RL(yuanyang)
    qvalue2 = brain.qlearning_lfa_fsr(num_iter=5000, alpha=0.1, epsilon=0.1)

    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    qvalue2 = np.zeros((100, 4))
    for i in range(400):
        y = int(i / 100)
        x = i - 100 * y
        qvalue2[x, y] = np.dot(brain.feature_fsr(x, y), brain.theta_fsr)
    yuanyang.action_value = qvalue2
    step_num = 0

    # 将最优路径打印出来
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy_fsr(s)
        print('%d->%s\t'%(s, a), qvalue2[s, 0], qvalue2[s, 1], qvalue2[s, 2], qvalue2[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.1)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()


if __name__ == "__main__":
    # qlearning_lfa_tr()
    qlearning_lfa_fsr()