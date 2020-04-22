import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuan_yang_env_td import YuanYangEnv


class TD_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        # 值函数的初始值
        self.qvalue = np.zeros(
            (len(self.yuanyang.states), len(self.yuanyang.actions))
        )
    
    # 定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.yuanyang.actions[amax]
    
    # 定义 epsilon 贪婪策略
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
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
    
    def sarsa(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        # 第1个大循环，产生了多少实验
        for iter in range(num_iter):
            # 随机初始化状态
            epsilon = epsilon * 0.99
            s_sample = []
            # 初始状态
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('sarsa 第 1 次完成任务需要的迭代次数为 {}'.format(iter_num[0]))
            if flag == 2:
                print('sarsa 第 1 次实现最短路径需要的迭代次数为 {}'.format(iter))
                break
            # 利用 epsilon-greedy 策略选初始动作
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
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
                    a1 = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(a1)
                    # Q-learning 的更新公式（SARSA）
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                # 利用 td 方法更新动作值函数 alpha
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                # 转到下一状态
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        
        return self.qvalue

    def greedy_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while done == False and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
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

    def qlearning(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        # 第1个大循环，产生了多少实验
        for iter in range(num_iter):
            # 随机初始化状态
            epsilon = epsilon * 0.99
            s_sample = []
            # 初始状态
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('q-learning 第 1 次完成任务需要的迭代次数为 {}'.format(iter_num[0]))
            if flag == 2:
                print('q-learning 第 1 次实现最短路径需要的迭代次数为 {}'.format(iter))
                break
            # 随机选取初始动作
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
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
                    a1 = self.greedy_policy(self.qvalue, s_next)
                    a1_num = self.find_anum(a1)
                    # Q-learning 的更新公式 TD(0)
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                # 利用 td 方法更新动作值函数 alpha
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                # 转到下一状态
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        
        return self.qvalue

def sarsa():
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    qvalue1 = brain.sarsa(num_iter=5000, alpha=0.1, epsilon=0.8)

    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    yuanyang.action_value = qvalue1
    step_num = 0

    # 将最优路径打印出来
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue1, s)
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

def qlearning():
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    qvalue2 = brain.qlearning(num_iter=5000, alpha=0.1, epsilon=0.1)

    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    yuanyang.action_value = qvalue2
    step_num = 0

    # 将最优路径打印出来
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue2, s)
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
    # sarsa()
    qlearning()