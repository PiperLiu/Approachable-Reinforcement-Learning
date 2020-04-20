import random
import time
from yuan_yang_env import YuanYangEnv


class DP_Policy_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states) + 1)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

        # 初始化策略
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]

    def policy_evaluate(self):
        # 策略评估在计算值函数
        for i in range(100):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                action = self.pi[state]
                s, r, t = self.yuanyang.transform(state, action)
                # 更新值
                new_v = r + self.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                # 更新值替换原来的值函数
                self.v[state] = new_v
            if delta < 1e-6:
                print('策略评估迭代次数：{}'.format(i))
                break
    
    def policy_improve(self):
        # 利用更新后的值函数进行策略改善
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
            flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            a1 = self.actions[0]
            s, r, t = self.yuanyang.transform(state, a1)
            v1 = r + self.gamma * self.v[s]
            # 找状态 s 时，采用哪种动作，值函数最大
            for action in self.actions:
                s, r, t = self.yuanyang.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]
            # 贪婪策略，进行更新
            self.pi[state] = a1
    
    def policy_iterate(self):
        for i in range(100):
            # 策略评估，变的是 v
            self.policy_evaluate()
            # 策略改善
            pi_old = self.pi.copy()
            # 改变 pi
            self.policy_improve()
            if (self.pi == pi_old):
                print('策略改善次数：{}'.format(i))
                break


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = DP_Policy_Iter(yuanyang)
    policy_value.policy_iterate()

    # 打印
    flag = 1
    s = 0
    path = []
    # 将 v 值打印出来
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    
    step_num = 0

    # 将最优路径打印出来
    while flag:
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        print('%d->%s\t'%(s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
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
