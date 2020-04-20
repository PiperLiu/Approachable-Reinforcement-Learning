import random
import time
from yuan_yang_env import YuanYangEnv

class DP_Value_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states) + 1)]
        self.pi = dict()
        self.yuanyang = yuanyang

        self.gamma = yuanyang.gamma
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]
    
    def value_iteration(self):
        for i in range(1000):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 2:
                    continue
                a1 = self.actions[int(random.random() * 4)]
                s, r, t = self.yuanyang.transform(state, a1)
                # 策略评估
                v1 = r + self.gamma * self.v[s]
                # 策略改善
                for action in self.actions:
                    s, r, t = self.yuanyang.transform(state, action)
                    if v1 < r + self.gamma * self.v[s]:
                        a1 = action
                        v1 = r + self.gamma * self.v[s]
                delta += abs(v1 - self.v[state])
                self.pi[state] = a1
                self.v[state] = v1
            if delta < 1e-6:
                print('迭代次数为：{}'.format(i))
                break


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = DP_Value_Iter(yuanyang)
    policy_value.value_iteration()

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
