# 笔记：《深入浅出强化学习：编程实战》

> 因为这本书 2019 年 11 月刚刚出版，我就先不公开我的笔记了（make it private）。

## 前言
作者是南开大学讲师，郭宪。
知道这本书是在知乎上，阅读强化学习文章时，发现其总来自一个账号（[@天津包子馅儿](https://www.zhihu.com/people/guoxiansia)），就是郭老师。
学校有活动：“你买书、学校掏钱。”在书市选书，免费寄到我家，开学后带到图书馆归还就可。
那正好有这本书。

粗略看了一下，蛮不错的。排版可能有些粗糙，但“一段代码、一段解读”是我喜欢的风格。有了代码，就不会云里雾里。

## 目录

- [附录 A PyTorch 入门](#A)
- 第 0 篇 先导篇
- - [1 一个及其简单的强化学习实例](#sec_1)
- - [2 马尔可夫决策过程](#sec_2)
- 第 1 篇 基于值函数的方法
- - [3 基于动态规划的方法](#sec_3)
- - [4 基于蒙特卡洛的方法](#sec_4)
- - [5 基于时间差分的方法](#sec_5)
- - [6 基于函数逼近的方法](#sec_6)
- 第 2 篇 直接策略搜索方法
- - [7 策略梯度方法](#sec_7)
- - [8 Actor-Critic 方法](#sec_8)
- - [9 PPO 方法](#sec_9)
- - [10 DDPG方法](#sec_10)
- 第 3 篇 基于模型的强化学习方法
- - [11 基于模型预测控制的强化学习方法](#sec_11)
- - [12 AlphaZero 原理浅析](#sec_12)
- - [13 AlphaZero 实战：从零学下五子棋](#sec_13)

#### 附录 A PyTorch 入门
<a id='A'></a>

[./pyTorch_learn/](./pyTorch_learn/)

介绍了 PyTorch 的基本使用，主要实例为：构建了一个极其单薄简单的卷积神经网络，数据集为 `CIFAR10` 。学习完附录 A ，我主要收获：
- 输入 PyTorch 的 `nn.Module` 应该是 `mini-batch` ，即比正常数据多一个维度；
- 输入 `nn.Module` 应该是 `Variable` 包裹的；
- 在网络类中， `__init__()` 并没有真正定义网络结构的关系，网络结构的输入输出关系在 `forward()` 中定义。

此外，让我们梳理一下神经网络的“学习”过程：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variabl
import torch.optim as optim

# 神经网络对象
net = Net()
# 损失函数：交叉熵
criterion = nn.CrossEntropyLoss()
# 优化方式
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 遍历所有数据 5 次
for epoch in range(5):
    # data 是一个 mini-batch ， batch-size 由 trainloader 决定
    for i, data in enumerate(trainloader, 0):
        # 特征值与标签：注意用 Variable 进行包裹
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # pytorch 中 backward() 函数执行时，梯度是累积计算，
        # 而不是被替换；
        # 但在处理每一个 batch 时并不需要与其他 batch 的梯度
        # 混合起来累积计算，
        # 因此需要对每个 batch 调用一遍 zero_grad()
        # 将参数梯度置0。
        optimizer.zero_grad()
        # 现在的输出值
        outputs = net(inputs)
        # 求误差
        loss = criterion(outputs, labels)
        # 在初始化 optimizer 时,我们明确告诉它应该更新模型的
        # 哪些参数；一旦调用了 loss.backward() ，梯度就会被
        # torch对象“存储”（它们具有grad和requires_grad属性）；
        # 在计算模型中所有张量的梯度后，调用 optimizer.step()
        # 会使优化器迭代它应该更新的所有参数（张量），
        # 并使用它们内部存储的 grad 来更新它们的值。
        loss.backward()
        optimizer.step()

# 模型的保存与加载
torch.save(net.state_dict(), './pyTorch_learn/data/' + 'model.pt')
net.load_state_dict(torch.load('./pyTorch_learn/data/' + 'model.pt'))
```

#### 第 0 篇 先导篇

##### 1 一个及其简单的强化学习实例
<a id='sec_1'></a>

[./ch_0/sec_1/](./ch_0/sec_1/)

很简答的一个实例，探讨“探索”与“利用”间的博弈平衡。

我对原书的代码进行了一些改进与 typo 。

##### 2 马尔可夫决策过程
<a id='sec_2'></a>

[./ch_0/sec_2/](./ch_0/sec_2/)

优势函数（Advantage Function）：$A(s, a) = q_\pi (s, a) - v_\pi (s)$

造了一个交互环境，以后测试可以用到。典型的“网格世界”。

#### 第 1 篇 基于值函数的方法

##### 3 基于动态规划的方法
<a id='sec_3'></a>

[./ch_1/sec_3/](./ch_1/sec_3/)

- 修改了“鸳鸯环境”：[yuan_yang_env.py](./ch_1/sec_3/yuan_yang_env.py)
- 策略迭代：[dp_policy_iter.py](./ch_1/sec_3/dp_policy_iter.py)
- 价值迭代：[dp_value_iter.py](./ch_1/sec_3/dp_value_iter.py)

##### 4 基于蒙特卡洛的方法
<a id='sec_4'></a>

[./ch_1/sec_4/](./ch_1/sec_4/)

基于蒙特卡洛方法，分别实现了：
- 纯贪心策略；
- 同轨策略下的 Epsilon-贪心策略。

##### 5 基于时间差分的方法
<a id='sec_5'></a>

[./ch_1/sec_5/](./ch_1/sec_5/)

本书让我对“离轨策略”的概念更加清晰：
- 离轨策略可以使用重复的数据、不同策略下产生的数据；
- 因为，离轨策略更新时，只用到了(s, a, r, s')，并不需要使用 a' 这个数据；
- 换句话说，并不需要数据由本策略产生。

**发现了一个神奇的现象：关于奖励机制的设置。**

书上说，本节使用的 env 可以与蒙特卡洛同，这是不对的。
先说上节蒙特卡洛的奖励机制：
- 小鸟撞到墙：-10；
- 小鸟到达目的地：+10；
- 小鸟走一步，什么都没有发生：-2。

如果你运行试验，你会发现，TD(0) 方法下，小鸟会 **畏惧前行** ：
- 在蒙特卡洛方法下，如此的奖励机制有效的，因为训练是在整整一幕结束之后；
- 在 TD(0) 方法下，小鸟状态墙得到的奖励是 -10 ，而它行走五步的奖励也是 -10 （不考虑折扣）；
- 但在这个环境中，小鸟要抵达目的地，至少要走 20 步；
- 因此，与其“披荆斩棘走到目的地”，还不如“在一开始就一头撞在墙上撞死”来的奖励多呢。

可以用如下几个例子印证，对于 [./ch_1/sec_5/yuan_yang_env_td.py](./ch_1/sec_5/yuan_yang_env_td.py) 的第 151-159 行：
```python
flag_collide = self.collide(next_position)
        if flag_collide == 1:
            return self.position_to_state(current_position), -10, True
        
        flag_find = self.find(next_position)
        if flag_find == 1:
            return self.position_to_state(next_position), 10, True
        
        return self.position_to_state(next_position), -2, False
```

现在 (撞墙奖励, 到达重点奖励, 走路奖励) 分别为 (-10, 10, -2) 。现在的实验结果是：小鸟宁可撞死，也不出门。

我们把出门走路的痛感与抵达目的地的快乐进行更改，：
- (-10, 10, -1)，出门走路没有那么疼了，小鸟倾向于抵达目的地；
- (-10, 10, -1.2)，出门走路的痛感上升，小鸟倾向于宁可开局就撞死自己；
- (-10, 100, -1.2)，出门走路虽然疼，但是到达目的地的快乐是很大很大的，小鸟多次尝试，掐指一算，还是出门合适，毕竟它是一只深谋远虑的鸟。

运行试验，我们发现，上述试验并不稳定，这是因为每次训练小鸟做出的随机决策不同，且“走路痛感”与“抵达快乐”很不悬殊，小鸟左右为难。

因此，一个好的解决方案是：**别管那么多！我们不是希望小鸟抵达目的地吗？那就不要让它走路感到疼痛！**
- 设置奖励为(-10, 10, 0)，我保证小鸟会“愿意出门”，并抵达目的地！

这也提醒了我：**作为一个强化学习算法实践者，不应该过多干涉智能体决策！告诉智能体几个简单的规则，剩下的交给其自己学习。过于复杂的奖励机制，会出现意想不到的状况！**

对于此问题，其实还可以另一个思路：调高撞墙的痛感。
- 设置奖励为(-1000, 10, -2)，如次，小鸟便不敢撞墙：因为撞墙太疼了！！！
- 并且，小鸟也不会“多走路”，因为多走路也有痛感。你会发现，如此得到的结果，小鸟总是能找到最优方案（最短的路径）。

##### 6 基于函数逼近的方法
<a id='sec_6'></a>

[./ch_1/sec_6/](./ch_1/sec_6/)

本节前半部分最后一次使用“鸳鸯系统”，我发现：
- 无论是正常向量状态表示，还是固定稀疏状态表示，书中都将 `epsilon = epsilon * 0.99` 在迭代中去掉；
- 事实证明，不应该将其去掉，尤其是第一组情况。第一组情况其实就是上节的表格型 q-learning ；
- 固定稀疏表示中，可以不加探索欲望的收敛（在这个环境中）。

此外，还发现：
- 固定稀疏中，鸟倾向于走直线；
- 我认为这是因为固定稀疏矩阵中，抽取了特征，同一个 x 或同一个 y 对应的状态，其价值更趋同。

本节后半部分：非线性函数逼近。

书中没有给代码地址，我 Google 到作者应该是借鉴了这个：[https://github.com/yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
- 我将这个项目写在了：[./ch_1/sec_6/flappy_bird/](./ch_1/sec_6/flappy_bird/)
- - 我添加了手动操作体验游戏的部分，按 H 键可以煽动翅膀：[./ch_1/sec_6/flappy_bird/game/keyboard_agent.py](./ch_1/sec_6/flappy_bird/game/keyboard_agent.py)
- - 书上是 tf 1 的代码，我使用 tf 2 重写，这个过程中参考了：[https://github.com/tomjur/TF2.0DQN](https://github.com/tomjur/TF2.0DQN)
- `python -u "d:\GitHub\rl\Approachable-Reinforcement-Learning\ch_1\sec_6\flappy_bird\dqn_agent.py"`以训练

