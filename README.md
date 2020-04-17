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
