# a.1.1
import torch
x = torch.Tensor(2, 3)
print(x)

x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(x)

x = torch.rand(2, 3)
print(x)

x = torch.zeros(2, 3)
print(x)

x = torch.ones(2, 3)
print(x)

print("")
print("x.size: {}".format(x.size()))
print("x.size[0]: {}".format(x.size()[0]))

# a.1.2
x = torch.ones(2, 3)
y = torch.ones(2, 3) * 2
print("")
print("a.1.2")
print("x + y = {}".format(x + y))

print(torch.add(x, y))

x.add_(y)
print(x)
print("注意：PyTroch中修改 Tensor 内容的操作\
        都会在方法名后加一个下划线，如copy_()、t_()等")

print("x.zero_(): {}".format(x.zero_()))
print("x: {}".format(x))

print("")
print("Tensor 也支持 NumPy 中的各种切片操作")

x[:, 1] = x[:, 1] + 2
print(x)

print("torch.view()相当于numpy中的reshape()")
print("x.view(1, 6): {}".format(x.view(1, 6)))

# a.1.3
print("")
print("a.1.3")

print("Tensor 与 NumPy 的 array 可以转化，但是共享地址")
import numpy as np
x = torch.ones(2, 3)
print(x)

y = x.numpy()
print("y = x.numpy(): {}".format(y))

print("x.add_(2): {}".format(x.add_(2)))

print("y: {}".format(y))

z = torch.from_numpy(y)
print("z = torch.from_numpy(y): {}".format(z))

# a.1.4
print("")
print("a.1.4")

print("Autograd 实现自动梯度")

from torch.autograd import Variable

x = Variable(torch.ones(2, 2)*2, requires_grad=True)

print(x)

print("x.data {}, \n x's type: {}\n".format(x.data, type(x)))

y = 2 * (x * x) + 5 * x
y = y.sum()
print("y: {}, \n y's type: {}\n".format(y, type(y))) 

print("y 可视为关于 x 的函数")
print("y 应该是一个标量，y.backward()自动计算梯度")

y.backward()
print("x.grad: {}".format(x.grad))
print("x.grad 中自动保存梯度")
