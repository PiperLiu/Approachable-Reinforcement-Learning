import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        在 __init__() 中并没有真正定义网络结构的关系
        输入输出关系在 forward() 中定义
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        '''
        注意， torch.nn 中要求输入的数据是一个 mini-batch ，
        由于数据图像是 3 维的，因此输入数据 x 是 4 维的
        因此进入全连接层前 x.view(-1, ...) 方法化为 2 维
        '''
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

def seeNet():
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print("第一层的 weight ：")
    print(params[0].size())
    print(net.conv1.weight.size())
    print("第一层的 bias ：")
    print(params[1].size())
    print(net.conv1.bias.size())
    print("")
    print(net.conv1.weight.requires_grad)

    '''
    神经网络的输入输出应该是 Variable
    '''
    inputV = Variable(torch.rand(1, 3, 32, 32))
    # net.__call__()
    output = net(inputV)
    print(output)

# seeNet()

# train
def trainNet():
    net = Net()
    inputV = Variable(torch.rand(1, 3, 32, 32))
    output = net(inputV)
    
    criterion = nn.CrossEntropyLoss()
    label = Variable(torch.LongTensor([4]))
    loss = criterion(output, label)
    # all data type should be 'Variable'
    print(loss)

    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print("\nbefore optim: {}".format(net.conv1.bias))

    optimizer.zero_grad()  # zeros the gradient buffers
    loss.backward()
    optimizer.step()  # Does the update
    '''
    参数有变化，但可能很小
    '''
    print("\nafter optim: {}".format(net.conv1.bias))

# trainNet()

# 实战：CIFAR-10
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
    root='./pyTorch_learn/data',
    train=True,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # windows 下线程参数设为 0 安全
)

testset = torchvision.datasets.CIFAR10(
    root='./pyTorch_learn/data',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0  # windows 下线程参数设为 0 安全
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

def cifar_10():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 6000 == 5999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 6000))
                running_loss = 0.0
    
    print('Finished Training')
    torch.save(net.state_dict(), './pyTorch_learn/data/' + 'model.pt')
    net.load_state_dict(torch.load('./pyTorch_learn/data/' + 'model.pt'))

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        # 返回可能性最大的索引 -> 输出标签
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total
    ))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):  # mini-batch's size = 4
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]
        ))
    
    # save net
    print(net.state_dict().keys())
    print(net.state_dict()['conv1.bias'])

    # torch.save(net.state_dict(), 'model.pt')
    # net.load_state_dict(torch.load('model.pt'))
    
cifar_10()