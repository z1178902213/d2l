import torch
from torch import nn

# nn.Linear(4, 8)表示4个输入特征，8个输出特征，其余的同理
# nn.Linear通常是WX+b
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X= torch.rand(size=(2, 4))
# 直接print(net)可以查看网络结构，服
print(net)
print(net(X))
# 由于net在前面sequential设置时有3个模块，net从零开始访问，可以访问到net[2]
print(net[2].state_dict())

# 提取参数实例，以net[2]的参数为例
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# 可以访问每个参数的梯度
print(net[2].weight.grad == None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 另一种访问参数的方式，访问的就是net[2]的bias参数
print(net.state_dict()['2.bias'].data)



# 嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
# 查看网络的结构
print(rgnet)

print(rgnet[0][1][0].bias.data)


def init_normal(m):
    if type(m) == nn.Linear:
        # 权重以均值为0，方差为1随机生成
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        # 将所有参数初始化为1
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


# 对不同的块应用不同的初始方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
# 用xavier方法初始化第一层
net[0].apply(xavier)
# 用init_42的方法初始化第三层的线性回归
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# 自定义了一个参数初始化方法
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# ？这怎么还能把方法作为参数传入？什么牛逼操作啊？
net.apply(my_init)
print(net[0].weight[:2])

# 直接设置参数是一直都可以的
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]


# 参数绑定
# 设置了一个共享层，在网络中使用该共享层时，所有使用了所有该共享层的
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
print(net)
# 检查改变了参数之后，是否共享层的值不变
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

