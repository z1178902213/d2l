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
