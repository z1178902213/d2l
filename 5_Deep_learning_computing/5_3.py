import torch
import torch.nn.functional as F
from torch import nn


# 构造一个不含任何参数的层，继承nn.Module类
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
print(Y.mean())


# 定义具有参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # torch.randn可以生成对应维度的随机数据
        # 生成了随机参数给了weight
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
