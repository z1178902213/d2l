# 自动求导，导包
import torch


# 1、简单例子
# 介绍一些自动求导
x = torch.arange(4.0)
print(f'x的值为：{x}')
# 计算梯度前，需要一个地方存储梯度
x.requires_grad_(True) # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad) # 不懂这个x.grad是啥，默认值是None
# 计算y
y = 2 * torch.dot(x, x) #计算向量内积
# 自动计算y关于每个x分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4*x)
# 计算x的另一个函数
x.grad.zero_() # pytorch会积累梯度，因此计算新梯度前应先清除之前积累的梯度
y = x.sum()
y.backward()
print(x.grad)


# 2、非标量变量的反向传播
