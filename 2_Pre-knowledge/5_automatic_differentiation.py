# 自动求导，导包
import torch

##
#这些一直不理解各个变量之间的梯度关系
##


# 1、简单例子
# 介绍一些自动求导
x = torch.arange(4.0)
print(f'x的值为：{x}')
# 计算梯度前，需要一个地方存储梯度
x.requires_grad_(True) # 等价于 `x = torch.arange(4.0, requires_grad=True)`
print(x.grad) # 目前x没有梯度
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


# 2、非标量变量的反向传播(不就是函数求导吗？这有啥问题？)
# 对非标量调用`backward`需要传入一个`gradient`参数，该参数指定微分函数关于`self`的梯度。在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad == 2 * x)


# 3、分离计算
# 若y是x的函数，z是x,y的函数，若要计算z的梯度时视y为常数不做梯度计算，可以使用分离计算的技巧
x.grad.zero_()
y = x * x
print(y)
u = y.detach() # 相当于让y独立了，不对y进行梯度计算
z = u * x
z.sum().backward()   #########为什么要加sum()？
print(x.grad)  # 意思应该是输出关于x的梯度


# 4、Python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:   # b.norm()取范数
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True) # torch.randn从均值0方差1的正态分布中获取随机数
d = f(a)
d.backward()

print(a.grad == d / a)  ### 不理解。。。