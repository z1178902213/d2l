# 导包，导入pytorch包
import torch


# 1、张量tensor的简单介绍
# 张量可以理解为向量
# 返回一个从0开始的12个张量，间隔默认为1，类型默认为浮点数
a = torch.arange(12)
print('生成了一个张量 a = ',a)
# shape属性返回张量维度
print('张量a的维度为：',a.shape)
# numel()方法返回张量内部元素个数
print('张量a的元素个数为：',a.numel())
# reshape()方法可以调整张量的形状
# !WARNING：reshape()方法并不直接对张量进行操作，而是返回一个调整形状过后的张量
A = a.reshape(3,4)
print(f'调整后的a的形状是：{A.shape}\n调整后的a的输出样式是：\n{A}')
# 当要改变的维度为2维时，只要提供其中一个维度就能得到另一个维度了，因此reshape()的时候是不需要手动输入维度的
# 使用-1代替需要自动生成的维度，就可以达到自动生成维度的目的
A = a.reshape(-1,3)
print(f'执行a.reshape(-1,3)之后的结果：\n{A}')
# 全0、全1的张量
zeros = torch.zeros(5,5)
ones = torch.ones(5,5)
print('返回一个全0的5x5的张量\n',zeros)
print('返回一个全1的5x5的张量\n',ones)
# 返回随机数据的张量，该随机数张量中的数据从均值0，标准差1的正态分布中随机取值
a = torch.randn(3,4)
print(f'生成了一个3x4的随机数张量：\n{a}')
# 从python列表中创建一个张量
a = torch.tensor([[1,2,3],[4,5,6]])
print(f'从Python列表[[1,2,3],[4,5,6]中创建一个张量：\n{a}')
print('\n\n\n')


# 2、运算部分
# 常见的运算符如+、-、*、/、**（求幂）都被升级成了按元素运算符
x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[5,4],[3,2]])
print(f'x = {x}\ny = {y}')
print(f'x + y = {x+y}\nx - y = {x-y}\nx * y = {x*y}\nx / y = {x/y}\nx ** y = {x**y}')
# 一元运算，自然对数e的对应元素次方
print('e的几次方：',torch.exp(x))
# 将张量拼接形成更大的张量，dim表示要合并的维度，从最外层开始。
# 如要合并维度为(2,3,4)的两个张量，dim=0表示将2个3x4的张量和2个3x4的张量结合成4个3x4的张量。
zeros = torch.zeros((2,3,4))
ones = torch.ones((2,3,4))
d1 = torch.cat((zeros, ones), dim=0)
d2 = torch.cat((zeros, ones), dim=1)
d3 = torch.cat((zeros, ones), dim=2)
print(f'第0维合并：\n{d1}\n')
print(f'第1维合并：\n{d2}\n')
print(f'第2维合并：\n{d3}\n')
# 对两个张量逐元素的值进行对比，可以用>,<,==
print(zeros < ones)
# 对整个张量进行求值运算，同时，在tensor类型的数据后面加上一个item()可以将只有一个元素的张量转换成python的浮点数值
print('d3所有元素的和：', d3.sum().item())
print('\n\n\n')


# 3、python的广播机制
# 广播机制允许两个维度不相同的张量进行操作。具体做法是对缺失的维度进行相同元素的填充。
# WARNING：当拥有的维度不能完美填充缺失维度的时候将会报错。
a = torch.arange(3).reshape(3,1)
b = torch.arange(4).reshape(1,4)
print(a)
print(b)
print(a+b)
print('\n\n\n')


# 4、索引和切片
# 张量中的元素可以被像python列表元素一样索引和切片
a = torch.arange(8).reshape(2,4)
print(a)
print(a[:,-1])
# 可以通过索引直接将目标值写入矩阵
a[1,1] = 888
print(a)
# 用切片可以一次性赋值多个值
a[0,:] = 666
print(a,'\n\n\n')


# 5、节省内存
# 当执行Y = X + Y的操作时，会抛弃掉原有的Y的空间重新为Y分配空间，这样造成了内存资源的浪费
Y = torch.tensor([[1,2],[3,4]])
X = torch.tensor([[3,2],[1,0]])
before = id(Y)
print('原先Y的地址：',id(Y))
Y = X + Y
print(Y)
print('代码执行后Y的地址：',id(Y))
print('是否与原地址相同？',id(Y) == before)
# 此时就需要一个原地操作
Y = torch.tensor([[1,2],[3,4]])
X = torch.tensor([[3,2],[1,0]])
before = id(Y)
print('\n原先Y的地址：',id(Y))
Y[:] = X + Y
print(Y)
print('现在Y的地址：',id(Y))
print('是否与原地址相同？',id(Y) == before)
print('\n\n\n')


# 6、转化为其他python对象
# 不论是tensor转numpy还是numpy转tensor，其都再申请新的内存空间，目的是避免同时访问同一个内存块，临界资源。
# 将tensor转换成numpy
A = X.numpy()
print('将tensor转换为numpy', type(A))
# 将numpy转换成tensor
B = torch.tensor(A)
print('将numpy转换为tensor', type(B))
# 将大小为1的张量转换成标量
s1 = torch.tensor([0.555])
print(f'转换成标量的值：{s1} {s1.item()} {float(s1)} {int(s1)}')