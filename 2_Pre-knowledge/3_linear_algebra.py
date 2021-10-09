# 导包
import torch


# 1、标量
# 标量可以由只有一个元素的张量表示
x = torch.tensor([3.0])
y = torch.tensor([5.0])
print(f'x+y={x+y}\nx-y={x-y}\nx*y={x*y}\nx/y={x/y}\nx**y={x**y}')


# 2、向量
# 没什么好说的，一维的tensor就是向量
x = torch.arange(4)
print(f'向量x={x}')
# 2.1、向量长度、维度和形状
# !WARNING:向量和轴的维度通常是指其长度。而张量的维度通常是指轴的个数。
print(f'向量x的长度为：{len(x)}')
print(f'向量x的维度为：{x.shape}')
print('\n\n\n')


# 3、矩阵
# 将向量从一阶推广到二阶就形成了矩阵
A = torch.arange(20).reshape(4,5)
print(f'矩阵A的值为：\n{A}')
# 计算A的转置
B = A.T
print(f'矩阵A的转置为：\n{B}')
print('\n\n\n')


# 4、张量
# 就像矩阵是向量的推广一样，张量是矩阵的推广
X = torch.arange(24).reshape(2,4,3)
print(f'张量X的值为：\n{X}')
print('\n\n\n')


# 5、张量按元素运算
A = torch.arange(24).reshape(2,3,4)
print(f'矩阵A的值为：\n{A}')
B = A.clone()
print(f'矩阵A+B的值为：\n{A+B}')
# 两个矩阵按元素乘法称为“哈达玛积”
print(f'矩阵A*B的值为：\n{A*B}')
print('\n\n\n')


# 6、降维
# 张量的求和，将多维张量降维成一个标量
x = torch.arange(4, dtype=torch.float32)
print(f'张量x的值为：{x}\n张量x求和为：{x.sum()}')
# 可以将二维张量沿着一个轴降维
X = torch.tensor([[1.0,1,1],[2,2,2],[3,3,3]])
print(f'张量X的值为：\n{X}')
# axis=0表示沿0轴降维
X_sum_axis0 = X.sum(axis=0)
print(f'沿着0轴降维：{X_sum_axis0}')
X_sum_axis1 = X.sum(axis=1)
print(f'沿着1轴降维：{X_sum_axis1}')
# 还可以沿着多个轴进行降维
X_sum_axis01 = X.sum(axis=[0,1])
print(f'沿着01轴降维：{X_sum_axis01}')
# 求平均值，平均值只能作用于浮点数
X_mean = X.mean()
print(f'X的平均值为：{X_mean}')
# 同样的，限制了axis可以沿着0轴1轴进行求平均
X_mean_axis0 = X.mean(axis=0)
print(f'沿着0轴求平均数：{X_mean_axis0}')
print('\n')
# 6.1非降维求和
# 有时求和的时候并不想降维，就要这么操作
sum_X = X.sum(axis=0, keepdims=True)
print(sum_X)
# 计算某个轴的累计总和，此时是不改变维度的
print(X.cumsum(axis=0))
print('\n\n\n')


# 7、向量的点积
# 计算两个矩阵的点积的方式
x = torch.arange(3, dtype=float)
y = torch.ones(3,dtype=float)
print(x,'\n',y)
print(torch.dot(x,y))


# 8、矩阵-向量积(matrix-vector product)
# 矩阵设为mxn，向量设为nx1，相乘得到mx1
A = torch.arange(20).reshape(4,5)
x = torch.arange(5)
Ax = torch.mv(A,x)
print(f'矩阵A的值：\n{A}\n向量x的值：{x}\n矩阵A和向量x的积：{Ax}')


# 9、矩阵-矩阵乘法(matrix-matrix multiplication)
B = torch.arange(20).reshape(5,4)
AB = torch.mm(A,B)
print(AB)


# 10、范数
# 向量范数是将向量映射到标量的函数f，二维向量a的长度||a||就是L2范数
a = torch.tensor([3.,4.]) # norm()不能计算long型数据
print(f'向量a：{a},向量a的范数：{a.norm()}') # norm()计算的是弗罗贝尼乌斯范数(Frobenius norm)


# 练习
# 1、证明(AT)T = A
A = torch.randn(3,4)
print((A.T).T == A)
# 2、证明转置的和等于和的转置
A = torch.randn(3,4)
B = torch.randn(3,4)
print((A + B).T == (A.T + B.T))
# 3、证明A+A.T总是对称
A = torch.randn(4,4) # 方针才有对称矩阵
print((A + A.T).T == (A + A.T))