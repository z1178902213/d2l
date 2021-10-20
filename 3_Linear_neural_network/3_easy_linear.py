import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 生成数据


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))
net = nn.Sequential(nn.Linear(2, 1))  # Linear(2, 1)表示输入2个特征输出1

net[0].weight.data.normal_(0, 0.01)  # net[0]貌似说是第一层神经网络的意思
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算损失函数，正向传播
        trainer.zero_grad()
        l.backward()  # 对损失函数进行反向传播
        trainer.step()  ###########？？？？？？？？？？？
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss = {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b)