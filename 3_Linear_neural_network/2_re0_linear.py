import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声。"""
    # torch.normal(mean, std, (size, num))) 输出num个符合mean,std正态分布的包含size个数的数组
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 两个张量的矩阵乘积
    y += torch.normal(0, 0.01, y.shape)  # 加上一个噪声
    return X, y.reshape((-1, 1))  # -1表示由其他参数决定


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


