import torch
from torch.distributions import multinomial
from d2l import torch as d2l


fair_probs = torch.ones([6])/6  # 根据官方API的意思，这里可以不用除以6，因为已经是ones([6])了，大家概率都一样都是1
print(fair_probs)
# Multinomial进行抽样，根据fair_probs传入的概率对应[0,1,2,3....]采样
print(multinomial.Multinomial(1000, fair_probs).sample())

# 也可以看到这些概率如何随着时间的推移收敛到真实概率
counts = multinomial.Multinomial(10, fair_probs).sample((500,))  # 500组采样，每次采样进行10次掷骰，得到一个[500,6]的数组
cum_counts = counts.cumsum(dim=0)  # 返回dim维的累积和
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('times')
d2l.plt.gca().set_ylabel('value')
d2l.plt.legend()

