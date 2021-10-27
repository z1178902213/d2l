import torch
from torch import nn
from torch.nn import functional as F


# 加载和保存张量
x = torch.arange(4)
# 将x中的数据保存到当前目录下的x-files文件中
torch.save(x, 'x-file')

# 使用torch的load方法加载x-files文件
x2 = torch.load('x-file')
print(x2)

# 可以多维度保存，但是读取的时候也需要
y = torch.zeros(4)
torch.save([x, y],'x-files')
unit = torch.load('x-files')
print(len(unit))

mydict = {'x':x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self):
        return self.output(F.relu(self.hidden(x)))

