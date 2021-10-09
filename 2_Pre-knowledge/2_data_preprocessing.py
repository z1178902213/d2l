# 导包
import os
import pandas as pd
import torch


# 1、读取数据集
# os.path.join()该函数是路径拼接函数，可以为传入的多个字符串之间插入/分隔符
# 以下代码将写入一些数据
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# 使用pandas读取数据并输出
data = pd.read_csv(data_file)
print(data)
print('\n\n\n')


# 2、处理缺失值
# NaN是缺失的数据，处理缺失数据的典型方式是：插值和删除。
# 牛啊，这data.iloc[:, 0:2]输出的居然是前两列
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 填充，就数值被用平均数填充了，其他类型的数据填充不了，会报错
inputs = inputs.fillna(inputs.mean())
print(inputs)
# 由于Alley列只有Pave与NaN两种类型，因此可以分为Alley_Pave与Alley_nan两列，并用0、1代替
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


# 3、转换为张量形式
# 现在数据都变成数值类型了，因此可以将其转换成张量的形式，转换成张量形式后就可以进行tensor操作了
X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(X)
print(y)
print('\n\n\n')


# 练习：删除缺失值最多的列
def delCol(table):
    nan_num = table.isnull().sum(axis=0) # 统计各列缺失值个数
    nan_max = nan_num.idxmax() # 返回缺失值列id
    table = table.drop([nan_max], axis=1) # 删除缺失值最多的列
    print(nan_max) # 输出
    return table # 返回
# 测试delCol()函数
data = delCol(data)
print(data)