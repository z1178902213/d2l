import numpy as np
from IPython import display
# d2l包，，说是可以把一些函数保存到这个包里面，用一些#@的东西，以后无需重新定义可以直接调用
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 1、导数与微分
# 定义f(x)函数
def f(x):
    return 3*x**2-4*x
# 定义导数形式
def numerical_lim(x,h):
    return ((f(x+h))-f(x))/h
# 观察随着h->0，(f(x+h)-f(x))/x越接近2
h = 0.1
for i in range(5):
    print(f'h={h:.5f},numerical limit={numerical_lim(1,h):.5f}')
    h *= 0.1
# 使用svg格式在jupyter显示绘图
#def use_svg_display(): #@save
#    display.set_matplotlib_formats('svg')
# 设置matplotlib的图表大小

def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小。"""
    #use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

# 设置matplotlib的轴的名称
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): #@save
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
# 设置plot参数，绘制数据点
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(7,5), axes=None): #@save
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果 `X` 有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
# 绘制图形以表示
# plot()函数传入变量x作为自变量，取值为[0,3)切间隔0.1。
# 第二个传入参数[]为因变量y的函数。该函数传入了一个列表，两个关于y的函数，因此绘制出来的有两条线
# 第3、4个参数是绘图是x,y轴的标签
# legend参数在左上角表示曲线的含义
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# d2l.plt.show()