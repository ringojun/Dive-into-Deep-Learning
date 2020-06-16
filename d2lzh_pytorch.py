import torch
import random
import matplotlib.pyplot as plt
from IPython import display


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    """每次返回batch_size个随机样本的特征和标签"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield torch.index_select(features, 0, j), torch.index_select(labels, 0, j)


def linreg(X, w, b):
    """定义线性回归的矢量表达式模型"""
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """定义线性回归的平方误差损失函数"""
    return (y_hat - y.view(y_hat.size()))**2 / 2


def sgd(params, lr, batch_size):
    """定义小批量随机梯度下降算法的优化算法"""
    for param in params:
        param.data -= lr * param.grad / batch_size
