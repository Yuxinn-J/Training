import numpy as np
from dataset.loader import load_mnist
import matplotlib.pyplot as plt

'''
x:
y, y_hat:
w:
b:
'''


class LinearModel:
    def __init__(self, input_size: int, output_size: int):
        self.weight = np.random.random((input_size, output_size))

    def __call__(self, x: np.ndarray):
        return np.dot(x, self.weight)


def propagate(x: np.ndarray, y):
    # forward propagation -> nll_loss
    x = x - x.max()
    softmax_x = np.exp(x) / np.exp(x).sum()
    y_hat = np.clip(softmax_x, 1e-8, None)
    nll_loss = -(y * np.log(y_hat)).sum()

    # backward propagation
    grad = -y / y_hat
    d_y_hat = np.where(grad == 0, y_hat, y_hat - 1)
    d_weight = np.dot(x.reshape((784, 1)), d_y_hat.reshape(1, 10))

    return d_weight, nll_loss


def train(model, data, label, lr=0.01, epoch=5):
    loss_list = []
    for epoch_iter in range(epoch):
        loss_accumulator = 0

        for i, x in enumerate(data):
            y = np.zeros(10)
            y[label[i]] = 1
            grad, loss = propagate(x, y)

            loss_accumulator += loss
            model.weight -= lr * grad

        loss_list.append(loss_accumulator / 60000)
        print('Epoch:', epoch_iter)

    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test(model, data, label):
    total = 10000
    correct = 0
    for i, x in enumerate(data):
        y_hat = model(x)
        if y_hat.argmax() == label[i]:
            correct += 1

    print(f'test accuracy: {correct/total * 100} %')


# Todo
# 1. 尝试训练 1 个 epoch, 3 个 epoch, 5 个 epoch, 并比较这几个 epoch 下训练出来的模型的准确度
# 2. 使用 matplotlib 打印出每个 epoch 的 loss 均值变化曲线图
# 3. 尝试使用不同的 learning rate, 并打印出每个 epoch 的 loss 均值变化曲线图
# 4. 声明一个线性模型 LinearModel, 并添加 bias: y = wx + b. 训练这个模型, 对比没有 bias 时的准确度(在梯度下降的时候除了更新 weight 还要更新 bias).


if __name__ == '__main__':
    # 加载数据集

    train_data, train_label, test_data, test_label = load_mnist()
    model = LinearModel(784, 10)

    train(model, train_data.reshape((60000, 784)), train_label, epoch=10)
    test(model, test_data.reshape((10000, 784)), test_label)
