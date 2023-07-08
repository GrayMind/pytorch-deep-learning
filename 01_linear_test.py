import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


# 线性模型 2x+1
def forward(w, b, x):
    return x * w + b


# 损失函数 (y_hat - y)^2
def loss(w, b, x, y):
    y_pred = forward(w, b, x)  # y_hat y的预测值
    return (y_pred - y) * (y_pred - y)


# 穷举法
def linear():
    W = np.arange(0, 4.1, 0.1)
    B = np.arange(0, 4.1, 0.1)
    w, b = np.meshgrid(W, B)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(w, b, x_val)
        loss_val = loss(w, b, x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE', l_sum / 3)

    # 画图
    # https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html
    fig = plt.figure()
    ax = fig.add_subplot('2x+1', projection='3d')
    ax.plot_surface(w, b, l_sum / 3)
    plt.show()


if __name__ == "__main__":
    linear()
