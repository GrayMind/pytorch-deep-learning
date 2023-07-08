import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_list = []
mse_list = []


# 线性模型 2x
def forward(w, x):
    return x * w


# 损失函数 (y_hat - y)^2
def loss(w, x, y):
    y_pred = forward(w, x)  # y_hat y的预测值
    return (y_pred - y) * (y_pred - y)


# 穷举法
def linear():
    for w in np.arange(0, 4.1, 0.1):
        print('w=', w)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(w, x_val)
            loss_val = loss(w, x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE', l_sum / 3)
        w_list.append(w)
        mse_list.append(l_sum / 3)

    # 画图
    plt.plot(w_list, mse_list)
    plt.xlabel('w')
    plt.ylabel('mse')
    plt.show()


if __name__ == "__main__":
    linear()
