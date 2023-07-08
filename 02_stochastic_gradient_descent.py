import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(w, x):
    return w * x


def loss(w, x, y):
    return (forward(w, x) - y) ** 2


def gradient(w, x, y):
    return 2 * x * (forward(w, x) - y)


def linear_gradient():
    w = 1.0
    loss_list = []
    for i in range(100):
        for x, y in zip(x_data, y_data):
            loss_val = loss(w, x, y)
            grad_val = gradient(w, x, y)
            w = w - 0.01 * grad_val

        loss_list.append(loss_val)
        print('Epoch:', i, 'w=', w, 'loss=', loss_val)

    # 画图
    plt.plot(range(100), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    linear_gradient()
