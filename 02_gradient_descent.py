import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(w, x):
    return w * x


def cost(w, xs, ys):
    l_sum = 0
    for x, y in zip(xs, ys):
        l_sum += (forward(w, x) - y) ** 2
    return l_sum / len(xs)


def gradient(w, xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (forward(w, x) - y)
    return grad / len(xs)


def linear_gradient():
    w = 1.0
    cost_list = []
    for i in range(100):
        cost_val = cost(w, x_data, y_data)
        cost_list.append(cost_val)
        grad_val = gradient(w, x_data, y_data)
        w = w - 0.01 * grad_val
        print('Epoch:', i, 'w=', w, 'loss=', cost_val)
    print('forward(4)=', forward(w, 4))
    # 画图
    plt.plot(range(100), cost_list)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()


if __name__ == '__main__':
    linear_gradient()
