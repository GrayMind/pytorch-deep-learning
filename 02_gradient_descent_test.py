import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


# 2*x+1
def forward(w, b, x):
    return w * x + b


def cost(w, b, xs, ys):
    l_sum = 0
    for x, y in zip(xs, ys):
        l_sum += (forward(w, b, x) - y) ** 2
    return l_sum / len(xs)


def gradient(w, b, xs, ys):
    grad_w = 0
    grad_b = 0
    for x, y in zip(xs, ys):
        grad_w += 2 * x * (forward(w, b, x) - y)
        grad_b += 2 * (forward(w, b, x) - y)
    return grad_w / len(xs), grad_b / len(xs)


def linear_gradient():
    w = 1.0
    b = 0.1
    cost_list = []
    for i in range(1000):
        cost_val = cost(w, b, x_data, y_data)
        cost_list.append(cost_val)
        grad_w_val, grad_b_val = gradient(w, b, x_data, y_data)
        w = w - 0.01 * grad_w_val
        b = b - 0.01 * grad_b_val
        print('Epoch:', i, 'w=', w, 'b=', b, 'cost=', cost_val)

    print('forward(4)=', forward(w, b, 4))
    # 画图
    plt.plot(range(1000), cost_list)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()


if __name__ == '__main__':
    linear_gradient()
