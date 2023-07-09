import torch

x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 13.0, 25.0]

w1 = torch.tensor([1.0])
w1.requires_grad = True
w2 = torch.tensor([1.0])
w2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True


# 2*x^2+2*x+1
def forward(x):
    return w1 * (x ** 2) + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

print('forward(4)', forward(4).item())
