import torch

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]

w = torch.tensor([1.0])
w.requires_grad = True

b = torch.tensor([1.0])
b.requires_grad = True


def forward(x):
    return w * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad', x, y, l.item())
        w.data = w.data - 0.01 * w.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w.grad.data.zero_()
        b.grad.data.zero_()
    print('\tgrad', 'w=', w.item(), 'b=', b.item(), 'loss=', l.item())
print('forward(4)', forward(4).item())
