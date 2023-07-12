import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=False)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32, requires_grad=True).view(-1, 1)
y_test = torch.tensor(np.array(y_test), dtype=torch.float32, requires_grad=False).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(30, 16)
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 4)
        self.linear4 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        return x


model = Model()
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_list = []
loss_list_test = []
for epoch in range(1000):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_test_pred = model(X_test)
        loss_test = loss_fn(y_test_pred, y_test)
        loss_list_test.append(loss_test.item())
        print(epoch, loss.item(), loss_test.item())

# 画图
fig, ax = plt.subplots()
ax.plot(range(1000), loss_list, label='train')
ax.plot(range(1000), loss_list_test, label='test')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend()
plt.show()
