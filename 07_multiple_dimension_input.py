import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('./ppt/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        return x


# 模型
model = Model()

# 损失函数
loss_fn = torch.nn.BCELoss(reduction='mean')
# 优化算法
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 开始训练
loss_list = []
for epoch in range(100):
    # forward
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)

    print('epoch =', epoch, 'loss =', loss.item())
    loss_list.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

# 画图
plt.plot(range(100), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
