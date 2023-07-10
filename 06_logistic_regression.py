import torch

# 数据准备
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 模型设计
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# 创建模型
model = LogisticRegression()
# 损失函数
criterion = torch.nn.BCELoss(reduction='sum')
# 优化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch =', epoch, 'loss =', loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_test =', y_test.item())
