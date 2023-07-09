import torch

# 数据准备
# x y 都为矩阵，3行1列，表示三组数据，每组一个特征值
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


# 模型设计
class LinearModel(torch.nn.Module):
    def __init__(self):
        print('init')
        super(LinearModel, self).__init__()
        # 输入x和输出y的维度都是1，也就是只有一个特征值
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# 创建模型
model = LinearModel()

# 创建损失函数
criterion = torch.nn.MSELoss(reduction='sum')
# 创建优化算法
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1000):
    y_pred = model(x_data)  # forward-1 获取 y 的预测值
    loss = criterion(y_pred, y_data)  # forward-2 计算损失
    print(epoch, loss.item())

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # backward, 自动计算梯度
    optimizer.step()  # update, 更新 w b 参数

print('w = ', model.linear.weight.item())  # 输出 w
print('b = ', model.linear.bias.item())  # 输出 b

# 测试
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_test = ', y_test.item())
