#step1:导入所需的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

#step2:定义脉冲神经元
class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0

    def forward(self, x):
        self.membrane_potential += x
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike) * self.decay
        return spike

#step3:定义脉冲神经网络模型
class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = SpikingNeuron()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

#step4:加载数据
X = torch.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).float()

# 创建数据加载器
dataset = data.TensorDataset(X, y)
data_loader = data.DataLoader(dataset, batch_size=10, shuffle=True)

#step5:训练模型
model = SNN(input_size=2, hidden_size=10, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.view(-1), y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += ((outputs.view(-1) > 0) == y_batch).sum().item()
        total += y_batch.size(0)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / total:.4f}, Accuracy: {correct / total:.4f}')

#step6：测试模型
X_test = torch.randn(10, 2)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).float()

with torch.no_grad():
    outputs = model(X_test)
    test_loss = criterion(outputs.view(-1), y_test)
    test_accuracy = ((outputs.view(-1) > 0) == y_test).sum().item() / y_test.size(0)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

