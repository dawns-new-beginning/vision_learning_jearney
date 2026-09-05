import torch
from torch import nn
import numpy as np
import matplotlib.pylab as plt

#数据集准备
x = np.array([1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],dtype= np.float32)
y = np.array([0.5, 0.7, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5, 3.7, 4.0, 4.2],dtype= np.float32)

#转换为张量
x_tensor = torch.tensor(x).view(-1, 1)
y_tensor = torch.tensor(y).view(-1, 1)


#定义模型
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

LinearModel = LinearRegressionModel()
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(LinearModel.parameters(), lr = 0.001)


def train(model, x_tensor, y_tensor, epochs, Batch_size):
    N = len(x)
    losses = []    
    indexs = np.random.permutation(N)
    x_tensor_shuffled = x_tensor[indexs]
    y_tensor_shuffled = y_tensor[indexs]

    for epoch in range(epochs):
        loss_sum = 0.0
        for i in range(0, len(x_tensor), Batch_size):
            x_batch = x_tensor_shuffled[i:i+Batch_size]
            y_batch = y_tensor_shuffled[i:i+Batch_size]
            y_pred = model(x_batch)
            current_loss = loss(y_batch,y_pred)

            loss_sum += current_loss.item()
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
        losses.append(loss_sum/(len(x_tensor)/Batch_size))
    return losses


epochs = 100
Batch_size = 3
losses = train(LinearModel,x_tensor,y_tensor,epochs,Batch_size)
plt.plot(range(epochs), losses)
plt.title('loss over epochs')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True, alpha=0.3)   
plt.show()















        