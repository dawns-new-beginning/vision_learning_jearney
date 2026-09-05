import torch 
from torchvision import datasets, transforms

#定义模型
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)   
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


#创建Mnist数据集
def create_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

net = LeNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_function = torch.nn.CrossEntropyLoss()

#模型训练

def train_model(net, train_dataset, optimizer, loss_function, epochs, Batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader): .4f}")



#模型评估
def evaluate_model(net, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

epochs = 100
Batch_size = 16
train_dataset, test_dataset = create_mnist_dataset()
train_model(net, train_dataset, optimizer, loss_function, epochs, Batch_size)
evaluate_model(net, test_dataset)