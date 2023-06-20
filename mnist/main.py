import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        # (입력 뉴런, 출력 뉴런)
        self.fc1 = nn.Linear(3136, 1000)  # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    i = 0
    for current_epoch in range(epoch):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("Train Step : {}\tLoss : {:6f}".format(i, loss.item()))
            i += 1

    torch.save(model.state_dict(), f"mnist_cnn_epoch_{epoch}.pt")


def test(model, device, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
    print('Test set Accuracy : {:.6f}%'.format(100. * correct / len(test_loader.dataset)))


def main(batch_size, learning_rate, epoch_num):
    CUDA_AVAILABLE = torch.cuda.is_available()
    device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

    print(f'Current Device: {device}')

    torch.manual_seed(922)

    train_data = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False,
                               transform=transforms.ToTensor())

    print(f'number of training data : {len(train_data)}')
    print(f'number of test data : {len(test_data)}')

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size, shuffle=True)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion,
          epoch=epoch_num)
    test(model=model, device=device, test_loader=test_loader)


if __name__ == "__main__":
    main(batch_size=50, learning_rate=0.0001, epoch_num=20)
