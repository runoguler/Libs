import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.simplenet import Net
from models.resnet import ResNet18
from models.mobilenetv2 import MobileNetV2


def train(epoch, model, train_loader, optimizer, device):
    log_interval = 20
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, labels = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, labels, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # hyper parameters
    training_batch_size = 64
    testing_batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    epochs = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_training_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=data_transform, download=True)
    cifar_testing_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(cifar_training_data, batch_size=training_batch_size, shuffle=True, **cuda_args)
    test_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=testing_batch_size, shuffle=True, **cuda_args)

    model = Net().to(device)
    # model = MobileNetV2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device)
        test(model, test_loader, device)



if __name__ == '__main__':
    main()
