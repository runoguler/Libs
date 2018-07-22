import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.simplenet import Net
from models.resnet import ResNet18
from models.mobilenet import MobileNet
from models.mobilenetv2 import MobileNetV2
from models.lenet import LeNet
from models.vgg import VGG


def train(epoch, model, train_loader, optimizer, device, log_interval):
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
    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--model-no', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_training_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=data_transform, download=True)
    cifar_testing_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(cifar_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
    test_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)

    model_no = args.model_no
    if model_no == 1: model = Net().to(device)
    elif model_no == 2: model = ResNet18().to(device)
    elif model_no == 3: model = MobileNet().to(device)
    elif model_no == 4: model = MobileNetV2().to(device)
    elif model_no == 5: model = LeNet().to(device)
    elif model_no == 6: model = VGG().to(device)
    else: model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, device, args.log_interval)
        test(model, test_loader, device)



if __name__ == '__main__':
    main()
