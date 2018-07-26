import torch
import argparse
import numpy as np
from torchvision import datasets, transforms

from torch.autograd import Variable

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from models.simple_gan import Generator, Discriminator


def display():
    generator = Generator()
    generator.eval()
    generator.load_state_dict(torch.load('./generator.pth'))

    z = torch.zeros((1, 10), dtype=torch.float)
    z[0][3] = 1
    fake = generator(z)[0]
    plt.plot(np.array(fake.detach())[0])
    plt.show()


def train(train_loader, device, use_cuda):
    epochs = 10

    loss = torch.nn.BCELoss()

    generator = Generator()
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)
    loss.to(device)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if use_cuda:
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    for epoch in range(epochs):
        for (imgs, label) in train_loader:
            img, label = imgs.to(device), label.to(device)

            ones = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            zeros = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            optim_gen.zero_grad()

            real = Variable(imgs.type(Tensor))

            z = np.zeros((len(label), 10), dtype=int)
            for i in range(len(label)):
                z[i][label[i]] = 1
            z = Variable(Tensor(z))

            fake = generator(z)
            loss_g = loss(discriminator(fake), ones)
            loss_g.backward()
            optim_gen.step()

            optim_dis.zero_grad()
            real_loss = loss(discriminator(real), ones)
            fake_loss = loss(discriminator(fake.detach()), zeros)
            loss_d = (real_loss + fake_loss) / 2
            loss_d.backward()
            optim_dis.step()

        print("Epoch: ", epoch, ", Loss(Gen): ", loss_g.item(), ", Loss(Dis): ", loss_d.item())

    torch.save(generator.state_dict(), './generator.pth')


def main():
    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, help='batch size for testing (default: 256)')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers for cuda')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = datasets.MNIST("../data/MNIST", train=True, transform=data_transform, download=True)
    test_data = datasets.MNIST("../data/MNIST", train=False, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)

    train(train_loader, device, use_cuda)
    # display()


if __name__ == '__main__':
    main()
