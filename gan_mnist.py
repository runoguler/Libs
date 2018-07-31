import torch
import argparse
import numpy as np
from torchvision import datasets, transforms

from torch.autograd import Variable

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# from models.simple_gan import Generator, Discriminator
# from models.conv_gan import Generator, Discriminator
# from models.simple_conv_gan import Generator, Discriminator
from models.aux_gan import Generator, Discriminator


def display(label):
    gen_len = 100
    generator = Generator(gen_len)
    generator.load_state_dict(torch.load('./generator.pth'))
    generator.eval()

    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (10, gen_len))))

    fake = generator(z)

    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('./discriminator.pth'))
    validity = discriminator(fake)
    print(validity)

    plt.imshow(np.array(fake.detach())[0])
    plt.show()


def train(args, train_loader, device, Tensor, LongTensor):
    gen_len = 100

    epochs = args.epochs

    validity_loss = torch.nn.BCELoss()
    result_loss = torch.nn.CrossEntropyLoss()

    generator = Generator(gen_len)
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)
    validity_loss.to(device)
    result_loss.to(device)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for (images, labels) in train_loader:
            batch_len = images.size(0)

            ones = Variable(Tensor(batch_len, 1).fill_(1.0), requires_grad=False)
            zeros = Variable(Tensor(batch_len, 1).fill_(0.0), requires_grad=False)

            images, labels = Variable(images.to(device).type(Tensor)), Variable(labels.to(device).type(LongTensor))

            rd_labels = Variable(LongTensor(np.random.randint(0, 10, batch_len)))
            z = Variable(Tensor(np.random.normal(0, 1, (batch_len, gen_len))))

            optim_gen.zero_grad()
            fake = generator(z, rd_labels)
            pred_label, validity = discriminator(fake)
            loss_g = validity_loss(validity, ones)
            loss_g.backward()
            optim_gen.step()

            optim_dis.zero_grad()
            real_labels, real_validity = discriminator(images)
            fake_labels, fake_validity = discriminator(fake.detach())
            real_loss = (validity_loss(real_validity, ones) + result_loss(real_labels, labels)) / 2
            fake_loss = (validity_loss(fake_validity, zeros) + result_loss(fake_labels, labels)) / 2
            loss_d = (real_loss + fake_loss) / 2
            loss_d.backward()
            optim_dis.step()

            pred = np.concatenate([real_validity.data.cpu().numpy(), fake_validity.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), rd_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        print("Epoch: ", epoch, ", Loss(Gen): ", loss_g.item(), ", Loss(Dis): ", loss_d.item(), ", Acc(Dis): ", d_acc)

    torch.save(generator.state_dict(), './generator.pth')
    torch.save(discriminator.state_dict(), './discriminator.pth')


def main():
    epochs = 10
    train_or_display = 1
    lr_g = 0.0002
    lr_d = 0.0002
    digit_to_display = 0

    parser = argparse.ArgumentParser(description="Parameters for Training GAN on MNIST dataset")
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers for cuda')
    parser.add_argument('--lr-g', type=float, default=lr_g, help='learning rate for the generator network')
    parser.add_argument('--lr-d', type=float, default=lr_d, help='learning rate for the discriminator network')
    parser.add_argument('--epochs', type=int, default=epochs, help='epoch number to train (default: 10)')
    parser.add_argument('--train', type=int, default=train_or_display, help='train(1) or display(0) (default: train(1))')
    parser.add_argument('--display-label', type=int, default=digit_to_display, help='which digit to display')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    training_data = datasets.MNIST("../data/MNIST", train=True, transform=data_transform, download=True)

    data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)

    if use_cuda:
        Tensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        Tensor = torch.FloatTensor
        LongTensor = torch.LongTensor


    if args.train:
        train(args, data_loader, device, Tensor, LongTensor)
    else:
        display(args.display_label)


if __name__ == '__main__':
    main()
