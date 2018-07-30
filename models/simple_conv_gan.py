import torch.nn as nn

img_shape = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self, input_len):
        super(Generator, self).__init__()

        self.fcl = nn.Linear(input_len, 32*7*7)

        self.model = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.fcl(z)
        img = img.view(img.size(0), 32, 7, 7)
        img = self.model(img)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(8, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16, 0.8),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32, 0.8)
        )

        self.result = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.model(img)
        img = img.view(img.size(0), -1)
        validity = self.result(img)
        return validity
