import torch
import torch.nn as nn


img_shape = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self, input_len=100, class_number=10):
        super(Generator, self).__init__()

        self.label = nn.Embedding(class_number, input_len)

        self.fcl = nn.Linear(input_len, 64 * (img_shape[1] // 4) * (img_shape[2] // 4))

        self.model = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),

            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),

            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, img_shape[0], 3, padding=1),

            nn.Tanh()
        )

    def forward(self, z, labels):
        label_and_noise = torch.mul(self.label(labels), z)
        fcl_out = self.fcl(label_and_noise)
        img = fcl_out.view(fcl_out.size(0), 64, img_shape[2] // 4, img_shape[2] // 4)
        return self.model(img)


class Discriminator(nn.Module):
    def __init__(self, class_number=10):
        super(Discriminator, self).__init__()

        def block(in_channels, out_channels, bn=True):
            layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Dropout2d(0.2)])
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(img_shape[0], 16, bn=False)
            *block(16, 32)
            *block(32, 64)
        )

        self.result = nn.Sequential(
            nn.Linear(64 * 4 * 4, class_number),
            nn.Softmax()
        )

        self.validity = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.model(img)
        fcl_in = img.view(img.size(0), -1)
        result = self.result(fcl_in)
        validity = self.validity(fcl_in)
        return result, validity
