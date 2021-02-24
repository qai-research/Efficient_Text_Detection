"""
@author: Vu Hoang Viet
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class HandWrittenGan:
    """
    Model class
    """

    def __init__(self,
                 input_dim,
                 target_size=(64, 64),
                 input_path=None,
                 target_path=None,
                 classes=None,
                 batch_size=1,
                 max_iter=1e+5):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.input_dim = input_dim
        self.classes = classes
        self.num_classes = len(classes)
        self.extractor = Extraction().to(self.device)
        self.generator = Generator(input_dim).to(self.device)
        self.discriminator = Discriminator(num_classes).to(self.device)
        self.real_label = 1
        self.fake_label = 0
        self.input_data = load_lmdb(input_path)
        self.target_data = load_lmdb(target_path)
        self.target_size = target_size
        self.batch_size = batch_size
        self.max_iter = max_iter

    def data_loader(self):
        while True:
            labels = []
            input_imgs = []
            real_imgs = []
            for _ in range(self.batch_size):
                label_ind = np.random.randint(self.num_classes)
                label = self.classes[label_ind]
                for i in range(10):
                    if np.random.rand() < 0.5:
                        img = self.input_data.get_sample(label)
                        img = np.resize(input_img, self.target_size)
                        input_img.append(img)
                real_img = self.target_data.get_sample(label)
                labels.append(np.eye(nb_classes)[label_ind])
                imput_imgs.append(input_img)
                real_imgs.append(real_img)
            yield np.array(labels), np.array(input_imgs), np.array(real_imgs)

    def trainer(self, epochs, batch_size=1, max_iter=10000):
        """

        @return:
        """
        optimizerE = optim.Adam(netD.parameters(), lr=opt.lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        criterion = nn.BCELoss()
        for epoch in epochs:
            for i, labels, input_imgs, real_imgs in enumerate(data_loader()):
                if i >= self.max_iter:
                    break
                self.discriminator.zero_grad()
                real_imgs = real_imgs.to(self.device)
                true_fake_label = torch.full((self.batch_size,), self.real_label,
                                   dtype=real_cpu.dtype, device=self.device)
                property_label = torch.full((self.batch_size,), labels,
                                   dtype=real_cpu.dtype, device=self.device)
                features_outs = self.extractor(input_imgs)
                features_outs = np.stack(
                    (features_outs, np.zeros((self.target_size[0], self.target_size[1], 10 - len(input_imgs)))))
                features_outs = np.reshape(features_outs, (1, h * w * 10))
                features_outs = features_outs.to(self.device)
                out_src, out_cls = self.discriminator(real_imgs)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = F.cross_entropy(out_cls, label_org)
                x_fake = self.generator(features_outs)
                out_src, out_cls = self.D(x_fake.detach())

        return

    def predictor(self):
        """

        @return:
        """
        return


class Extraction(nn.Module):

    def __init__(self):
        super(Extraction, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )

    def forward(self, input):
        """

        @param input:
        @return:
        """
        output = self.model(input)
        return output


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        @param input:
        @return:
        """
        output = self.model(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels=3, out_channels=16, kernel_size=4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels=64, out_channels=128, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels=128, out_channels=1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.fake_real_model = Sequential(
            nn.Conv2d(input_channels=128, out_channels=1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid())
        self.property_model = Sequential(
            nn.Conv2d(input_channels=128, out_channels=num_classes, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        """
        @param input:
        @return:
        """
        feature = self.feature_extraction(input)
        fake_real = self.fake_real(feature)
        property = self.property(feature)
        return fake_real, property
