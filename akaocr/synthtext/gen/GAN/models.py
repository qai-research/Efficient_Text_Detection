"""
@author: Vu Hoang Viet
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import lmdb_dataset_loader
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torchsummary import summary

class Extraction(nn.Module):

    def __init__(self):
        super(Extraction, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4,stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4,stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=4, stride=2),
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
            nn.ConvTranspose2d(in_channels=64*10, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=4, stride=2, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=1, bias=False),
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
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=False),
            nn.Flatten()
        )
        self.fake_real_model = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid())
        self.property_model = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax())

    def forward(self, input):
        """
        @param input:
        @return:
        """
        feature = self.feature_extraction(input)
        fake_real = self.fake_real_model(feature)
        property = self.property_model(feature)
        return fake_real, property

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
                 max_features=10,
                 max_iter=1e+5):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.input_dim = input_dim
        self.classes = classes
        self.num_classes = len(classes)
        self.max_features = max_features
        self.real_label = 1
        self.fake_label = 0
        
        self.input_data = lmdb_dataset_loader(input_path)
        self.target_data = lmdb_dataset_loader(target_path)
        self.target_size = target_size
        self.batch_size = batch_size
        self.max_iter = max_iter
        
        self.extractor = Extraction().to(self.device)
        self.generator = Generator(input_dim).to(self.device)
        self.discriminator = Discriminator(self.num_classes).to(self.device)

    def data_loader(self):
        while True:
            labels = []
            input_imgs = []
            real_imgs = []
            for _ in range(self.batch_size):
                label_ind = np.random.randint(self.num_classes)
                label = self.classes[label_ind]
                input_img = []
                for i in range(10):
                    if np.random.rand() < 0.5:
                        img = np.array(self.input_data.random_sample(label))
                        img = cv2.resize(img, self.target_size, interpolation = cv2.INTER_AREA)
                        img = np.reshape(img, (1,self.target_size[0],self.target_size[1]))
                        input_img.append(img)
                real_img = np.array(self.target_data.random_sample(label))
                real_img = cv2.resize(real_img, self.target_size, interpolation = cv2.INTER_AREA)
                real_img = np.reshape(real_img, (1,self.target_size[0],self.target_size[1]))
                labels.append(np.eye(self.num_classes)[label_ind])
                input_imgs.append(input_img)
                real_imgs.append(real_img)
#             print("="*100)
#             for i,imgs in enumerate(input_imgs):
#                 for img in imgs:
#                     plt.imshow(img)
#                     plt.show()
                
#                 plt.imshow(real_imgs[i])
#                 plt.show()
            yield torch.tensor(np.array(labels)), torch.tensor(np.array(input_imgs)), torch.tensor(np.array(real_imgs))

    def trainer(self, epochs, batch_size=1, max_iter=10000):
        """

        @return:
        """
        optimizerE = optim.Adam(self.extractor.parameters(), lr=1e-1, betas=(0.1, 0.999))
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-1, betas=(0.1, 0.999))
        optimizerG = optim.Adam(self.generator.parameters(), lr=1e-1, betas=(0.1, 0.999))
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            count = 0
            for labels, input_imgs, real_imgs in self.data_loader():
                if count >= self.max_iter:
                    break
                else:
                    count += 1
                self.discriminator.zero_grad()
                real_imgs = real_imgs.to(self.device)

                # Đánh label true fake images
                true_label = torch.full((1,self.batch_size), self.real_label,
                                             device=self.device)
                fake_label = torch.full((1,self.batch_size,), self.fake_label,
                                             device=self.device)

                # Đánh label phân loại
                property_label = torch.tensor(labels,
                                            device=self.device)

                # Trích xuất feature bằng E-Models
                features_outs = []
                for input_img in input_imgs:
                    input_img = input_img.to(self.device)
                    features_out = self.extractor(input_img.float())
                    empty = torch.zeros((self.max_features - len(features_out),
                                         features_out.shape[1],
                                         features_out.shape[2],
                                         features_out.shape[3])).to(self.device)
                    features_out = torch.cat(
                        [features_out,empty
                         ],dim  = 0)
                    features_out = features_out.reshape(1,64*self.max_features,features_out.shape[2], features_out.shape[3])
                    print(features_out.shape)
#                     features_out = features_out.reshape((1, -1))
                    features_outs.append(features_out)

                # Training model D với ảnh reals
                real_imgs = real_imgs.float().to(self.device)
                out_real_src, out_real_cls = self.discriminator(real_imgs)
                d_loss_real = F.binary_cross_entropy(out_real_src, true_label.float())
                d_loss_cls_real = F.binary_cross_entropy_with_logits(out_real_cls, property_label)
                print(d_loss_real.item(), d_loss_cls_real.item())

                # Tạo ảnh fake bằng G-models từ features từ E-Models
                for feature in features_outs:
                    x_fake = self.generator(torch.tensor(feature))

                # Training models D với ảnh fake
                out_fake_src, _ = self.discriminator(x_fake.detach())
                d_loss_fake = F.binary_cross_entropy(out_fake_src, fake_label.float())
# #                 d_loss_cls_fake = F.binary_cross_entropy_with_logits(out_fake_cls, property_label)

#                 alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
#                 x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
#                 out_src, _ = self.D(x_hat)
#                 d_loss_gp = self.gradient_penalty(out_src, x_hat)
                
                d_loss = d_loss_real + d_loss_fake + d_loss_cls_real
                d_loss.backward()
                optimizerD.stop()
                if count % 10 == 0:
                    x_fake = self.generator(torch.tensor(feature))
                    out_fake_src, out_fake_cls = self.discriminator(x_fake.detach())
                    d_loss_fake = F.binary_cross_entropy(out_fake_src, fake_label.float())
                    d_loss_cls_fake = F.binary_cross_entropy_with_logits(out_fake_cls, property_label)
                    
            print(d_loss_fake.item(), d_loss_cls_fake.item())

    def predictor(self):
        """

        @return:
        """
        return        