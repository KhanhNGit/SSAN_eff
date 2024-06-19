import torch
import torch.nn as nn
from .pub_mod import *
import torchvision.models as models


class Discriminator(nn.Module):
    def __init__(self, max_iter, num_train):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_train)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


class SSAN_R(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000, num_train=4):
        super(SSAN_R, self).__init__()
        model_eff = models.efficientnet_b3(weights='DEFAULT')

        self.input_layer = model_eff.features[0]
        self.layer1 = model_eff.features[1]
        self.layer2 = model_eff.features[2]
        self.layer3 = model_eff.features[3]
        self.layer4 = model_eff.features[4]
        self.layer5 = model_eff.features[5]
        self.layer6 = model_eff.features[6]
        self.layer7 = model_eff.features[7]

        self.layer8 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(512) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.cls_head = nn.Linear(1024, 2, bias=True)

        self.gamma = nn.Linear(512, 512, bias=False)
        self.beta = nn.Linear(512, 512, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.SiLU(inplace=True)
        )
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(48),
            nn.SiLU(inplace=True)
        )
        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(96),
            nn.SiLU(inplace=True)
        )
        self.ada_conv4 = nn.Sequential(
            nn.Conv2d(96, 136, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(136),
            nn.SiLU(inplace=True)
        )
        self.ada_conv5 = nn.Sequential(
            nn.Conv2d(136, 232, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(232),
            nn.SiLU(inplace=True)
        )
        self.ada_conv6 = nn.Sequential(
            nn.Conv2d(232, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(384),
            nn.SiLU(inplace=True)
        )
        self.ada_conv7 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512)
        )
        self.dis = Discriminator(max_iter, num_train)

    def cal_gamma_beta(self, x1):
        x1 = self.input_layer(x1)
        x1_1 = self.layer1(x1)
        x1_2 = self.layer2(x1_1)
        x1_3 = self.layer3(x1_2)
        x1_4 = self.layer4(x1_3)
        x1_5 = self.layer5(x1_4)
        x1_6 = self.layer6(x1_5)
        x1_7 = self.layer7(x1_6)

        x1_8 = self.layer8(x1_7)
        
        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add)+x1_2
        x1_add = self.ada_conv2(x1_add)+x1_3
        x1_add = self.ada_conv3(x1_add)+x1_4
        x1_add = self.ada_conv4(x1_add)+x1_5
        x1_add = self.ada_conv5(x1_add)+x1_6
        x1_add = self.ada_conv6(x1_add)+x1_7
        x1_add = self.ada_conv7(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = torch.nn.functional.adaptive_avg_pool2d(x1_8, 1).reshape(x1_8.shape[0], -1)

        return x1_8, gamma, beta, domain_invariant

    def forward(self, input1, input2):
        x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input1)
        x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
        fea_x1_x1 = self.conv_final(fea_x1_x1)
        fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)
        fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)
        cls_x1_x1 = self.cls_head(fea_x1_x1)

        fea_x1_x2 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
        fea_x1_x2 = self.conv_final(fea_x1_x2)
        fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
        fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)

        dis_invariant = self.dis(domain_invariant)
        return cls_x1_x1, fea_x1_x1, fea_x1_x2, dis_invariant
