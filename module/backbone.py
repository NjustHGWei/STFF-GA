import torch
import torch.nn as nn
import torch.nn.functional as F

from PSA10 import PSA

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PSA(channel=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_res1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PSA(channel=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_res2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PSA(channel=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_res3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PSA(channel=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_res4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PSA(channel=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_res5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.layer1(x) + self.conv_res1(x) # 8,64,256,256
        x1_pool1 = self.pool1(x1)   # 8,64,128,128

        x2 = self.layer2(x1_pool1) + self.conv_res2(x1_pool1)  # 8,128,128,128
        x2_pool1 = self.pool2(x2)  # 8,128,64,64

        x3 = self.layer3(x2_pool1) + self.conv_res3(x2_pool1)  # 8,256,64,64
        x3_pool1 = self.pool3(x3)  # 8,256,32,32

        x4 = self.layer4(x3_pool1) + self.conv_res4(x3_pool1)  # 8,512,32,32
        x4_pool1 = self.pool4(x4)  # 8,512,16,16

        x4 = self.layer5(x4_pool1) + self.conv_res5(x4_pool1)  # 8,512,16,16

        return x1, x2, x3, x4

if __name__=='__main__':

    net = my_net().cuda()
    out1, out2, out3, out4 = net(torch.rand((2, 3, 256, 256)).cuda())
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)