# H. Wei et al., "Spatio-Temporal Feature Fusion and Guide Aggregation Network for Remote Sensing Change Detection,"
# in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5642216, doi: 10.1109/TGRS.2024.3470314.


import torch
import torch.nn as nn
import torch.nn.functional as F
from module._utils import KaimingInitMixin, Identity
from module.backbones import resnet
from torch.nn import Module, init

def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d
class Backbone(nn.Module, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet101':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet152':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # x 3 256 256
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1, stride=1, padding=0, groups=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class STFF (Module):
    def __init__(self, in_channels):
        super(STFF, self).__init__()
        self.DWC1 = DepthWiseConv(in_channels // 2, in_channels // 2)
        self.DWC2 = DepthWiseConv(in_channels // 2, in_channels // 2)
        self.DWC3 = DepthWiseConv(in_channels // 2, in_channels // 2)

    def forward(self, x, y):
        F_abs = torch.abs(x - y)
        x_1, x_2 = torch.split(x, x.size(1) // 2, dim=1)
        y_1, y_2 = torch.split(y, x.size(1) // 2, dim=1)
        map_abs1, map_abs2 = torch.split(F_abs, x.size(1) // 2, dim=1)

        F_STFF = torch.cat([self.DWC1(x_1 + map_abs2), self.DWC2(map_abs1 + y_2),
                             self.DWC3(y_1 + x_2)], dim=1)

        return F_STFF


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        query = self.query_conv(x)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class DeepFeatureGuideModule(nn.Module):
    def __init__(self):
        super(DeepFeatureGuideModule, self).__init__()
        self.conv11 = nn.Conv2d(768, 384, 1)
        self.conv = nn.Sequential(BasicConv2d(768, 256, 3, 1, 1),
                                  Self_Attention(256),
                                  nn.Conv2d(256, 1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y):
        size = x.size()[2:4]
        y = self.conv11(y)
        y_up = F.interpolate(y, size, mode='bilinear', align_corners=True)
        F_DFG = self.conv(torch.cat([x, y_up], dim=1))

        return F_DFG

class GuideAggregationModule(nn.Module):
    def __init__(self, channel, out_c):
        super().__init__()
        self.C = [512, 256, 128]
        self.conv11 = BasicConv2d(channel, channel // 2, kernel_size=1)

        self.conv13 = nn.Conv2d(channel // 2, channel // 2, (1, 3), padding=(0, 1))
        self.conv31 = nn.Conv2d(channel // 2, channel // 2, (3, 1), padding=(1, 0))
        self.conv33 = nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(channel // 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv11_2 = BasicConv2d(channel // 2, out_c, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y, dfg):
        xy = torch.cat([x, y], dim=1)
        xy = (torch.sigmoid(dfg) + 1) * xy

        xy = self.conv11(xy)

        xy13 = self.conv13(xy)
        xy31 = self.conv31(xy)
        xy33 = self.conv33(xy)

        out = (xy13 + xy31 + xy33) / 3
        out = self.conv11_2(self.relu(self.bn(out)))

        return out

class STFFGA(nn.Module):
    def __init__(self, in_ch=3):
        super(STFFGA, self).__init__()
        self.extract = Backbone(in_ch=in_ch, arch='resnet18')  #
        self.C = [64, 128, 256, 512]
        #  STFF Module
        self.STFF1 = STFF(self.C[0])
        self.STFF2 = STFF(self.C[1])
        self.STFF3 = STFF(self.C[2])
        self.STFF4 = STFF(self.C[3])

        # DFG Module
        self.CFM = DeepFeatureGuideModule()
        self.sigmoid = nn.Sigmoid()

        # GA Module
        self.GA3 = GuideAggregationModule(1152, self.C[3])
        self.GA2 = GuideAggregationModule(704, self.C[2])
        self.GA1 = GuideAggregationModule(352, self.C[1])

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1),
                                       nn.Conv2d(64, 1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, A, B):
        size = A.size()[2:]
        layer1_A, layer2_A, layer3_A, layer4_A = self.extract(A)
        layer1_B, layer2_B, layer3_B, layer4_B = self.extract(B)

        layer1 = self.STFF1(layer1_B, layer1_A)   # 96
        layer2 = self.STFF2(layer2_B, layer2_A)   # 192
        layer3 = self.STFF3(layer3_B, layer3_A)   # 384
        layer4 = self.STFF4(layer4_B, layer4_A)   # 768

        DFG = self.CFM(layer3, layer4)

        DFG4 = F.interpolate(DFG, layer4.size()[2:], mode='bilinear', align_corners=True)
        layer4_out = (self.sigmoid(DFG4) + 1) * layer4

        feature3 = self.GA3(self.upsample2x(layer4_out), layer3, DFG)

        DFG3 = F.interpolate(DFG, layer2.size()[2:], mode='bilinear', align_corners=True)
        feature2 = self.GA2(self.upsample2x(feature3), layer2, DFG3)


        DFG2 = F.interpolate(DFG, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature1 = self.GA1(self.upsample2x(feature2), layer1, DFG2)

        output_map = self.decoder_final(feature1)

        output_map = F.interpolate(output_map, size, mode='bilinear', align_corners=True)
        dfg_map = F.interpolate(DFG, size, mode='bilinear', align_corners=True)

        return dfg_map, output_map



if __name__=='__main__':
    net = STFFGA().cuda()
    out = net(torch.rand((2, 3, 256, 256)).cuda(), torch.rand((2, 3, 256, 256)).cuda())[0]
    print(out.shape)
