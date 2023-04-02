from functools import partial
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabv3Plus(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=2):
        super(DeepLabv3Plus, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)
        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        # representation = self.representation(torch.cat([x_low, output_feature], dim=1))
        return prediction


if __name__ == '__main__':
    model = DeepLabv3Plus(models.resnet50(pretrained=True))
    print(model(torch.randn(1,3,352,352)).shape)