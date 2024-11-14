import timm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torchvision
import torch.utils.model_zoo as model_zoo
from spp_layer import spatial_pyramid_pool
import math


# print("timm version:", timm.__version__)
# print(timm.__file__)
# model_names = timm.list_models()
# for name in model_names:
#     print(name)



class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


# #resnet基准模型
# class Resnet50_gap_fine_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_gap_fine_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         #self.conv = nn.Sequential(
#         #    nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#         #    nn.BatchNorm2d(1024),
#         #    nn.ReLU(inplace=True),
#         #    nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#         #    nn.BatchNorm2d(512),
#         #    nn.ReLU(inplace=True),
#         #)
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             #x = self.conv(x)
#             x = self.pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
# #pool_type
# class Resnet50_gmp_fine_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_gmp_fine_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.AdaptiveMaxPool2d((1, 1))
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             x = self.pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
#
# class Resnet50_mix_fine_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_mix_fine_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 256, kernel_size=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
#
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             x1 = self.pool1(x)
#             x2 = self.pool2(x)
#             x = torch.cat((x1, x2), dim=1).view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
#
# class Resnet50_sp_fine_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_sp_fine_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.Sequential(
#             nn.Conv2d(512, 4, kernel_size=1, stride=1, bias=True),
#             nn.ReLU(),
#         )
#
#         self.mlp = nn.Sequential(
#             nn.Linear(576, 256),
#             # nn.BatchNorm1d(72),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             x = self.pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
#
# class Resnet50_std_fine_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_std_fine_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 256, kernel_size=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             mean = self.pool(x).view(x.size(0), -1)
#             std = torch.std(x.view(x.size(0), x.size(1), -1), dim=2)
#             x = torch.cat((mean, std), dim=1)
#             x = self.mlp(x)
#             return x
#
#
#
#
# #train_type
# class Resnet50_gap_end2end_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_gap_end2end_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=False, pretrained_cfg=pretrained_cfg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             x = self.pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
# class Resnet50_gap_fixed_single(nn.Module):
#     def __init__(self):
#         super(Resnet50_gap_fixed_single, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         for param in self.model.parameters():
#             param.requires_grad = False 
#         self.conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=1, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#             x = self.model.forward_features(x)
#             x = self.conv(x)
#             x = self.pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.mlp(x)
#             return x
#
#
#
#
# #Multi
# class Resnet50_gap_fine_multi(nn.Module):
#     def __init__(self):
#         super(Resnet50_gap_fine_multi, self).__init__()
#
#         pretrained_cfg = timm.models.create_model('resnet50').default_cfg
#         pretrained_cfg['file'] = r"checkpoint/hugging_face/resnet50.bin"
#         self.model = timm.create_model('resnet50', pretrained=True, pretrained_cfg=pretrained_cfg)
#
#         self.media_features = SaveOutput()
#         layers = ['layer1', 'layer2', 'layer3', 'layer4']
#         for name, layer in self.model.named_modules():
#             for lname in layers:
#                 if name == lname:
#                     layer.register_forward_hook(self.media_features)
#
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 128, kernel_size=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(2048, 512, kernel_size=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 128, kernel_size=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#
#         x = self.model.forward_features(x)
#         ft_l1 = self.conv0(self.media_features.outputs[0])
#         ft_l2 = self.conv1(self.media_features.outputs[1])
#         ft_l3 = self.conv2(self.media_features.outputs[2])
#         ft_l4 = self.conv3(self.media_features.outputs[3])
#
#         self.media_features.clear()
#         ft_l1, ft_l2, ft_l3, ft_l4 = self.pool(ft_l1), self.pool(ft_l2), self.pool(ft_l3), self.pool(ft_l4)
#         x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1).view(x.size(0), -1)
#         x = self.mlp(x)
#         return x

#resnet_new
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        out = {}
        out['f1'] = f1
        out['f2'] = f2
        out['f3'] = f3
        out['f4'] = f4

        return out

def resnet50_backbone(pretrained=False):
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model

#resnet_model
class Resnet50_gap_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_gap_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

class Resnet50_gmp_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_gmp_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

class Resnet50_mix_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_mix_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x = torch.cat((x1, x2), dim=1).view(x.size(0), -1)
        x = self.mlp(x)
        return x

class Resnet50_sp_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_sp_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

class Resnet50_std_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_std_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        mean = self.pool(x).view(x.size(0), -1)
        std = torch.std(x.view(x.size(0), x.size(1), -1), dim=2)
        x = torch.cat((mean, std), dim=1)
        x = self.mlp(x)
        return x


class Resnet50_gap_fine_multi(nn.Module):
    def __init__(self):
        super(Resnet50_gap_fine_multi, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ft = self.res(x)
        x1, x2, x3, x4 = ft['f1'], ft['f2'], ft['f3'], ft['f4']
        x1 = self.gap(self.conv1(x1))
        x2 = self.gap(self.conv2(x2))
        x3 = self.gap(self.conv3(x3))
        x4 = self.gap(self.conv4(x4))

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.mlp(x)
        return x


class Resnet50_spp_fine_single(nn.Module):
    def __init__(self):
        super(Resnet50_spp_fine_single, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(512 * sum([i * i for i in self.output_num]), 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.res(x)['f4']
        x = self.conv(x)
        spp = spatial_pyramid_pool(x, x.size(0), [x.size(2), x.size(3)], self.output_num)
        x = self.mlp(spp)
        return x


class Resnet50_spp_fine_multi(nn.Module):
    def __init__(self):
        super(Resnet50_spp_fine_multi, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(256 * sum([i * i for i in self.output_num]) * 4, 512),  
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ft = self.res(x)
        x1, x2, x3, x4 = ft['f1'], ft['f2'], ft['f3'], ft['f4']

        x1 = spatial_pyramid_pool(self.conv1(x1), x1.size(0), [x1.size(2), x1.size(3)], self.output_num)
        x2 = spatial_pyramid_pool(self.conv2(x2), x2.size(0), [x2.size(2), x2.size(3)], self.output_num)
        x3 = spatial_pyramid_pool(self.conv3(x3), x3.size(0), [x3.size(2), x3.size(3)], self.output_num)
        x4 = spatial_pyramid_pool(self.conv4(x4), x4.size(0), [x4.size(2), x4.size(3)], self.output_num)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.mlp(x)
        return x



#vgg基准模型
class Vgg19_gap_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_gap_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




#pool type
class Vgg19_gmp_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_gmp_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




class Vgg19_mix_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_mix_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            x = self.conv(x)
            x1 = self.pool1(x)
            x2 = self.pool2(x)
            x = torch.cat((x1, x2), dim=1).view(x.size(0), -1)
            x = self.mlp(x)
            return x




class Vgg19_sp_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_sp_fine_single, self).__init__()
       
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)


        self.pool = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(576, 256),
            # nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




class Vgg19_std_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_std_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            x = self.conv(x)
            mean = self.pool(x).view(x.size(0), -1)
            std = torch.std(x.view(x.size(0), x.size(1), -1), dim=2)
            x = torch.cat((mean, std), dim=1)
            x = self.mlp(x)
            return x




#train_type
class Vgg19_gap_end2end_single(nn.Module):
    def __init__(self):
        super(Vgg19_gap_end2end_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=False, pretrained_cfg=pretrained_cfg)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x




class Vgg19_gap_fixed_single(nn.Module):
    def __init__(self):
        super(Vgg19_gap_fixed_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        for param in self.model.parameters():
            param.requires_grad = False 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x




#Multi
class Vgg19_gap_fine_multi(nn.Module):
    def __init__(self):
        super(Vgg19_gap_fine_multi, self).__init__()

        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['features.9', 'features.18', 'features.27', 'features.36']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.features(x)
            ft_l1 = self.media_features.outputs[0]
            ft_l2 = self.conv1(self.media_features.outputs[1])
            ft_l3 = self.conv2(self.media_features.outputs[2])
            ft_l4 = self.conv2(self.media_features.outputs[3])

            self.media_features.clear()
            ft_l1, ft_l2, ft_l3, ft_l4 = self.pool(ft_l1), self.pool(ft_l2), self.pool(ft_l3), self.pool(ft_l4)
            x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1).view(x.size(0), -1)
            x = self.mlp(x)
            return x


class Vgg19_spp_fine_single(nn.Module):
    def __init__(self):
        super(Vgg19_spp_fine_single, self).__init__()

        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(512 * sum([i * i for i in self.output_num]), 256),  
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = spatial_pyramid_pool(x, x.size(0), [x.size(2), x.size(3)], self.output_num)
        x = self.mlp(x)
        return x


class Vgg19_spp_fine_multi(nn.Module):
    def __init__(self):
        super(Vgg19_spp_fine_multi, self).__init__()

        pretrained_cfg = timm.models.create_model('vgg19').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vgg19.bin"
        self.model = timm.create_model('vgg19', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['features.9', 'features.18', 'features.27', 'features.36']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(128 * sum([i * i for i in self.output_num]) * 3 + 512 * sum([i * i for i in self.output_num]),
                      256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.features(x)
        ft_l1 = self.media_features.outputs[0]
        ft_l2 = self.conv1(self.media_features.outputs[1])
        ft_l3 = self.conv2(self.media_features.outputs[2])
        ft_l4 = self.media_features.outputs[3]

        self.media_features.clear()

        ft_l1 = spatial_pyramid_pool(ft_l1, ft_l1.size(0), [ft_l1.size(2), ft_l1.size(3)], self.output_num)
        ft_l2 = spatial_pyramid_pool(ft_l2, ft_l2.size(0), [ft_l2.size(2), ft_l2.size(3)], self.output_num)
        ft_l3 = spatial_pyramid_pool(ft_l3, ft_l3.size(0), [ft_l3.size(2), ft_l3.size(3)], self.output_num)
        ft_l4 = spatial_pyramid_pool(ft_l4, ft_l4.size(0), [ft_l4.size(2), ft_l4.size(3)], self.output_num)

        x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1)
        x = self.mlp(x)
        return x


#inception_resnet_v2基准模型
class InceptionResnetV2_gap_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_gap_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x





#pool_type
class InceptionResnetV2_gmp_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_gmp_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




class InceptionResnetV2_mix_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_mix_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x1 = self.pool1(x)
            x2 = self.pool2(x)
            x = torch.cat((x1, x2), dim=1).view(x.size(0), -1)
            x = self.mlp(x)
            return x





class InceptionResnetV2_sp_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_sp_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.Sequential(
            nn.Conv2d(512, 5, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(500, 256),
            # nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x





class InceptionResnetV2_std_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_std_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            mean = self.pool(x).view(x.size(0), -1)
            std = torch.std(x.view(x.size(0), x.size(1), -1), dim=2)
            x = torch.cat((mean, std), dim=1)
            x = self.mlp(x)
            return x




#train_type
class InceptionResnetV2_gap_end2end_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_gap_end2end_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=False, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




class InceptionResnetV2_gap_fixed_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_gap_fixed_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        for param in self.model.parameters():
            param.requires_grad = False  
        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            return x




#Multi
class InceptionResnetV2_gap_fine_multi(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_gap_fine_multi, self).__init__()
        
        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['conv2d_4a', 'repeat', 'repeat_1', 'conv2d_7b']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.conv0 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1088, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):

            x = self.model.forward_features(x)
            ft_l1 = self.conv0(self.media_features.outputs[0])
            ft_l2 = self.conv1(self.media_features.outputs[1])
            ft_l3 = self.conv2(self.media_features.outputs[2])
            ft_l4 = self.conv3(self.media_features.outputs[3])

            self.media_features.clear()
            ft_l1, ft_l2, ft_l3, ft_l4 = self.pool(ft_l1), self.pool(ft_l2), self.pool(ft_l3), self.pool(ft_l4)
            x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1).view(x.size(0), -1)
            x = self.mlp(x)
            return x


class InceptionResnetV2_spp_fine_single(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_spp_fine_single, self).__init__()

        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(512 * sum([i * i for i in self.output_num]), 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.conv(x)
        x = spatial_pyramid_pool(x, x.size(0), [x.size(2), x.size(3)], self.output_num)
        x = self.mlp(x)
        return x


class InceptionResnetV2_spp_fine_multi(nn.Module):
    def __init__(self):
        super(InceptionResnetV2_spp_fine_multi, self).__init__()

        pretrained_cfg = timm.models.create_model('inception_resnet_v2').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/inception.bin"
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['conv2d_4a', 'repeat', 'repeat_1', 'conv2d_7b']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.conv0 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1088, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.output_num = [4, 2, 1]  
        self.mlp = nn.Sequential(
            nn.Linear(128 * sum([i * i for i in self.output_num]) * 4, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        ft_l1 = self.conv0(self.media_features.outputs[0])
        ft_l2 = self.conv1(self.media_features.outputs[1])
        ft_l3 = self.conv2(self.media_features.outputs[2])
        ft_l4 = self.conv3(self.media_features.outputs[3])

        self.media_features.clear()

        ft_l1 = spatial_pyramid_pool(ft_l1, ft_l1.size(0), [ft_l1.size(2), ft_l1.size(3)], self.output_num)
        ft_l2 = spatial_pyramid_pool(ft_l2, ft_l2.size(0), [ft_l2.size(2), ft_l2.size(3)], self.output_num)
        ft_l3 = spatial_pyramid_pool(ft_l3, ft_l3.size(0), [ft_l3.size(2), ft_l3.size(3)], self.output_num)
        ft_l4 = spatial_pyramid_pool(ft_l4, ft_l4.size(0), [ft_l4.size(2), ft_l4.size(3)], self.output_num)

        x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1)
        x = self.mlp(x)
        return x


#vit基准模型
class Vit_token_fine_single(nn.Module):
    def __init__(self):
        super(Vit_token_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vit_base_patch16_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vit_url.npz"
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)

        # self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1)
        
    def forward(self, x):
            x = self.model(x)
            return x




#pool_type
class Vit_avg_fine_single(nn.Module):
    def __init__(self):
        super(Vit_avg_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vit_base_patch16_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vit_url.npz"
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1, global_pool='avg', pretrained_cfg=pretrained_cfg)

    def forward(self, x):
            x = self.model(x)
            return x




#train_type
class Vit_token_end2end_single(nn.Module):
    def __init__(self):
        super(Vit_token_end2end_single, self).__init__()
 
        pretrained_cfg = timm.models.create_model('vit_base_patch16_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vit_url.npz"
        self.model = timm.create_model('vit_base_patch16_384', pretrained=False, num_classes=1, pretrained_cfg=pretrained_cfg)

    def forward(self, x):
            x = self.model(x)
            return x




class Vit_token_fixed_single(nn.Module):
    def __init__(self):
        super(Vit_token_fixed_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vit_base_patch16_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vit_url.npz"
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)
        
        modules = list(self.model.named_modules())
        last_four_modules = [name for name, _ in modules[-4:]]
        for name, param in self.model.named_parameters():
            if not any(l in name for l in last_four_modules):
                param.requires_grad = False

    def forward(self, x):
            x = self.model(x)
            return x




#Multi
class Vit_token_fine_multi(nn.Module):
    def __init__(self):
        super(Vit_token_fine_multi, self).__init__()
        
        pretrained_cfg = timm.models.create_model('vit_base_patch16_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/vit_url.npz"
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['blocks.2', 'blocks.5', 'blocks.8', 'blocks.11']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.norm = self.model.norm
        self.fc_norm = self.model.fc_norm
        self.head_drop = self.model.head_drop
        self.head = self.model.head

    def forward(self, x):

        x = self.model.forward_features(x)
        ft_l1 = self.norm(self.media_features.outputs[0])
        x1 = ft_l1[:, 0]
        ft_l2 = self.norm(self.media_features.outputs[1])
        x2 = ft_l2[:, 0]
        ft_l3 = self.norm(self.media_features.outputs[2])
        x3 = ft_l3[:, 0]
        ft_l4 = self.norm(self.media_features.outputs[3])
        x4 = ft_l4[:, 0]
        self.media_features.clear()

        tensor_list = [x1, x2, x3, x4]
        final_feature = torch.stack(tensor_list, dim=0).mean(dim=0)
        x = self.fc_norm(final_feature)
        x = self.head_drop(x)
        x = self.head(x)
        return x





#swin基准模型
class Swin_avg_fine_single(nn.Module):
    def __init__(self):
        super(Swin_avg_fine_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('swin_base_patch4_window12_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/swin.bin"
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)

    def forward(self, x):
            x = self.model(x)
            return x




#train_type
class Swin_avg_end2end_single(nn.Module):
    def __init__(self):
        super(Swin_avg_end2end_single, self).__init__()
   
        pretrained_cfg = timm.models.create_model('swin_base_patch4_window12_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/swin.bin"
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=1, pretrained_cfg=pretrained_cfg)

    def forward(self, x):
            x = self.model(x)
            return x




class Swin_avg_fixed_single(nn.Module):
    def __init__(self):
        super(Swin_avg_fixed_single, self).__init__()
        
        pretrained_cfg = timm.models.create_model('swin_base_patch4_window12_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/swin.bin"
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)

        modules = list(self.model.named_modules())
        last_eight_modules = [name for name, _ in modules[-8:]]
        for name, param in self.model.named_parameters():
            if not any(l in name for l in last_eight_modules):
                param.requires_grad = False

    def forward(self, x):
            x = self.model(x)
            return x




#Multi
class Swin_avg_fine_multi(nn.Module):
    def __init__(self):
        super(Swin_avg_fine_multi, self).__init__()
        
        pretrained_cfg = timm.models.create_model('swin_base_patch4_window12_384').default_cfg
        pretrained_cfg['file'] = r"checkpoint/hugging_face/swin.bin"
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=1, pretrained_cfg=pretrained_cfg)

        self.media_features = SaveOutput()
        layers = ['layers.0', 'layers.1', 'layers.2', 'layers.3']
        for name, layer in self.model.named_modules():
            for lname in layers:
                if name == lname:
                    layer.register_forward_hook(self.media_features)

        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # self.pool = self.model.head.global_pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(128)
        self.norm2 = norm_layer(256)
        self.norm3 = norm_layer(512)
        self.norm4 = norm_layer(1024)

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )


    def forward(self, x):

        x = self.model.forward(x)
        ft_l1 = self.norm1(self.media_features.outputs[0])
        ft_l2 = self.norm2(self.media_features.outputs[1])
        ft_l3 = self.norm3(self.media_features.outputs[2])
        ft_l4 = self.norm4(self.media_features.outputs[3])
        self.media_features.clear()

        ft_l1 = ft_l1.permute(0, 3, 1, 2)
        ft_l2 = ft_l2.permute(0, 3, 1, 2)
        ft_l3 = ft_l3.permute(0, 3, 1, 2)
        ft_l4 = ft_l4.permute(0, 3, 1, 2)

        ft_l2 = self.conv0(ft_l2)
        ft_l3 = self.conv1(ft_l3)
        ft_l4 = self.conv2(ft_l4)

        ft_l1, ft_l2, ft_l3, ft_l4 = self.pool(ft_l1), self.pool(ft_l2), self.pool(ft_l3), self.pool(ft_l4)
        x = torch.cat((ft_l1, ft_l2, ft_l3, ft_l4), dim=1).view(x.size(0), -1)

        x = self.mlp(x)
        return x




if __name__ == '__main__':

    # preprocess = transforms.Compose([
    #     transforms.Resize(800),
    #     transforms.CenterCrop((384, 384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    preprocess = transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize((384,384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))])

    image_path = r'F:\dataset\koniq10k_1024x768\1024x768/10004473376.jpg' 
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0).cuda() 
    model = InceptionResnetV2_spp_fine_multi().cuda()
    model_name = model.__class__.__name__
    print("Model Name:", model_name)
    a = model.forward(image)
    print(a.shape)


# # -> running
# * -> ok!
# ! -> error

# *Resnet50_gap_fine_single
# *Resnet50_gmp_fine_single
# *Resnet50_mix_fine_single
# *Resnet50_sp_fine_single
# *Resnet50_std_fine_single
# *Resnet50_gap_end2end_single
# *Resnet50_gap_fixed_single
# *Resnet50_gap_fine_multi
# *Vgg19_gap_fine_single
# *Vgg19_gmp_fine_single
# *Vgg19_mix_fine_single
# *Vgg19_sp_fine_single
# *Vgg19_std_fine_single
# #Vgg19_gap_end2end_single
# *Vgg19_gap_fixed_single
# *Vgg19_gap_fine_multi
# *InceptionResnetV2_gap_fine_single
# *InceptionResnetV2_gmp_fine_single
# *InceptionResnetV2_mix_fine_single
# *InceptionResnetV2_sp_fine_single
# *InceptionResnetV2_std_fine_single
# *InceptionResnetV2_gap_end2end_single
# *InceptionResnetV2_gap_fixed_single
# *InceptionResnetV2_gap_fine_multi
# *Vit_token_fine_single
# !Vit_avg_fine_single无法启动训练
# *Vit_token_end2end_single
# *Vit_token_fixed_single
# *Vit_token_fine_multi
# *Swin_avg_fine_single
# *Swin_avg_end2end_single
# *Swin_avg_fixed_single
# *Swin_avg_fine_multi
