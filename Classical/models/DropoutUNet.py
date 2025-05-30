import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision import models
from utils.helpers import load_config

config = load_config()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)

class DropoutUNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Set a fixed random seed for reproducibility
        config = load_config()
        torch.manual_seed(config.seed)
        
        # resnet = models.resnet34(pretrained=True)
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # Encoder (ResNet34 backbone)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Output: [64, H/2, W/2]
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # Output: [64, H/4, W/4]
        self.enc3 = resnet.layer2                                         # Output: [128, H/8, W/8]
        self.enc4 = resnet.layer3                                         # Output: [256, H/16, W/16]
        self.enc5 = resnet.layer4                                         # Output: [512, H/32, W/32]

        # Decoder
        self.dec4 = ConvBlock(512 + 256, 256, dropout_rate)
        self.dec3 = ConvBlock(256 + 128, 128, dropout_rate)
        self.dec2 = ConvBlock(128 + 64, 64, dropout_rate)
        self.dec1 = ConvBlock(64 + 64, 64, dropout_rate)

        # split head for final prediction
        # mean
        self.mean = nn.Conv2d(64, 1, kernel_size=1)
        # log_var
        self.log_var = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with interpolation instead of cropping
        d4 = self.dec4(torch.cat([
            F.interpolate(e5, size=e4.shape[-2:], mode='bilinear', align_corners=False),
            e4
        ], dim=1))

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False),
            e2
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False),
            e1
        ], dim=1))

        mean = self.mean(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        # Output non-negative mean depth values and aleatoric uncertainty
        mean = torch.sigmoid(mean)*10
        log_var = self.log_var(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return torch.cat([mean, log_var], dim=1)


class DropoutUNet_ResNet50(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Set a fixed random seed for reproducibility
        config = load_config()
        torch.manual_seed(config.seed)
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Encoder (ResNet50 backbone)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Output: [64, H/2, W/2]
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # Output: [64, H/4, W/4]
        self.enc3 = resnet.layer2                                         # Output: [128, H/8, W/8]
        self.enc4 = resnet.layer3                                         # Output: [256, H/16, W/16]
        self.enc5 = resnet.layer4                                         # Output: [512, H/32, W/32]

        # Decoder
        self.dec4 = ConvBlock(512 + 256, 256, dropout_rate)
        self.dec3 = ConvBlock(256 + 128, 128, dropout_rate)
        self.dec2 = ConvBlock(128 + 64, 64, dropout_rate)
        self.dec1 = ConvBlock(64 + 64, 64, dropout_rate)

        # mean
        self.mean = nn.Conv2d(64, 1, kernel_size=1)
        # log_var
        self.log_var = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with interpolation instead of cropping
        d4 = self.dec4(torch.cat([
            F.interpolate(e5, size=e4.shape[-2:], mode='bilinear', align_corners=False),
            e4
        ], dim=1))

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False),
            e2
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False),
            e1
        ], dim=1))

        mean = self.mean(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        # Output non-negative mean depth values and aleatoric uncertainty
        mean = torch.sigmoid(mean)*10
        log_var = self.log_var(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return torch.cat([mean, log_var], dim=1)

class DropoutUNet_ResNet101(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)

        # Encoder (ResNet101 backbone)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Output: [64, H/2, W/2]
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # Output: [256, H/4, W/4]
        self.enc3 = resnet.layer2                                         # Output: [512, H/8, W/8]
        self.enc4 = resnet.layer3                                         # Output: [1024, H/16, W/16]
        self.enc5 = resnet.layer4                                         # Output: [2048, H/32, W/32]

        # Decoder
        self.dec4 = ConvBlock(2048 + 1024, 1024, dropout_rate)
        self.dec3 = ConvBlock(1024 + 512, 512, dropout_rate)
        self.dec2 = ConvBlock(512 + 256, 256, dropout_rate)
        self.dec1 = ConvBlock(256 + 64, 64, dropout_rate)

        deep_supervision = config.model.deep_supervision
        if deep_supervision:
            # supervision heads for auxiliary outputs
            decoder_channels = [64, 256]
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(ch, 2, kernel_size=1) for ch in decoder_channels
            ])

        # split head for final prediction
        # mean
        self.mean = nn.Conv2d(64, 1, kernel_size=1)
        # log_var
        self.log_var = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with interpolation instead of cropping
        d4 = self.dec4(torch.cat([
            F.interpolate(e5, size=e4.shape[-2:], mode='bilinear', align_corners=False),
            e4
        ], dim=1))

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False),
            e2
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False),
            e1
        ], dim=1))

        if deep_supervision := config.model.deep_supervision:
            aux_outputs = []
            for i, (d, head) in enumerate(zip([d1, d2], self.aux_heads)):
                out = head(F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False))
                mean = torch.sigmoid(out[:, 0:1]) * 10  # constrain range
                log_var = out[:, 1:2]
                aux_outputs.append(torch.cat([mean, log_var], dim=1))

        # Final prediction
        mean = self.mean(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        # Output non-negative mean depth values and aleatoric uncertainty
        mean = torch.sigmoid(mean)*10
        log_var = self.log_var(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))

        if deep_supervision:
            return torch.cat([mean, log_var], dim=1), aux_outputs
        
        return torch.cat([mean, log_var], dim=1)