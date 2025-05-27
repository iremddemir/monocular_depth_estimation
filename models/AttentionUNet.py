import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.resnet import ResNet34_Weights
from torchvision import models
from utils.helpers import load_config

config = load_config()

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder input
        # x: encoder feature

        # Ensure g has same spatial size as x
        if g.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=g.shape[2:], mode='bilinear', align_corners=False)
            # print(f"Resized x from {x.shape} to {g.shape}")

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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

class AttentionUNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
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

        # Attention Gates
        self.attn4 = AttentionGate(F_g=512, F_l=256, F_int=128)  # Gating enc4 with enc5
        self.attn3 = AttentionGate(F_g=256, F_l=128, F_int=64)   # Gating enc3 with dec4
        self.attn2 = AttentionGate(F_g=128, F_l=64, F_int=32)    # Gating enc2 with dec3
        self.attn1 = AttentionGate(F_g=64,  F_l=64, F_int=32)    # Gating enc1 with dec2

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

        # Decoder with attention gates
        g4 = self.attn4(g=e5, x=e4)
        d4 = self.dec4(torch.cat([F.interpolate(e5, size=g4.shape[-2:], mode='bilinear', align_corners=False), g4], dim=1))

        g3 = self.attn3(g=d4, x=e3)
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=g3.shape[-2:], mode='bilinear', align_corners=False), g3], dim=1))

        g2 = self.attn2(g=d3, x=e2)
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=g2.shape[-2:], mode='bilinear', align_corners=False), g2], dim=1))

        g1 = self.attn1(g=d2, x=e1)
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=g1.shape[-2:], mode='bilinear', align_corners=False), g1], dim=1))
        
        # Final prediction
        mean = self.mean(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        # Output non-negative mean depth values and aleatoric uncertainty
        mean = torch.sigmoid(mean)*10
        log_var = self.log_var(F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return torch.cat([mean, log_var], dim=1)


class AttentionUNet_ResNet101(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

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

        # Attention Gates (input features adjusted to match encoder)
        self.attn4 = AttentionGate(F_g=2048, F_l=1024, F_int=512)
        self.attn3 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.attn2 = AttentionGate(F_g=512,  F_l=256, F_int=128)
        self.attn1 = AttentionGate(F_g=256,  F_l=64,  F_int=64)

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

        # Decoder with attention gates
        g4 = self.attn4(g=e5, x=e4)
        d4 = self.dec4(torch.cat([F.interpolate(e5, size=g4.shape[-2:], mode='bilinear', align_corners=False), g4], dim=1))

        g3 = self.attn3(g=d4, x=e3)
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=g3.shape[-2:], mode='bilinear', align_corners=False), g3], dim=1))

        g2 = self.attn2(g=d3, x=e2)
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=g2.shape[-2:], mode='bilinear', align_corners=False), g2], dim=1))

        g1 = self.attn1(g=d2, x=e1)
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=g1.shape[-2:], mode='bilinear', align_corners=False), g1], dim=1))

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

