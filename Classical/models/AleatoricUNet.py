import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
class AleatoricUNet(nn.Module):
    def __init__(self):
        super(AleatoricUNet, self).__init__()

        # Encoder blocks
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.enc5 = UNetBlock(512, 1024)

        # Center block
        self.center = UNetBlock(1024, 2048)
        # self.center = UNetBlock(512, 1024)

        # Decoder blocks
        self.dec5 = UNetBlock(2048 + 1024, 1024)
        self.dec4 = UNetBlock(1024 + 512, 512)
        self.dec3 = UNetBlock(512 + 256, 256)
        self.dec2 = UNetBlock(256 + 128, 128)
        self.dec1 = UNetBlock(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        e5 = self.enc5(self.pool(e4)) # [B, 1024, H/16, W/16]

        # Center
        center = self.center(e5) # [B, 2048, H/16, W/16]
        # center = self.center(e4)
        # [B, 1024, H/8, W/8]

        # Decoder
        d5 = self.dec5(torch.cat([
            F.interpolate(center, size=e5.shape[-2:], mode='bilinear', align_corners=False),
            e5
        ], dim=1))
        d4 = self.dec4(torch.cat([
            F.interpolate(d5, size=e4.shape[-2:], mode='bilinear', align_corners=False),
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

        x = self.final(d1) # [B, 2, H, W]
        
        # Output non-negative mean depth values and aleatoric uncertainty
        mean = torch.sigmoid(x[:,0:1, :, :])*10
        log_var = x[:,1:2, :, :]
        return torch.cat([mean, log_var], dim=1)

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Encoder blocks
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.enc5 = UNetBlock(512, 1024)

        # Center block
        self.center = UNetBlock(1024, 2048)

        # Decoder blocks
        self.dec5 = UNetBlock(2048 + 1024, 1024)
        self.dec4 = UNetBlock(1024 + 512, 512)
        self.dec3 = UNetBlock(512 + 256, 256)
        self.dec2 = UNetBlock(256 + 128, 128)
        self.dec1 = UNetBlock(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        e5 = self.enc5(self.pool(e4)) # [B, 1024, H/16, W/16]

        # Center
        center = self.center(e5) # [B, 2048, H/16, W/16]

        # Decoder
        d5 = self.dec5(torch.cat([
            F.interpolate(center, size=e5.shape[-2:], mode='bilinear', align_corners=False),
            e5
        ], dim=1))
        d4 = self.dec4(torch.cat([
            F.interpolate(d5, size=e4.shape[-2:], mode='bilinear', align_corners=False),
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

        x = self.final(d1) # [B, 1, H, W]
        
        # Output non-negative mean depth values
        x = torch.sigmoid(x[:,0:1, :, :])*10
        return x