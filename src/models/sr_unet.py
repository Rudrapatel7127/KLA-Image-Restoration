import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DoubleConv, ResidualBlock, UpBlock


class SRUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = DoubleConv(1, 64)
        self.res1 = ResidualBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.res2 = ResidualBlock(128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.res3 = ResidualBlock(256)
        self.pool3 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = DoubleConv(256, 512)
        self.res_bottleneck = ResidualBlock(512)

        # -------------------------
        # Decoder
        # -------------------------

        self.up3 = UpBlock(512, 256)
        self.dec3 = DoubleConv(512, 256)
        self.dec3_res = ResidualBlock(256)

        self.up2 = UpBlock(256, 128)
        self.dec2 = DoubleConv(256, 128)
        self.dec2_res = ResidualBlock(128)

        self.up1 = UpBlock(128, 64)
        self.dec1 = DoubleConv(128, 64)
        self.dec1_res = ResidualBlock(64)

        # -------------------------
        # Final Super Resolution Head
        # -------------------------

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),          # 128x128 -> 256x256
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self._initialize_weights()

    def forward(self, x):
        # Encoder
        e1 = self.res1(self.enc1(x))
        e2 = self.res2(self.enc2(self.pool1(e1)))
        e3 = self.res3(self.enc3(self.pool2(e2)))

        # Bottleneck
        b = self.res_bottleneck(self.bottleneck(self.pool3(e3)))

        # Decoder with skip connections
        d3 = self.dec3_res(self.dec3(torch.cat([self.up3(b), e3], dim=1)))
        d2 = self.dec2_res(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1_res(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))

        # Bicubic/Bilinear baseline
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # Predict residual
        residual = self.final_conv(d1)

        # Final output
        out = base + residual

        # Keep output in valid range
        return torch.clamp(out, 0.0, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Zero-init the final residual conv so the network
                # starts as a pure bilinear upsampler (residual = 0).
                # This stabilizes early training and lets the model
                # learn to *correct* the baseline rather than fight
                # a random initialization.
                if m is self.final_conv[-1]:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    continue

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
