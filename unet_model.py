import torch
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """A lightweight U-Net for 32x32 inpainting tasks."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base: int = 32,
    ) -> None:
        super().__init__()
        width = base

        self.enc1 = ConvBlock(in_channels, width)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = ConvBlock(width, width * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConvBlock(width * 2, width * 4)

        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)

        self.up1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)

        self.out_conv = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.tanh(self.out_conv(dec1))
