import torch
import torch.nn as nn

# --- Generator: U-Net ---
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def downsample(in_channels, out_channels, batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if batchnorm: layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def upsample(in_channels, out_channels, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if dropout: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (4 layers)
        self.down1 = downsample(3, 32, batchnorm=False) # 256 -> 128
        self.down2 = downsample(32, 64)                 # 128 -> 64
        self.down3 = downsample(64, 128)                # 64 -> 32
        self.down4 = downsample(128, 256)               # 32 -> 16 (bottleneck)
        
        # Decoder with Skip Connections
        self.up1 = upsample(256, 128, dropout=True)     # 16 -> 32
        self.up2 = upsample(256, 64)                    # 32 -> 64 (128 from up1 + 128 from down3)
        self.up3 = upsample(128, 32)                    # 64 -> 128 (64 from up2 + 64 from down2)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),         # 128 -> 256 (32 from up3 + 32 from down1)
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        
        return self.final(torch.cat([u3, d1], 1))

# --- Discriminator: PatchGAN ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def d_layer(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        # The input is the concatenated image (condition + image), so it has 6 channels
        self.model = nn.Sequential(
            d_layer(6, 32),
            d_layer(32, 64),
            d_layer(64, 128),
            d_layer(128, 256, stride=1),
            nn.Conv2d(256, 1, 4, 1, 1) # Output a patch of real/fake predictions
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))