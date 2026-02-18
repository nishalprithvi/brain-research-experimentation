
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.time_mlp = nn.Linear(time_dim, out_channels)

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        return x + time_emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # In this implementation, 'in_channels' is the output channel count of the PREVIOUS layer (before concat).
        # But we concat. So the convolution takes in_channels + skip_channels.
        # My implementation of Up here seems to assume in_channels is the concatenated size. Not quite.
        # Let's clean up standard UNet Up block.
        # Up(512, 256): Upsample 512 -> 256. Concat with 256 -> 512. Conv 512 -> 256.
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels // 2) # Adjusted. 
        self.time_mlp = nn.Linear(time_dim, out_channels // 2)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        return x + time_emb

# Re-implementing a cleaner lightweight UNet class
# ConvergenceUNet (16x16 input)
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Encoder
        # 16x16
        self.inc = DoubleConv(in_channels, 64)
        
        # 16->8
        self.down1 = Down(64, 128, time_dim)
        # 8->4
        self.down2 = Down(128, 256, time_dim)
        # 4->2
        self.down3 = Down(256, 512, time_dim) 
        
        # Decoder
        # 2->4. Input 512. Skip 256.
        # My Up block logic:
        # Up(in_ch, out_ch):
        # x = up(x1) -> ch same? No upsample usually preserves ch or reduces?
        # Upsample is spatial. 
        # Let's simplify. Standard UNet logic:
        
        self.up1 = UpSimple(512, 256, time_dim) # 512 in (from down3), 256 out. (Concat with 256 from down2 -> 512 total input to conv)
        self.up2 = UpSimple(256, 128, time_dim)
        self.up3 = UpSimple(128, 64, time_dim)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        x1 = self.inc(x)       # 64, 16x16
        x2 = self.down1(x1, t) # 128, 8x8
        x3 = self.down2(x2, t) # 256, 4x4
        x4 = self.down3(x3, t) # 512, 2x2
        
        x = self.up1(x4, x3, t) # 256, 4x4
        x = self.up2(x, x2, t)  # 128, 8x8
        x = self.up3(x, x1, t)  # 64, 16x16
        
        return self.outc(x)

class UpSimple(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        # in_channels is the distinct channel count of the lower feature map (e.g. 512).
        # We upsample it to match upper map (which has out_channels, e.g. 256).
        # Then we concat -> 512+256? No. 
        # Standard UNet: Down doubles channels. Up halves them.
        # x4 has 512. x3 has 256.
        # Upsample x4 -> 512. Concat with x3 -> 768? 
        # Usually we reduce channel count during upsampling OR after concatenation.
        # Torch UNet recommendation: ConvTranspose2d (halves channels) -> Concat -> DoubleConv.
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels) # in_ch (512/2=256) + skip (256) = 512.
        self.time_mlp = nn.Linear(time_dim, out_channels)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # x1 is now compatible with x2
        
        # Pad logic
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        # Add time
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        return x + time_emb

if __name__ == "__main__":
    model = DiffusionUNet()
    x = torch.randn(10, 1, 16, 16)
    t = torch.randint(0, 1000, (10,))
    out = model(x, t)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
