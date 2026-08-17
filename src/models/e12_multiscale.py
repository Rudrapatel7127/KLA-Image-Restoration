import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DoubleConv, ResidualBlock, UpBlock


class AttentionBlock(nn.Module):
    """CBAM-like attention block that focuses on high-frequency detail preservation."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention: global pool -> mlp -> sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        # Spatial attention: avg+max pool -> concat -> conv -> sigmoid
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention
        avg_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        ch_att = self.mlp(avg_pooled).view(B, C, 1, 1)
        x = x * ch_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spa_input = torch.cat([avg_out, max_out], dim=1)
        sp_att = torch.sigmoid(self.spatial(spa_input))
        x = x * sp_att

        return x


class RRDBBlock(nn.Module):
    """Residual-in-Residual Dense Block for Stage 2."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out, inplace=True)


class Stage1CoarseNet(nn.Module):
    """Stage 1: Coarse restoration focusing on structure/edges/brightness.
    
    Input: 128×128 noisy LR
    Output: 256×256 coarse prediction
    Architecture: smaller encoder (32→16→8 channels), 3 pool/up cycles,
    produces residual at 128×128, upsampled to 256×256 via bilinear add.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        # Encoder: stride-2 convs reduce spatial dim by 2 each time
        # 128 → 64 → 32 → 16 (bottleneck)
        self.enc1 = DoubleConv(1, base_channels)      # 32 channels at 128→64 via stride
        # Actually DoubleConv uses padding=1, no stride change. Let me use explicit pool.
        self.pool1 = nn.MaxPool2d(2)                   # 128 → 64

        self.enc2 = DoubleConv(base_channels, base_channels * 2)  # 64 → 32
        self.pool2 = nn.MaxPool2d(2)                   # 64 → 32

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)  # 32 → 16
        self.pool3 = nn.MaxPool2d(2)                   # 32 → 16

        # Bottleneck at 16×16 with 128 channels
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.res_bottleneck = ResidualBlock(base_channels * 8)

        # Decoder: 3 up-sampling blocks (16 → 32 → 64 → 128)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)   # 16 → 32
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.dec3_res = ResidualBlock(base_channels * 4)

        self.up2 = UpBlock(base_channels * 4, base_channels * 2)   # 32 → 64
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.dec2_res = ResidualBlock(base_channels * 2)

        self.up1 = UpBlock(base_channels * 2, base_channels)       # 64 → 128
        self.dec1 = DoubleConv(base_channels * 2, base_channels)   # 128 → 64 (wait, this is wrong)
        # Actually DoubleConv(in=base_channels*2, out=base_channels) would be 128→64 if in is 128
        # But after up1, we have base_channels (32) from UpBlock output, then concat with skip e1
        # e1 is at 128 with base_channels (32) channels
        # After UpBlock, output is base_channels (32) at 128×128
        # Concat with e1 (32 channels at 128×128) → 64 channels at 128×128
        # Then dec1 should reduce from 64 to 32

        # Final layers: produce residual prediction at 128×128
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        # Output: 256×256 = bilinear upsample(128×128 coarse residual) × 2

        self._initialize_weights()

    def forward(self, x):
        # Encoder
        e1 = self.pool1(self.enc1(x))      # 128→64, 1→32 channels
        e2 = self.pool2(self.enc2(e1))     # 64→32, 32→64 channels
        e3 = self.pool3(self.enc3(e2))     # 32→16, 64→128 channels

        # Bottleneck
        b = self.res_bottleneck(self.bottleneck(e3))  # 16×16, 128 channels

        # Decoder
        d3 = self.up3(b)                      # 16→32, 128→64 channels
        d3 = torch.cat([d3, e3], dim=1)       # concat skip: 64+64=128 channels at 32×32
        d3 = self.dec3(d3)
        d3 = self.dec3_res(d3)

        d2 = self.up2(d3)                      # 32→64, 64→32 channels
        d2 = torch.cat([d2, e2], dim=1)       # concat skip: 32+64=96? wait e2 is 64 channels at 32×32
        # Hmm, after pool2, e2 has base_channels*2 = 64 channels at 32×32
        # After UpBlock output is base_channels*2=64 channels? No, UpBlock preserves channels conceptually
        # Actually UpBlock(in_channels, out_channels) - the output has out_channels.
        # UpBlock(base_channels*4, base_channels*2) takes 128→64 channels.
        # But after concat with e3 (64 channels from enc3), we have 128, then dec3 reduces to 64.
        # Let me be more careful.

        # Let me re-examine. After up3: output is base_channels*2 = 64 channels at 32×32
        # e3 after enc3 has base_channels*4 = 128 channels at 16×16
        # But we concat d3 (after dec3 which reduces to base_channels*4=128? no) with e3.
        # This is getting confusing. Let me just look at the actual tensor shapes.

        # Actually, let me simplify. The UpBlock does:
        # Conv2d(in, out*4, 3, padding=1) -> PixelShuffle(2) -> ReLU
        # So if input is N×C×H×W, output is N×(out/4)×(H*2)×(W*2) -- wait no.
        # PixelShuffle(2) with G groups: takes in=C×H×W, output= C/G × H*2 × W*2, where G = out_channels/4? 
        # Actually for UpBlock(in_channels, out_channels):
        # self.up = Conv2d(in, out*4, 3, padding=1) -> PixelShuffle(2) -> ReLU
        # PixelShuffle(2) rearranges: C_in → C_in/4 channels, 2x spatial upscale
        # So if in=128, out=64: Conv2d(128, 64*4=256), PixelShuffle(2) → 64 channels, 2x spatial
        # Output channels = in/4, spatial ×2

        # So UpBlock(base_channels*8=256, base_channels*4=128):
        # Input: 16×16×256, Output: 32×32×128
        # Then dec3 = DoubleConv(128, 64) -> output 64 channels at 32×32
        # Then concat with e3: e3 is at 16×16, but we're at 32×32 now. 
        # Problem: the spatial sizes don't match for concat!

        # I think the issue is that the skip connections need to be aligned.
        # In the original SRUNet and U-Net, the skips are from the same resolution as the decoder input.
        # After up3, we're at 32×32, and e3 should also be at 32×32 for concat to work.
        # But e3 is from the encoder pool3 output at 16×16.

        # The typical U-Net solution: the skip connections are from BEFORE the pool, so:
        # e1 is after pool1: 64×64
        # e2 is after pool2: 32×32  
        # e3 is after pool3: 16×16
        # And the decoder upsamples to match: up3 goes 16→32, concat with e2 (32×32)? No, e2 is 32×32 and up3 output is 32×32, that works!
        # Wait no: pool1: 128→64, e1 at 64×64
        # pool2: 64→32, e2 at 32×32
        # pool3: 32→16, e3 at 16×16
        # decoder: up3: 16→32, concat with e2 (32×32) ✓
        # up2: 32→64, concat with e1 (64×64) ✓ 
        # up1: 64→128, no concat (bottom of decoder)
        
        # OK so the skips are from the pool outputs, not the encoder conv outputs directly.
        # But in the code, e1 = self.pool1(self.enc1(x)) means e1 is AFTER pooling, at 64×64.
        # And e2 = self.pool2(self.enc2(e1)) at 32×32.
        # e3 = self.pool3(self.enc3(e2)) at 16×16.
        # Then decoder: up3 output at 32×32, concat with e2 at 32×32 ✓
        # up2 output at 64×64, concat with e1 at 64×64 ✓
        # up1 output at 128×128, no skip (input was at 128×128)

        # But wait, in the original two_stage.py code:
        # e1 = self.res1(self.enc1(x))  -- NO pool here
        # e2 = self.res2(self.enc2(self.pool1(e1)))  -- pool1 applied inside
        # e3 = self.res3(self.enc3(self.pool2(e2)))  -- pool2 inside
        # pool3 = nn.MaxPool2d(2) applied separately
        
        # So the skips are from after the conv but the pool is applied before the next enc layer.
        # The tensor shapes: x is 128×128, enc1 keeps 128×128 (padding=1), pool1 makes it 64×64.
        # enc2 takes 64×64, output 64×64, pool2 makes it 32×32.
        # enc3 takes 32×32, output 32×32, pool3 makes it 16×16.
        
        # Then decoder up3: 16→32, and we concat with e2 which is at 32×32 ✓
        # up2: 32→64, concat with e1 at 64×64 ✓
        # up1: 64→128, no concat

        # OK so my code structure should work. Let me just make sure the channel dimensions are right.

        # After up3: output is base_channels*4=128 channels? No.
        # UpBlock(256, 128): Conv2d(256, 128*4=512), PixelShuffle(2) → 128 channels at 32×32
        # Then dec3 = DoubleConv(128, 64): output 64 channels at 32×32
        # Concat with e2: e2 has 64 channels (base_channels*2) at 32×32 → total 128 channels
        # Then dec3_res on 128 channels...

        # Hmm, this is getting complex. Let me just look at what the actual channel counts should be.

        # Let me simplify: I'll use the same pattern as two_stage.py but with smaller channels.
        # two_stage.py uses base_channels=64 throughout, with:
        # enc1: DoubleConv(1, 64), enc2: DoubleConv(64, 128), enc3: DoubleConv(128, 256)
        # And the decoder uses UpBlock with matching channels.

        # For E12 with base_channels=32:
        # enc1: DoubleConv(1, 32), enc2: DoubleConv(32, 64), enc3: DoubleConv(64, 128)
        # And the decoder UpBlocks and Decoders should match.

        # Let me just rewrite this more carefully, keeping track of channel dimensions.

        # Actually, I realize I'm overthinking this. Let me just look at the existing two_stage.py
        # and adapt it, making sure the channel dimensions are consistent.

        # two_stage.py Denoiser with base_channels=64:
        # enc1 = DoubleConv(1, 64)    # input 1→64, 128×128 output
        # pool1 = MaxPool2d(2)        # 128→64
        # enc2 = DoubleConv(64, 128)  # input 64→128, 64×64 output
        # pool2 = MaxPool2d(2)        # 64→32
        # enc3 = DoubleConv(128, 256) # input 128→256, 32×32 output
        # pool3 = MaxPool2d(2)        # 32→16
        # bottleneck = DoubleConv(256, 512)  # 16×16×512
        
        # decoder:
        # up3 = UpBlock(512, 256)     # Input 512, output 256 channels at 16→32
        # dec3 = DoubleConv(512, 256) # Wait, UpBlock output is 256 channels, but it concatenates with e3
        # # e3 is 32×32 with 256 channels (from enc3)
        # # But up3 output is 32×32 with 256 channels
        # # Concatenated: 512 channels at 32×32
        # # But dec3 = DoubleConv(512, 256) - yes that matches!
        # # Actually looking at the code: self.dec3 = DoubleConv(base_channels * 4, base_channels * 2)
        # # base_channels=64: DoubleConv(256, 128) - but the concat gives 512...
        # # Hmm wait, let me re-read the two_stage.py more carefully.

        # From two_stage.py:
        # self.up3 = UpBlock(base_channels * 8, base_channels * 4)  # 512→256
        # self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)  # 512→256
        # # But the forward does: d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        # # e3 has base_channels*4 = 256 channels (from enc3 DoubleConv(128, 256))
        # # up3 output has base_channels*4 = 256 channels (from UpBlock(512, 256))
        # # cat dim=1: 256+256=512 channels
        # # dec3 takes 512 input → 256 output. ✓

        # OK so the pattern is clear. Let me now design E12 with base_channels=32.

        # E12 with base_channels=32:
        # enc1 = DoubleConv(1, 32)      # 1→32
        # pool1 → 64×64
        # enc2 = DoubleConv(32, 64)    # 64→64, output 64 channels at 64×64
        # pool2 → 32×32
        # enc3 = DoubleConv(64, 128)   # 128 channels at 32×32
        # pool3 → 16×16
        
        # bottleneck = DoubleConv(128, 256)  # 256 channels at 16×16
        
        # up3 = UpBlock(256, 64)       # UpBlock(in=256, out=64): Conv2d(256, 64*4=256), PixelShuffle(2)→ 64 channels at 16→32
        # dec3 = DoubleConv(256, 64)   # but wait, concat with e3...
        # # e3 has 128 channels (from enc3 DoubleConv(64, 128))
        # # up3 output has 64 channels
        # # cat: 64+128=192, but dec3 expects 256. MISMATCH!
        
        # Hmm, the issue is the UpBlock output channels don't match the skip connections.
        # In two_stage.py with base_channels=64:
        # UpBlock(512, 256): output 256 channels
        # e3 has 256 channels (from enc3 DoubleConv(128, 256))
        # cat: 256+256=512
        # dec3 = DoubleConv(512, 256)? No, two_stage.py has dec3 = DoubleConv(base_channels*8, base_channels*4) = DoubleConv(512, 256). ✓
        
        # Wait, but the UpBlock documentation says: "Conv2d(in, out*4, 3, padding=1), PixelShuffle(2), ReLU"
        # So UpBlock(512, 256): Conv2d(512, 256*4=1024), PixelShuffle(2) → 256 channels
        # That's correct.

        # Now for E12 with base_channels=32:
        # enc1 = DoubleConv(1, 32)     # output 32 ch at 128×128
        # pool1 → 64×64
        # enc2 = DoubleConv(32, 64)    # output 64 ch at 64×64
        # pool2 → 32×32
        # enc3 = DoubleConv(64, 128)   # output 128 ch at 32×32
        # pool3 → 16×16
        # bottleneck = DoubleConv(128, 256)  # 256 ch at 16×16
        
        # up3 = UpBlock(256, 64)       # Conv2d(256, 64*4=256), PixelShuffle(2)→ 64 ch at 16→32
        # # e3 has 128 ch at 32×32 (from enc3)
        # # up3 output has 64 ch at 32×32
        # # cat dim=1: 64+128=192
        # # But dec3 needs to take 192 input. 
        # # two_stage.py dec3 = DoubleConv(base_channels*8, base_channels*4) = DoubleConv(512, 256) for base_channels=64
        # # For base_channels=32: dec3 should be DoubleConv(192, 64)? Or should I adjust the UpBlock output channels?
        
        # The pattern from two_stage.py: UpBlock(in=base_channels*8, out=base_channels*4)
        # and dec3 = DoubleConv(base_channels*8, base_channels*4)
        # The concat gives: base_channels*4 (from UpBlock) + base_channels*4 (from skip e3) = base_channels*8
        # And dec3 takes base_channels*8 input → base_channels*4 output. ✓

        # So for E12 base_channels=32:
        # UpBlock(256, 128): output 128 channels (since 256/4=64... wait)
        # UpBlock(in_channels, out_channels): Conv2d(in, out*4, 3, padding=1), PixelShuffle(2)
        # Output channels = in/4. So UpBlock(256, 128): output = 256/4 = 64 channels? No...
        # Let me re-read: Conv2d(in, out*4, 3, padding=1), then PixelShuffle(2).
        # PixelShuffle(2) with G groups: input C × H × W, output (C/G) × (H*2) × (W*2), where G = out_channels/... 
        # Actually, for PixelShuffle with scale=2: input has C channels, output has C/4 channels, spatial ×2.
        # The formula: if we want output to have out_channels, we need Conv2d(in, out*4, 3, padding=1)
        # because PixelShuffle(2) divides channels by 4.
        
        # So UpBlock(in_channels=256, out_channels=128):
        # Conv2d(256, 128*4=512), PixelShuffle(2) → 128 channels. ✓
        
        # And e3 for base_channels=32: enc3 = DoubleConv(64, 128) → output 128 channels. ✓
        # cat: 128 (UpBlock) + 128 (e3) = 256 channels
        # dec3 = DoubleConv(256, 128) → takes 256 input, outputs 128. ✓ (matches base_channels*8=256? No, base_channels*8=256, and dec3 output is 128=base_channels*4. Hmm.)
        
        # two_stage.py: dec3 = DoubleConv(base_channels*8, base_channels*4)
        # For base_channels=64: DoubleConv(512, 256). The concat gives 512, output 256. ✓
        # For base_channels=32: I need DoubleConv(256, 128) if concat gives 256. ✓
        
        # OK so the pattern is:
        # UpBlock(base_channels*8, base_channels*4) 
        # # Output channels = base_channels*4 (since in/4 = (base_channels*8)/4 = base_channels*2... wait that's wrong)
        
        # Let me just carefully compute:
        # UpBlock(in_C, out_C): Conv2d(in_C, out_C*4, 3, padding=1), PixelShuffle(2)
        # PixelShuffle(2): takes C_in → C_in/4 channels, ×2 spatial
        # So output channels = in_C / 4
        # For UpBlock(256, 128): output = 256/4 = 64 channels, NOT 128. 
        
        # Hmm, that means the two_stage.py code doesn't work as I thought.
        # Let me re-read two_stage.py:
        # self.up3 = UpBlock(base_channels * 8, base_channels * 4)  # base_channels=64: UpBlock(512, 256)
        # # UpBlock(512, 256): Conv2d(512, 256*4=1024), PixelShuffle(2) → 256 channels. ✓
        # # 512/4 = 128? No wait, 1024/4 = 256. The Conv2d output is 256*4=1024, then PixelShuffle divides by 4 → 256. ✓
        # # So output channels = out_C = 256. The out_C parameter IS the output channels.
        
        # # How? PixelShuffle(2) with scale=2: input (N, C, H, W) → output (N, C/4, H*2, W*2)
        # # But if we set Conv2d weight such that the effective output after PixelShuffle is out_C...
        # # Actually, I think the way PixelShuffle works: 
        # # We have weight of shape (out*4, in, k, k). After conv, we have (out*4, H, W) feature maps.
        # # PixelShuffle(2) rearranges: each group of 4 channels becomes 1 channel at 2x spatial.
        # # So with G=4 groups, we have out groups, each producing 1 channel at upscaled spatial.
        # # Total output channels = out.
        
        # # OK so the formula is: UpBlock(in, out) produces output with 'out' channels.
        # # And the concat in the forward: torch.cat([self.up3(b), e3], dim=1)
        # # e3 has base_channels*4 channels (from enc3 DoubleConv(base_channels*2, base_channels*4))
        # # up3 output has base_channels*4 channels (from UpBlock parameter)
        # # cat gives base_channels*8 channels
        # # dec3 = DoubleConv(base_channels*8, base_channels*4) reduces to base_channels*4

        # # For base_channels=64: 
        # # UpBlock(512, 256) → 256 output channels ✓
        # # e3: 256 channels ✓
        # # cat: 512 channels ✓
        # # dec3: DoubleConv(512, 256) → 256 output ✓

        # # For base_channels=32:
        # # UpBlock(256, 128) → 128 output channels ✓
        # # e3: enc3 = DoubleConv(64, 128) → 128 channels ✓
        # # cat: 128+128=256 channels
        # # dec3 = DoubleConv(256, 128) → 128 output ✓ (base_channels*8=256, base_channels*4=128)

        # # Great, now I understand the pattern.

        # Let me now properly write the E12 Stage1 model with base_channels=32.

        # enc1 = DoubleConv(1, 32)           # 1→32, 128×128
        # pool1 → 64×64
        # enc2 = DoubleConv(32, 64)         # 64→64, 64×64
        # pool2 → 32×32
        # enc3 = DoubleConv(64, 128)        # 128 ch, 32×32
        # pool3 → 16×16
        # bottleneck = DoubleConv(128, 256)  # 256 ch, 16×16

        # up3 = UpBlock(256, 128)           # 128 output ch, 16→32
        # dec3 = DoubleConv(256, 128)       # takes concat(128+128=256) → 128 output
        # # cat: up3(128) + e3(128) = 256, dec3 takes 256 → 128 ✓
        
        # up2 = UpBlock(128, 64)            # 64 output ch, 32→64
        # # e2 has 64 channels (from enc2 DoubleConv(32, 64))
        # # cat: up2(64) + e2(64) = 128
        # dec2 = DoubleConv(128, 64)        # takes 128 → 64 ✓ (base_channels*8=256? No, 128=base_channels*4 for base_channels=32)
        
        # up1 = UpBlock(64, 32)             # 32 output ch, 64→128
        # # e1 has 32 channels (from enc1 DoubleConv(1, 32))
        # # cat: up1(32) + e1(32) = 64
        # # But wait, do we concat with e1? In the original two_stage.py forward:
        # # d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # # Yes, always concat with the corresponding skip.
        # dec1 = DoubleConv(64, 32)         # takes concat(32+32=64) → 32 ✓

        # final_conv = nn.Conv2d(32, 1, 3, padding=1)  # 1 channel at 128×128
        # # Output residual at 128×128, then upsample to 256×256

        # This all makes sense now. Let me rewrite the model properly.
  
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.pool1(self.enc1(x))      # 128→64, 32 channels
        e2 = self.pool2(self.enc2(e1))     # 64→32, 64 channels
        e3 = self.pool3(self.enc3(e2))     # 32→16, 128 channels

        # Bottleneck
        b = self.res_bottleneck(self.bottleneck(e3))  # 16×16, 256 channels

        # Decoder
        d3 = self.up3(b)                      # 16→32, 128 output channels
        d3 = torch.cat([d3, e3], dim=1)       # 128+128=256 channels at 32×32
        d3 = self.dec3(d3)
        d3 = self.dec3_res(d3)

        d2 = self.up2(d3)                      # 32→64, 64 output channels
        d2 = torch.cat([d2, e2], dim=1)       # 64+64=128 channels at 64×64
        d2 = self.dec2(d2)
        d2 = self.dec2_res(d2)

        d1 = self.up1(d2)                       # 64→128, 32 output channels
        d1 = torch.cat([d1, e1], dim=1)       # 32+32=64 channels at 128×128
        d1 = self.dec1(d1)
        d1 = self.dec1_res(d1)

        # Final conv: 1 channel residual at 128×128
        residual_128 = self.final_conv(d1)    # 1 channel, 128×128

        # Upsample to 256×256
        coarse = F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False) + residual_128  # Wait, this would add the residual to its own upsample. Let me think.
        # Actually, the design says Stage1 outputs 256×256 coarse prediction.
        # The residual_128 is the predicted detail at 128×128. The coarse prediction should be:
        # - The bilinearly-upsampled original input + residual
        # OR just the upsampled residual
        # 
        # Looking at the original SRUNet: out = F.interpolate(x, scale_factor=2) + residual
        # So the output is the original LR upsampled + predicted residual.
        # For E12 Stage 1, the "coarse prediction" should be structure + edges.
        # Let me output: bilinear upsample of the 128×128 prediction + residual refinement.
        
        # Actually, I think the simplest: the coarse prediction is just the residual_128 upsampled to 256×256.
        # Or: coarse = F.interpolate(x, scale_factor=2) + residual_128 (but x is 128×128 input, upsampled to 256×256)
        # That would be: base = LR upsampled, residual = predicted detail, output = base + residual = 256×256
        
        # But wait, the input x is the noisy LR. The coarse prediction should be the model's best guess of the clean HR.
        # So: coarse = F.interpolate(x, scale_factor=2) + residual_128 upsampled? No.
        # Let me just output the residual upsampled, or output base + residual.
        
        # I think the cleanest: the Stage1 output is the coarse HR prediction.
        # It should be: base (bilinear upsample of LR) + predicted residual.
        # But we don't have the clean base easily. Let me just do:
        # coarse = F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False)
        # This gives 256×256 but it's just the residual upsampled, not a meaningful "coarse prediction".
        
        # Actually, looking at the E12 design notes again:
        # "Stage 1 — Coarse Restoration: Output: 256×256 coarse prediction (focuses on structure, edges, brightness)"
        # So the output should be a meaningful 256×256 image focusing on structure.
        # 
        # The simplest approach: the model predicts a residual at 128×128, and the coarse output is
        # the bilinearly-upsampled original input + the residual. This is what the original SRUNet does.
        # But we need the "original input" in the forward. Since the forward receives the noisy LR,
        # we can do: base = F.interpolate(x, scale_factor=2), coarse = base + F.interpolate(residual_128, scale_factor=2)
        # Or simply: coarse = F.interpolate(residual_128 + x_upsampled, ...)
        
        # Hmm, let me just keep it simple and have Stage1 output the residual-prediction-upsampled,
        # and let the Stage2 refinement work on top of it. The "coarse" label is just terminology.
        
        # Actually I'll just have Stage1 output: F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False)
        # giving 256×256. The "coarse" nature will be validated by the training results.
        
        # No wait, let me look at what makes sense. The residual_128 is the model's prediction of the detail
        # that's missing from the 128×128 input. If I upsample it, I get high-frequency detail at 256×256.
        # But the "coarse" prediction should capture the low-frequency structure.
        # 
        # I think the right approach is: the Stage1 output = F.interpolate(x, scale_factor=2) + F.interpolate(residual_128, scale_factor=2)
        # This gives the bilinearly-upsampled LR (which has some structure) + the predicted high-frequency detail.
        # The result is a 256×256 image that has both coarse structure and some detail.
        
        # But actually, the residual_128 already contains both low and high frequency content (it's the model's prediction
        # of what's needed to go from the noisy LR to the clean HR). The bilinear upsample of the residual adds detail.
        # 
        # Let me just do: coarse = F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False) + F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # No that doesn't make sense either.
        
        # OK let me simplify drastically. I'll have Stage1 produce the 256×256 output as:
        # coarse = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + residual_128_upsampled
        # where residual_128_upsampled = F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False)
        # And residual_128 = self.final_conv(d1) is the predicted detail at 128×128.
        
        # This matches the SRUNet pattern: out = base + residual, where base = LR×2 bilinear.
        
        coarse = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) + F.interpolate(residual_128, scale_factor=2, mode="bilinear", align_corners=False)
        return coarse

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class Stage2FineNet(nn.Module):
    """Stage 2: Fine-detail denoising.
    
    Input: [original 128×128 LR, Stage1 256×256 coarse output] 
    But we need to resize LR to 256×256 for concat, or coarse to 128×128.
    Design spec: input = [original LR, Stage1 output] concatenated.
    We'll upsample the 128×128 LR to 256×256, then concat with Stage1 output.
    Output: refined prediction, skip connection: Stage1 output + Stage2 refinement.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        # Input: 2 channels at 256×256 [upsampled LR, Stage1 coarse]
        self.conv_init = nn.Conv2d(2, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels)
        
        # Attention block
        self.attention = AttentionBlock(base_channels)
        
        # RRDB stacking for fine-detail removal
        self.rrdb1 = RRDBBlock(base_channels)
        self.rrdb2 = RRDBBlock(base_channels)
        self.rrdb3 = RRDBBlock(base_channels)
        
        # Output: predict residue to add to Stage1
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        
        self._initialize_weights()

    def forward(self, lr_input, coarse_output):
        # lr_input: 128×128 noisy LR
        # coarse_output: 256×256 from Stage1
        
        # Upsample LR to 256×256
        lr_upscaled = F.interpolate(lr_input, scale_factor=2, mode="bilinear", align_corners=False)
        
        # Concatenate: 2 channels at 256×256
        x = torch.cat([lr_upscaled, coarse_output], dim=1)
        
        # Initial conv and residual block
        x = self.conv_init(x)
        x = self.res1(x)
        
        # Attention
        x = self.attention(x)
        
        # RRDB blocks
        x = self.rrdb1(x)
        x = self.rrdb2(x)
        x = self.rrdb3(x)
        
        # Predict residue to add to Stage1
        residue = self.final_conv(x)  # 1 channel at 256×256
        
        # Refine: Stage1 output + residue
        refined = coarse_output + residue
        
        return refined

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class E12MultiscaleModel(nn.Module):
    """E12: Multi-scale two-stage restoration architecture.
    
    Stage 1: Coarse structure recovery (128×128 LR → 256×256 coarse prediction)
    Stage 2: Fine-detail denoising ( [LR + Stage1] → refined 256×256 )
    Skip connection: Stage1 output + Stage2 refinement (residual learning)
    """
    def __init__(self):
        super().__init__()
        self.stage1 = Stage1CoarseNet(base_channels=32)
        self.stage2 = Stage2FineNet(base_channels=32)

    def forward(self, x):
        # x: 128×128 noisy LR
        # Stage 1: coarse structure recovery
        coarse = self.stage1(x)  # 256×256
        
        # Stage 2: fine-detail denoising
        # Save the original LR for Stage2 concatenation
        self._last_lr = x
        refined = self.stage2(x, coarse)
        
        # Skip connection: Stage1 output + Stage2 refinement (residual learning)
        # Both are 256×256
        out = coarse + refined
        
        # Clamp output to [0,1] and ensure finite (no NaN/Inf from residual pathways)
        out = torch.clamp(out, 0.0, 1.0)
        out = torch.where(torch.isfinite(out), out, torch.tensor(0.0, device=out.device))
        
        return out

    def get_stage1_output(self, x):
        """Return Stage1 output for inspection/analysis."""
        return self.stage1(x)

    def get_stage2_refinement(self, x, coarse):
        """Return Stage2 refinement for inspection."""
        return self.stage2(x, coarse)