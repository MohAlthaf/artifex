"""
SGRGANGenerator - Van Gogh Art Restoration Model
Exact architecture matching the trained baseline_best.pth checkpoint
With BiSCCFormer blocks (windowed attention) and FocalBlocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BiSCCFormer Components (Memory-Efficient Windowed Attention)
# ============================================================================

class BiLevelRoutingAttention(nn.Module):
    """
    Bi-level routing attention from SGRGAN paper
    Uses windowed attention to avoid O(N²) memory explosion
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def forward(self, x):
        B, H, W, C = x.shape
        window_size = min(self.window_size, H, W)
        
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        
        x_windows = self.window_partition(x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)
        
        B_w = x_windows.shape[0]
        N_w = window_size * window_size
        
        qkv = self.qkv(x_windows).reshape(B_w, N_w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        out = self.proj(out)
        
        out = out.view(-1, window_size, window_size, C)
        out = self.window_reverse(out, window_size, Hp, Wp)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :].contiguous()
        
        return out


class SCConv(nn.Module):
    """Spatial-Channel Convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 
                                     kernel_size=3, padding=1, groups=in_channels)
        self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.channel_conv(x)
        return x


class BiSCCFormerBlock(nn.Module):
    """BiSCCFormer Block combining attention with SCConv"""
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BiLevelRoutingAttention(dim, num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.scconv = SCConv(dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Attention branch
        x_attn = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_attn = x_attn + self.attn(self.norm1(x_attn))
        x_attn = x_attn.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # SCConv branch
        x_out = x_attn + self.scconv(self.norm2(x_attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x_out


# ============================================================================
# Focal Block Components (Multi-scale Dilated Convolutions)
# ============================================================================

class FocalModulation(nn.Module):
    """Focal modulation for hierarchical context aggregation"""
    def __init__(self, dim):
        super().__init__()
        self.focal_layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim)
        ])
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.modulation = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
    
    def forward(self, x):
        focal_features = []
        for layer in self.focal_layers:
            focal_features.append(layer(x))
        
        focal_cat = torch.cat(focal_features, dim=1)
        focal_fused = self.fusion(focal_cat)
        modulator = self.modulation(focal_fused)
        out = x * modulator + focal_fused
        
        return out


class FocalBlock(nn.Module):
    """Focal Block for structure encoder"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.focal_mod = FocalModulation(dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )
    
    def forward(self, x):
        x = x + self.focal_mod(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Main Generator (SGRGANGenerator)
# ============================================================================

class SGRGANGenerator(nn.Module):
    """
    Complete SGRGAN Generator with:
    - BiSCCFormer blocks (texture encoder)
    - Focal blocks (structure encoder)
    - Dual-branch architecture
    - Multi-head attention fusion
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        
        # === TEXTURE ENCODER (BiSCCFormer blocks) ===
        self.texture_enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(64, num_heads=4)
        )
        self.texture_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(128, num_heads=4)
        )
        self.texture_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(256, num_heads=8)
        )
        self.texture_enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            BiSCCFormerBlock(512, num_heads=8)
        )
        
        # === STRUCTURE ENCODER (Focal blocks) ===
        self.structure_enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(64)
        )
        self.structure_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(128)
        )
        self.structure_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(256)
        )
        self.structure_enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            FocalBlock(512)
        )
        
        # === FUSION MODULE ===
        self.fusion_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # === DECODER ===
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # === OUTPUT ===
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        """
        x: (B, 3, 512, 512) corrupted image
        mask: (B, 1, 512, 512) damage mask
        """
        # Texture encoder
        inp = torch.cat([x, mask], dim=1)
        t1 = self.texture_enc1(inp)
        t2 = self.texture_enc2(t1)
        t3 = self.texture_enc3(t2)
        t4 = self.texture_enc4(t3)
        
        # Structure encoder
        s1 = self.structure_enc1(mask)
        s2 = self.structure_enc2(s1)
        s3 = self.structure_enc3(s2)
        s4 = self.structure_enc4(s3)
        
        # Attention fusion
        B, C, H, W = t4.shape
        t4_flat = t4.view(B, C, -1).permute(2, 0, 1)
        s4_flat = s4.view(B, C, -1).permute(2, 0, 1)
        
        fused_flat, _ = self.fusion_attention(t4_flat, s4_flat, s4_flat)
        fused = fused_flat.permute(1, 2, 0).view(B, C, H, W)
        
        combined = torch.cat([t4, fused], dim=1)
        fused_features = self.fusion_conv(combined)
        
        # Decoder
        d1 = self.dec1(fused_features)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        
        out = self.output(d4)
        return out


# Alias for backward compatibility with app.py
BrushstrokeAwareGenerator = SGRGANGenerator
