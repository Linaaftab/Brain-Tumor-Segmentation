# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.up(x)

class BrainTumorSegmentationModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=16):
        super(BrainTumorSegmentationModel, self).__init__()
        
        self.base_channels = base_channels
        
        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder (Upsampling path)
        self.up4 = UpConv(base_channels * 16, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = UpConv(base_channels * 8, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = UpConv(base_channels * 4, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = UpConv(base_channels * 2, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Final convolution
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        # Handle different input sizes with interpolation
        original_size = x.shape[2:]
        
        # Resize to expected size if needed (common size for 3D medical images)
        if original_size != (32, 64, 64):
            x = F.interpolate(x, size=(32, 64, 64), mode='trilinear', align_corners=True)
        
        # Encoder
        e1 = self.enc1(x)  # base_channels
        e2 = self.enc2(self.pool(e1))  # base_channels * 2
        e3 = self.enc3(self.pool(e2))  # base_channels * 4
        e4 = self.enc4(self.pool(e3))  # base_channels * 8
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4))  # base_channels * 16
        
        # Decoder with skip connections
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        output = self.final_conv(d1)
        
        # Resize back to original size if we resized the input
        if original_size != (32, 64, 64):
            output = F.interpolate(output, size=original_size, mode='trilinear', align_corners=True)
        
        return output

class SimpleBSCLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SimpleBSCLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        # Binary Cross Entropy + Soft Dice Loss
        bce_loss = self.bce_loss(pred, target)
        
        # Dice loss for each class
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = 0
        
        for i in range(target.shape[1]):
            intersection = (pred_sigmoid[:, i] * target[:, i]).sum()
            union = pred_sigmoid[:, i].sum() + target[:, i].sum()
            dice_loss += 1 - (2. * intersection + 1e-8) / (union + 1e-8)
        
        dice_loss /= target.shape[1]
        
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# Test the model
if __name__ == "__main__":
    # Test with different input sizes
    model = BrainTumorSegmentationModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test different input sizes
    test_sizes = [
        (1, 4, 32, 64, 64),    # Expected size
        (1, 4, 128, 128, 128), # BraTS size
        (1, 4, 32, 128, 128),  # Mixed size
    ]
    
    for size in test_sizes:
        try:
            x = torch.randn(*size)
            output = model(x)
            print(f"✅ Input {size} -> Output {output.shape}")
        except Exception as e:
            print(f"❌ Input {size} -> Error: {e}")