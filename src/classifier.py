# src/classifier.py
import torch
import torch.nn as nn

class SimpleCNNBackbone(nn.Module):
  
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /16
        )

    def forward(self, x):
        return self.features(x)


class PixelClassifier(nn.Module):
    """Classifier 1 (pixel): 3 input channels"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SimpleCNNBackbone(in_channels=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SpectrumClassifier(nn.Module):
    """Classifier 2 (spectrum): 1 input channel"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SimpleCNNBackbone(in_channels=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DualBranchClassifier(nn.Module):
    """Classifier 3 (dual-branch): pixel + spectrum → concat → linear head"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.pixel_backbone = SimpleCNNBackbone(in_channels=3)
        self.spectrum_backbone = SimpleCNNBackbone(in_channels=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 拼接后维度: 256 + 256 = 512
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pixel_input, spectrum_input):
        p = self.pixel_backbone(pixel_input)
        s = self.spectrum_backbone(spectrum_input)
        p = self.global_pool(p)
        s = self.global_pool(s)
        p = torch.flatten(p, 1)
        s = torch.flatten(s, 1)
        fused = torch.cat([p, s], dim=1)  # [B, 512]
        out = self.classifier(fused)
        return out


# # ===== test=====
# if __name__ == "__main__":
#     # 测试像素分类器
#     pixel_model = PixelClassifier()
#     x_pixel = torch.randn(2, 3, 256, 256)
#     print("Pixel output shape:", pixel_model(x_pixel).shape)  # 应为 [2, 2]

#     # 测试频谱分类器
#     spec_model = SpectrumClassifier()
#     x_spec = torch.randn(2, 1, 256, 256)
#     print("Spectrum output shape:", spec_model(x_spec).shape)  # 应为 [2, 2]

#     # 测试双分支分类器
#     dual_model = DualBranchClassifier()
#     out_dual = dual_model(x_pixel, x_spec)
#     print("Dual output shape:", out_dual.shape)  # 应为 [2, 2]