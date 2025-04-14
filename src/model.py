# Created by MacBook Pro at 13.04.25


import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output: (B, embed_dim, 1, 1)
        )

    def forward(self, patches):  # (B, N, 3, 16, 16)
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        features = self.embed(patches)  # (B*N, D, 1, 1)
        features = features.view(B, N, -1)  # (B, N, D)
        return features


class TinyTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):  # x: (B, N, D)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = self.encoder(x)
        return x[:, 0]  # return CLS token


class AdaptiveViTClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_classes=5):
        super().__init__()
        self.patcher = PatchEmbedder(in_channels=3, embed_dim=embed_dim)
        self.transformer = TinyTransformer(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, patches):  # patches: (B, N, 3, 16, 16)
        tokens = self.patcher(patches)     # (B, N, D)
        cls_repr = self.transformer(tokens)  # (B, D)
        out = self.classifier(cls_repr)    # (B, num_classes)
        return out

