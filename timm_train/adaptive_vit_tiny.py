# Created by MacBook Pro at 16.04.25
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


class SimplePatchMasker(nn.Module):
    def __init__(self, embed_dim, threshold=0.5):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.threshold = threshold

    def forward(self, x):  # x: [B, N, C]
        scores = self.score_fn(x).squeeze(-1)  # [B, N]
        mask = (scores > self.threshold).float()  # binary mask [B, N]
        return mask


class AdaptivePatchVisionTransformer(VisionTransformer):
    def __init__(self, *args, patch_masker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_masker = patch_masker  # Optional module to learn patch importance

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, dim]

        if self.patch_masker is not None:
            mask = self.patch_masker(x)  # mask shape: [B, num_patches]
            x = x * mask.unsqueeze(-1)  # apply mask

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return self.pre_logits(x[:, 0])


@register_model
def adaptive_vit_tiny_patch16_224(pretrained=False, **kwargs):
    masker = SimplePatchMasker(embed_dim=192, threshold=0.5)
    model = AdaptivePatchVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm,
        patch_masker=masker,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model