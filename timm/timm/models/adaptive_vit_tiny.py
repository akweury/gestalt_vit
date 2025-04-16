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
        # Remove extra args that timm might pass in
        EXTRA_KWARGS = [
            'pretrained_cfg', 'features_only', 'global_pool',
            'pretrained_cfg_overlay', 'cache_dir'
        ]
        for k in EXTRA_KWARGS:
            kwargs.pop(k, None)


        super().__init__(*args, **kwargs)
        self.patch_masker = patch_masker  # Optional module to learn patch importance

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.patch_masker is not None:
            mask = self.patch_masker(x)
            x = x * mask.unsqueeze(-1)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # [CLS] token

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)  # logits


@register_model
def adaptive_vit_tiny_patch16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm,
        patch_masker=SimplePatchMasker(embed_dim=192, threshold=0.5),
    )
    model = AdaptivePatchVisionTransformer(**model_args, **kwargs)
    model.default_cfg = _cfg()
    return model
