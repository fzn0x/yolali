"""Plain ViT backbone (DINOv2 / DINOv3 via ``timm``) with a Spatial Tuning Adapter.

The transformer produces one token grid at stride ``patch_size``; the
adapter turns it into the ``[P3, P4, P5]`` pyramid every neck expects.
Patch size 16 is required so the grid sits at stride 16; DINOv2 models are
natively patch 14 and ``timm`` resamples their patch embedding when asked
for 16 (``patch_size=16``), which is what RF-DETR does as well.

Licensing: DINOv2 weights are Apache-2.0; DINOv3 weights are under Meta's
DINOv3 license. Pick the model name accordingly.
"""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn

from yolali.nn.backbone.sta import SpatialTuningAdapter
from yolali.nn.interfaces import Backbone, Channels


def create_vit(model_name: str, *, patch_size: int, img_size: int, pretrained: bool) -> nn.Module:
    """Instantiate a ``timm`` ViT that accepts arbitrary image sizes."""
    try:
        import timm  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise ImportError("ViT backbones need `uv sync --extra vit`") from exc
    model: Any = timm.create_model(
        model_name,
        pretrained=pretrained,
        patch_size=patch_size,
        img_size=img_size,
        dynamic_img_size=True,
        num_classes=0,
    )
    return cast(nn.Module, model)


class ViTBackbone(Backbone):
    """``(B, 3, H, W)`` -> ``[P3, P4, P5]`` from a single-scale ViT.

    Args:
        model_name: ``timm`` name, e.g. ``vit_small_patch14_dinov2.lvd142m``
            or ``vit_small_patch16_dinov3.lvd1689m``.
        out_channels: pyramid channels ``(c3, c4, c5)``.
        img_size: nominal training resolution (positional embeddings are
            interpolated for other sizes).
        patch_size: token stride; must be 16 for the ``(8, 16, 32)`` contract.
        pretrained: download the pretrained weights.
        freeze: stop gradients into the transformer (adapter still trains).
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        out_channels: Channels = (128, 256, 384),
        *,
        img_size: int = 640,
        patch_size: int = 16,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if patch_size != 16:
            raise ValueError("patch_size must be 16 so the token grid sits at stride 16")
        self.vit = create_vit(
            model_name, patch_size=patch_size, img_size=img_size, pretrained=pretrained
        )
        vit_any = cast(Any, self.vit)
        embed_dim = int(vit_any.embed_dim)
        self.num_prefix_tokens = int(getattr(vit_any, "num_prefix_tokens", 1))
        self.patch_size = patch_size
        self.adapter = SpatialTuningAdapter(embed_dim, out_channels)
        self.out_channels = out_channels
        self.frozen = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def tokens_to_map(self, tokens: Tensor, hw: tuple[int, int]) -> Tensor:
        """``(B, prefix + N, C)`` -> ``(B, C, H/16, W/16)``."""
        h, w = hw[0] // self.patch_size, hw[1] // self.patch_size
        patches = tokens[:, self.num_prefix_tokens :]
        return patches.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the transformer once and expand to three scales."""
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("ViT backbone needs H and W to be multiples of 32")
        forward_features = cast(Any, self.vit).forward_features
        tokens: Tensor = forward_features(x)
        return self.adapter(self.tokens_to_map(tokens, (int(x.shape[-2]), int(x.shape[-1]))))

    def train(self, mode: bool = True) -> ViTBackbone:
        """Keep a frozen transformer in eval mode (no dropout / drop-path)."""
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self
