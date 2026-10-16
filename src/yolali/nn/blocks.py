"""Building blocks of the GELAN family.

Reimplemented from the YOLOv9 paper (Wang, Yeh, Liao 2024, arXiv:2402.13616,
https://github.com/WongKinYiu/yolov9). Each class names its counterpart in the
reference repository so a reader can cross-check; parameter layouts match
so official checkpoints convert one-to-one. No code was copied.

Channel arguments are explicit. Width/depth scaling happens in
``yolali.nn.config``, never inside a block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - torch convention
from torch import Tensor, nn


def make_divisible(value: float, divisor: int) -> int:
    """Round ``value`` up to the nearest multiple of ``divisor`` (at least ``divisor``)."""
    return max(divisor, int(-(-value // divisor) * divisor))


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold a BatchNorm that follows ``conv`` into a new ``Conv2d`` with bias.

    ``bn(conv(x)) = gamma * (conv(x) - mean) / sqrt(var + eps) + beta`` is
    affine in the conv output, so it folds into the kernel and a bias term.
    """
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        (conv.kernel_size[0], conv.kernel_size[1]),
        (conv.stride[0], conv.stride[1]),
        conv.padding if isinstance(conv.padding, str) else (conv.padding[0], conv.padding[1]),
        (conv.dilation[0], conv.dilation[1]),
        conv.groups,
        bias=True,
    ).to(conv.weight.device)
    running_var = bn.running_var if bn.running_var is not None else torch.ones_like(bn.weight)
    running_mean = bn.running_mean if bn.running_mean is not None else torch.zeros_like(bn.weight)
    scale = bn.weight / torch.sqrt(running_var + bn.eps)  # (C_out,)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)
        assert fused.bias is not None
        fused.bias.copy_(bn.bias + (conv_bias - running_mean) * scale)
    return fused


class Conv(nn.Module):
    """``Conv2d -> BatchNorm2d -> SiLU`` (reference: ``Conv``).

    Args:
        c_in: input channels.
        c_out: output channels.
        k: kernel size.
        s: stride.
        p: padding; ``None`` means ``k // 2`` ("same" for odd kernels).
        g: groups.
        act: apply SiLU after BN.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        *,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn: nn.Module = nn.BatchNorm2d(c_out)
        self.act: nn.Module = nn.SiLU() if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H/s, W/s)``."""
        return self.act(self.bn(self.conv(x)))

    def fuse(self) -> None:
        """Fold BN into the conv for inference. Idempotent."""
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = fuse_conv_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class RepConv(nn.Module):
    """Parallel ``3x3`` and ``1x1`` conv-BN branches, summed, then SiLU.

    Reference: ``RepConvN`` (no identity branch). ``fuse()`` collapses both
    branches into one ``3x3`` conv following RepVGG (Ding et al. 2021).
    """

    def __init__(self, c_in: int, c_out: int, s: int = 1, g: int = 1) -> None:
        super().__init__()
        self.conv1 = Conv(c_in, c_out, 3, s, p=1, g=g, act=False)
        self.conv2 = Conv(c_in, c_out, 1, s, p=0, g=g, act=False)
        self.act = nn.SiLU()
        self.fused: nn.Conv2d | None = None

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H/s, W/s)``."""
        if self.fused is not None:
            return self.act(self.fused(x))
        return self.act(self.conv1(x) + self.conv2(x))

    def fuse(self) -> None:
        """Merge both branches into one ``3x3`` conv with bias. Idempotent."""
        if self.fused is not None:
            return
        bn1, bn2 = self.conv1.bn, self.conv2.bn
        assert isinstance(bn1, nn.BatchNorm2d) and isinstance(bn2, nn.BatchNorm2d)
        k3 = fuse_conv_bn(self.conv1.conv, bn1)
        k1 = fuse_conv_bn(self.conv2.conv, bn2)
        assert k3.bias is not None and k1.bias is not None
        with torch.no_grad():
            k3.weight += F.pad(k1.weight, (1, 1, 1, 1))
            k3.bias += k1.bias
        self.fused = k3
        del self.conv1, self.conv2


class RepBottleneck(nn.Module):
    """``RepConv(3x3) -> Conv(3x3)`` with optional residual (reference: ``RepNBottleneck``)."""

    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = RepConv(c_in, c_hidden)
        self.cv2 = Conv(c_hidden, c_out, 3, 1)
        self.add = shortcut and c_in == c_out

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H, W)``."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RepCSP(nn.Module):
    """CSP bottleneck with ``n`` :class:`RepBottleneck` blocks (reference: ``RepNCSP``).

    Two ``1x1`` convs split the input; one path runs the bottlenecks, the
    other is an identity shortcut; a final ``1x1`` merges them.
    """

    def __init__(
        self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True, e: float = 0.5
    ) -> None:
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_in, c_hidden, 1, 1)
        self.cv3 = Conv(2 * c_hidden, c_out, 1)
        self.m = nn.Sequential(
            *(RepBottleneck(c_hidden, c_hidden, shortcut, e=1.0) for _ in range(n))
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H, W)``."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepCSPELAN(nn.Module):
    """GELAN unit (reference: ``RepNCSPELAN4``).

    ``1x1`` expands to ``c_mid`` channels, split in two halves. The second half
    runs through two stacked ``(RepCSP -> Conv3x3)`` stages, each producing
    ``c_stage`` channels; the four tensors (two halves, two stage outputs)
    are concatenated and merged by a ``1x1``.

    Args:
        c_in: input channels.
        c_out: output channels.
        c_mid: channels after the first ``1x1`` (split into halves).
        c_stage: channels of each stage output.
        n: bottlenecks per ``RepCSP``.
    """

    def __init__(self, c_in: int, c_out: int, c_mid: int, c_stage: int, n: int = 1) -> None:
        super().__init__()
        if c_mid % 2:
            raise ValueError("c_mid must be even so the split halves are equal")
        self.cv1 = Conv(c_in, c_mid, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c_mid // 2, c_stage, n), Conv(c_stage, c_stage, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c_stage, c_stage, n), Conv(c_stage, c_stage, 3, 1))
        self.cv4 = Conv(c_mid + 2 * c_stage, c_out, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H, W)``."""
        half_a, half_b = self.cv1(x).chunk(2, 1)
        stage1 = self.cv2(half_b)
        stage2 = self.cv3(stage1)
        return self.cv4(torch.cat((half_a, half_b, stage1, stage2), 1))


class ADown(nn.Module):
    """Stride-2 downsample mixing average and max pooling (reference: ``ADown``).

    A ``2x2`` stride-1 average pool smooths the input, then the channels are
    split: half go through a stride-2 ``3x3`` conv, half through a ``3x3``
    stride-2 max pool followed by a ``1x1`` conv.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.cv1 = Conv(c_in // 2, c_out // 2, 3, 2, p=1)
        self.cv2 = Conv(c_in // 2, c_out // 2, 1, 1, p=0)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H/2, W/2)`` for even ``H``, ``W``."""
        x = F.avg_pool2d(x, 2, 1, 0, ceil_mode=False, count_include_pad=True)
        a, b = x.chunk(2, 1)
        a = self.cv1(a)
        b = self.cv2(F.max_pool2d(b, 3, 2, 1))
        return torch.cat((a, b), 1)


class SPPELAN(nn.Module):
    """Spatial pyramid pooling in ELAN form (reference: ``SPPELAN``).

    ``1x1`` to ``c_mid``, three cascaded ``5x5`` max pools, concat all four,
    ``1x1`` to ``c_out``.
    """

    def __init__(self, c_in: int, c_out: int, c_mid: int) -> None:
        super().__init__()
        self.cv1 = Conv(c_in, c_mid, 1, 1)
        self.cv2 = nn.MaxPool2d(5, 1, 2)
        self.cv3 = nn.MaxPool2d(5, 1, 2)
        self.cv4 = nn.MaxPool2d(5, 1, 2)
        self.cv5 = Conv(4 * c_mid, c_out, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, c_in, H, W) -> (B, c_out, H, W)``."""
        y0 = self.cv1(x)
        y1 = self.cv2(y0)
        y2 = self.cv3(y1)
        y3 = self.cv4(y2)
        return self.cv5(torch.cat((y0, y1, y2, y3), 1))


def fuse_all(module: nn.Module) -> None:
    """Fuse every fusable block in ``module`` for inference.

    A block that defines ``fuse()`` is responsible for its own children (a
    :class:`RepConv` folds both branches at once), so recursion stops there.
    """
    fuse = getattr(module, "fuse", None)
    if callable(fuse):
        fuse()
        return
    for child in module.children():
        fuse_all(child)
