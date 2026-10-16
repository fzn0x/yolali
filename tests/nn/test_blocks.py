"""Shapes, hand-derived parameter counts, and BN/RepConv fusion equivalence."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from yolali.nn import (
    SPPELAN,
    ADown,
    Conv,
    RepBottleneck,
    RepConv,
    RepCSP,
    RepCSPELAN,
    fuse_all,
    make_divisible,
)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def conv_params(c_in: int, c_out: int, k: int, g: int = 1) -> int:
    """Conv2d without bias + BatchNorm2d (weight, bias)."""
    return c_out * (c_in // g) * k * k + 2 * c_out


def randomise_bn(m: nn.Module) -> None:
    """Give every BatchNorm non-trivial statistics so fusion is actually tested."""
    for bn in m.modules():
        if isinstance(bn, nn.BatchNorm2d):
            with torch.no_grad():
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.uniform_(-0.5, 0.5)
                assert bn.running_mean is not None and bn.running_var is not None
                bn.running_mean.uniform_(-1, 1)
                bn.running_var.uniform_(0.5, 2.0)


def test_make_divisible():
    assert make_divisible(64 * 0.25, 8) == 16
    assert make_divisible(3, 8) == 8
    assert make_divisible(65, 8) == 72


def test_conv_shape_and_params():
    m = Conv(3, 16, 3, 2)
    assert m(torch.zeros(1, 3, 32, 32)).shape == (1, 16, 16, 16)
    assert n_params(m) == conv_params(3, 16, 3)


def test_conv_groups_params():
    m = Conv(64, 64, 3, g=4)
    assert n_params(m) == conv_params(64, 64, 3, g=4)


def test_repconv_params_are_3x3_plus_1x1():
    m = RepConv(8, 16)
    assert n_params(m) == conv_params(8, 16, 3) + conv_params(8, 16, 1)
    assert m(torch.zeros(1, 8, 10, 10)).shape == (1, 16, 10, 10)


def test_repcspelan_shape_and_params():
    # RepCSPELAN(c_in=128, c_out=256, c_mid=128, c_stage=64, n=1), the first block of GELAN-C.
    m = RepCSPELAN(128, 256, 128, 64, 1)
    assert m(torch.zeros(1, 128, 16, 16)).shape == (1, 256, 16, 16)

    def repcsp(c_in: int, c_out: int) -> int:
        c_hidden = c_out // 2  # e = 0.5
        bottleneck = (
            conv_params(c_hidden, c_hidden, 3) + conv_params(c_hidden, c_hidden, 1)  # RepConv
        ) + conv_params(c_hidden, c_hidden, 3)  # Conv 3x3
        return (
            conv_params(c_in, c_hidden, 1)
            + conv_params(c_in, c_hidden, 1)
            + conv_params(2 * c_hidden, c_out, 1)
            + bottleneck
        )

    expected = (
        conv_params(128, 128, 1)  # cv1
        + repcsp(64, 64)
        + conv_params(64, 64, 3)  # cv2
        + repcsp(64, 64)
        + conv_params(64, 64, 3)  # cv3
        + conv_params(128 + 2 * 64, 256, 1)  # cv4
    )
    assert n_params(m) == expected


def test_adown_halves_resolution():
    m = ADown(256, 256)
    assert m(torch.zeros(1, 256, 40, 40)).shape == (1, 256, 20, 20)
    assert n_params(m) == conv_params(128, 128, 3) + conv_params(128, 128, 1)


def test_sppelan_shape():
    m = SPPELAN(512, 512, 256)
    assert m(torch.zeros(1, 512, 20, 20)).shape == (1, 512, 20, 20)
    assert n_params(m) == conv_params(512, 256, 1) + conv_params(1024, 512, 1)


def test_repbottleneck_residual_only_when_channels_match():
    assert RepBottleneck(32, 32).add
    assert not RepBottleneck(32, 64).add


@pytest.mark.parametrize(
    ("factory", "c_in"),
    [
        (lambda: Conv(8, 16, 3, 2), 8),
        (lambda: RepConv(8, 16), 8),
        (lambda: RepCSP(16, 32, n=2), 16),
        (lambda: RepCSPELAN(32, 64, 32, 16, 1), 32),
        (lambda: ADown(32, 32), 32),
        (lambda: SPPELAN(32, 32, 16), 32),
    ],
)
def test_fuse_preserves_outputs(factory: Callable[[], nn.Module], c_in: int):
    torch.manual_seed(0)
    m = factory().eval()
    randomise_bn(m)
    x = torch.randn(2, c_in, 16, 16)
    with torch.no_grad():
        before = m(x)
        fuse_all(m)
        after = m(x)
    assert not any(isinstance(sub, nn.BatchNorm2d) for sub in m.modules())
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()


def test_fuse_is_idempotent():
    m = RepConv(4, 4).eval()
    randomise_bn(m)
    x = torch.randn(1, 4, 8, 8)
    fuse_all(m)
    y1 = m(x)
    fuse_all(m)
    assert torch.equal(m(x), y1)
