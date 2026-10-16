"""The trainer overfits a tiny synthetic set, and resume reproduces an uninterrupted run."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from yolali.data import DetectionDataset, Letterbox, collate_samples
from yolali.data.formats import LabelRecord
from yolali.data.sample import Sample
from yolali.data.transforms.base import Compose
from yolali.eval import EvalConfig, Evaluator
from yolali.loss import DetectionLoss, LossConfig
from yolali.nn.config import ModelConfig
from yolali.nn.model import build_model
from yolali.train import CsvLogger, TrainConfig, Trainer, load_checkpoint


class SyntheticDataset(DetectionDataset):
    """Three fixed images with one coloured square each; no disk access."""

    def __init__(self, img_size: int = 64) -> None:
        records = [
            LabelRecord(
                f"img{i}", img_size, img_size, np.zeros((1, 4), np.float32), np.zeros(1, np.int64)
            )
            for i in range(3)
        ]
        super().__init__(records, img_size=img_size, transform=Compose([Letterbox(img_size)]))
        self.samples: list[Sample] = []
        for i in range(3):
            img = np.full((img_size, img_size, 3), 40, dtype=np.uint8)
            x1, y1 = 8 + 12 * i, 10 + 6 * i
            img[y1 : y1 + 24, x1 : x1 + 20] = (200, 60 + 60 * i, 30)
            boxes = np.array([[x1, y1, x1 + 20, y1 + 24]], dtype=np.float32)
            self.samples.append(Sample(img, boxes, np.array([i % 2]), path=f"img{i}"))

    def load_raw(self, index: int) -> Sample:
        s = self.samples[index]
        return Sample(s.image.copy(), s.boxes.copy(), s.classes.copy(), path=s.path)


def tiny_model_cfg() -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "name": "tiny",
            "nc": 2,
            "backbone": {"type": "gelan", "width": 0.125, "depth": 0.33},
            "neck": {"type": "gelan_pan", "width": 0.125, "depth": 0.33},
            "head": {"type": "detect", "reg_max": 0},
        }
    )


def make_trainer(out_dir: Path, epochs: int, *, with_eval: bool = False, seed: int = 0) -> Trainer:
    torch.manual_seed(seed)
    model = build_model(tiny_model_cfg())
    ds = SyntheticDataset()
    loader = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_samples)
    cfg = TrainConfig(
        epochs=epochs,
        img_size=64,
        lr=0.02,
        warmup_epochs=0,
        warmup_min_iters=0,
        amp=False,
        close_mosaic=0,
        out_dir=out_dir,
        log_every=1,
        seed=seed,
    )
    criterion = DetectionLoss(LossConfig(reg_max=0), num_classes=2)
    evaluator = (
        Evaluator(model, 2, EvalConfig(conf=0.01, iou=0.5, max_det=10)) if with_eval else None
    )
    return Trainer(
        model,
        criterion,
        loader,
        cfg,
        evaluator=evaluator,
        val_loader=loader if with_eval else None,
        loggers=[CsvLogger(out_dir)],
        device="cpu",
    )


def test_loss_decreases_and_files_are_written(tmp_path: Path):
    trainer = make_trainer(tmp_path, epochs=12, with_eval=True)
    first = trainer.train_epoch(0)["train/total"]
    last = trainer.fit()
    assert last["epoch/total"] < first
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "config.json").exists()
    assert "val/mAP50" in last
    ckpt = load_checkpoint(tmp_path / "last.pt")
    assert ckpt.epoch == 11 and ckpt.ema is not None and ckpt.optimizer is not None


def test_resume_matches_uninterrupted_run(tmp_path: Path):
    straight = make_trainer(tmp_path / "a", epochs=4)
    straight.fit()
    straight_state = {k: v.clone() for k, v in straight.model.state_dict().items()}
    assert straight.ema is not None
    straight_ema = {k: v.clone() for k, v in straight.ema.module.state_dict().items()}

    # Same 4-epoch configuration, "interrupted" after two epochs (what fit does, minus eval).
    first_half = make_trainer(tmp_path / "b", epochs=4)
    for epoch in range(2):
        first_half.train_epoch(epoch)
        first_half.save("last.pt", epoch)
    second_half = make_trainer(tmp_path / "b", epochs=4)
    second_half.resume(tmp_path / "b" / "last.pt")
    assert second_half.start_epoch == 2
    second_half.fit()

    for k, v in second_half.model.state_dict().items():
        assert torch.equal(v, straight_state[k]), k
    assert second_half.ema is not None
    for k, v in second_half.ema.module.state_dict().items():
        assert torch.equal(v, straight_ema[k]), k


def test_multi_scale_and_accumulate_run(tmp_path: Path):
    torch.manual_seed(0)
    model = build_model(tiny_model_cfg())
    loader = DataLoader(SyntheticDataset(), batch_size=1, collate_fn=collate_samples)
    cfg = TrainConfig(
        epochs=1,
        img_size=64,
        amp=False,
        close_mosaic=0,
        accumulate=3,
        multi_scale=(32, 96),
        out_dir=tmp_path,
        warmup_epochs=0,
        warmup_min_iters=0,
    )
    trainer = Trainer(model, DetectionLoss(LossConfig(reg_max=0), 2), loader, cfg, device="cpu")
    out = trainer.fit()
    assert "epoch/total" in out and np.isfinite(out["epoch/total"])


def test_close_mosaic_swaps_transform(tmp_path: Path):
    torch.manual_seed(0)
    model = build_model(tiny_model_cfg())
    ds = SyntheticDataset()
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_samples)
    cfg = TrainConfig(
        epochs=2,
        img_size=64,
        amp=False,
        close_mosaic=1,
        out_dir=tmp_path,
        warmup_epochs=0,
        warmup_min_iters=0,
    )
    replacement = Compose([Letterbox(64)])
    trainer = Trainer(
        model,
        DetectionLoss(LossConfig(reg_max=0), 2),
        loader,
        cfg,
        close_mosaic_transform=replacement,
        device="cpu",
    )
    trainer.fit()
    assert ds.transform is replacement


@pytest.mark.gpu
def test_amp_bf16_on_cuda(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    model = build_model(tiny_model_cfg())
    loader = DataLoader(SyntheticDataset(), batch_size=3, collate_fn=collate_samples)
    cfg = TrainConfig(
        epochs=1,
        img_size=64,
        amp=True,
        close_mosaic=0,
        out_dir=tmp_path,
        warmup_epochs=0,
        warmup_min_iters=0,
    )
    trainer = Trainer(model, DetectionLoss(LossConfig(reg_max=0), 2), loader, cfg, device="cuda")
    out = trainer.fit()
    assert np.isfinite(out["epoch/total"])
