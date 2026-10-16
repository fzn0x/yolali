"""Train a detector.

    uv run scripts/train.py --model configs/models/gelan-n.yaml --data configs/data/coco128.yaml \\
        --epochs 50 --batch-size 16 --out runs/coco128-n

Multi-GPU (one process per GPU, DistributedSampler is set up here):

    uv run torchrun --nproc_per_node 2 scripts/train.py ... --workers 4

Resume: add ``--resume runs/coco128-n/last.pt``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, DistributedSampler

from yolali.data import (
    DetectionDataset,
    LabelCache,
    RectBatchSampler,
    collate_samples,
    default_train_transforms,
    read_yolo_txt,
)
from yolali.data.dataset import AugmentConfig
from yolali.data.formats import LabelRecord, list_images
from yolali.eval import EvalConfig, Evaluator
from yolali.loss import DetectionLoss, DualLoss, LossConfig
from yolali.nn.config import ModelConfig
from yolali.nn.model import build_model, load_model
from yolali.train import CsvLogger, TrainConfig, Trainer
from yolali.train.loggers import Logger, TensorBoardLogger, WandbLogger

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, required=True, help="model yaml")
    p.add_argument("--data", type=Path, required=True, help="data yaml with train/val/nc/names")
    p.add_argument("--weights", type=Path, default=None, help="initial yolali state_dict")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--multi-scale", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--nms-free", action="store_true", help="evaluate the dual head without NMS")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("runs/exp"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--wandb", type=str, default=None, metavar="PROJECT")
    return p.parse_args()


def load_records(images: Path) -> list[LabelRecord]:
    cache = LabelCache(images.parent / f"{images.name}.cache")
    return cache.load_or_build(list_images(images), lambda: read_yolo_txt(images))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    with args.data.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    nc = int(data["nc"])
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))

    model_cfg = ModelConfig.from_yaml(args.model)
    if model_cfg.num_classes != nc:
        model_cfg = ModelConfig(
            model_cfg.name, nc, model_cfg.backbone, model_cfg.neck, model_cfg.head
        )
    model = load_model(args.model, args.weights) if args.weights else build_model(model_cfg)
    if model.num_classes != nc:
        model = build_model(model_cfg)

    train_records = load_records(ROOT / data["train"])
    val_records = load_records(ROOT / data["val"])
    aug = AugmentConfig(mosaic=args.mosaic, mixup=args.mixup, scale=args.scale)
    train_ds = DetectionDataset(train_records, img_size=args.img_size, seed=args.seed)
    train_ds.transform = default_train_transforms(train_ds, args.img_size, aug)
    no_mosaic = default_train_transforms(
        train_ds, args.img_size, AugmentConfig(mosaic=0.0, mixup=0.0, scale=args.scale)
    )
    sampler: DistributedSampler[int] | None = (
        DistributedSampler(train_ds, shuffle=True, seed=args.seed) if distributed else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_samples,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_sampler = RectBatchSampler(
        [r.aspect_ratio for r in val_records], args.batch_size * 2, args.img_size
    )
    val_ds = DetectionDataset(
        val_records, img_size=args.img_size, rect_shapes=val_sampler.batch_shapes
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=args.workers, collate_fn=collate_samples
    )

    loss_cfg = LossConfig(reg_max=model.reg_max)
    criterion = (
        DualLoss(loss_cfg, nc) if model.cfg.head.type == "dual" else DetectionLoss(loss_cfg, nc)
    )
    evaluator = Evaluator(model, nc, EvalConfig(nms=not args.nms_free))

    cfg = TrainConfig(
        epochs=args.epochs,
        img_size=args.img_size,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        amp=not args.no_amp,
        ema=not args.no_ema,
        close_mosaic=args.close_mosaic,
        accumulate=args.accumulate,
        multi_scale=tuple(args.multi_scale) if args.multi_scale else None,
        compile=args.compile,
        val_every=args.val_every,
        seed=args.seed,
        out_dir=args.out,
    )
    loggers: list[Logger] = []
    if rank == 0:
        loggers.append(CsvLogger(args.out))
        if args.tensorboard:
            loggers.append(TensorBoardLogger(args.out / "tb"))
        if args.wandb:
            loggers.append(WandbLogger(args.wandb, name=args.out.name))

    trainer = Trainer(
        model,
        criterion,
        train_loader,
        cfg,
        evaluator=evaluator,
        val_loader=val_loader,
        loggers=loggers,
        close_mosaic_transform=no_mosaic,
        device=args.device,
    )
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
