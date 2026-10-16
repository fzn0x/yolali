# Trainer

One class, one `fit`. Every optional feature is a single `if cfg.flag` at
the place where it acts, so `src/yolali/train/trainer.py` reads top to
bottom:

```
for epoch:
    maybe close mosaic
    for batch:
        maybe multi-scale
        forward (maybe autocast) -> loss
        backward (maybe scaled)
        every `accumulate` batches: maybe clip, step, maybe EMA
    maybe validate; save last / best
```

`Trainer(model, criterion, train_loader, cfg, evaluator=, val_loader=,
loggers=, close_mosaic_transform=, device=)`. `criterion(model_output,
targets)` returns anything with `.total` and `.as_dict()`, so
`DetectionLoss` and `DualLoss` plug in unchanged.

## TrainConfig

| Field | Default | Switch |
|---|---|---|
| `optimizer`, `lr`, `momentum`, `weight_decay` | `sgd`, 0.01, 0.937, 5e-4 | three param groups: decayed weights, undecayed norm weights, biases |
| `warmup_epochs`, `warmup_min_iters`, `warmup_bias_lr` | 3, 100, 0.1 | linear warmup; biases start high |
| `cosine`, `final_lr_factor` | `True`, 0.01 | cosine or linear decay to `lr · factor` |
| `amp` | `True` | bf16 autocast on GPUs that support it, else fp16 with a `GradScaler` |
| `ema`, `ema_decay`, `ema_tau` | `True`, 0.9999, 2000 | EMA with a ramped decay; evaluated and exported weights |
| `close_mosaic` | 10 | swap the dataset's transform for the last N epochs |
| `accumulate` | 1 | optimizer step every N batches |
| `multi_scale` | `None` | `(min, max)` size sampled per batch, multiples of 32 |
| `compile` | `False` | `torch.compile` the forward |
| `grad_clip` | 10.0 | max gradient norm |
| `val_every` | 1 | evaluate every N epochs; `best.pt` tracks `0.1·mAP50 + 0.9·mAP50-95` |

## Checkpoints and resume

A `Checkpoint` holds the model, EMA, optimizer, scaler and RNG state
(Python, torch CPU, CUDA), the epoch, the best fitness and both configs.
`Trainer.resume(path)` restores all of it; the test trains four epochs
straight, then two plus two with a save and resume in between, and asserts
`torch.equal` on every model and EMA tensor.

## Distributed

Launch with `torchrun`; the trainer reads `RANK` and `WORLD_SIZE`, wraps
the model in `DistributedDataParallel`, and only rank 0 logs, validates and
saves. `scripts/train.py` builds the `DistributedSampler`.

## Loggers

`Logger` is a protocol with `log_scalars(step, values)`, `log_config(cfg)`
and `close()`. `CsvLogger` is always available; `TensorBoardLogger` and
`WandbLogger` need their extras and raise a clear `ImportError` otherwise.

Source: `src/yolali/train/`. API: [yolali.train](../api/train.md).
