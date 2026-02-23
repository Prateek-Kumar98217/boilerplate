# 03 — Training

Production-grade PyTorch training boilerplate with full VRAM optimization.

## Structure

```
03-training/
├── config.py                       # Pydantic Settings (hardware, loop, optimizer, scheduler…)
├── .env.example
├── models/
│   ├── base.py                     # Abstract BaseModel with param counting, weight init
│   ├── example_cnn.py              # ResNet-style ConvNet with residual blocks
│   └── example_transformer.py     # Pre-norm Transformer encoder for sequence classification
├── data/
│   ├── dataset.py                  # ArrayDataset, CSVDataset, ImageFolderDataset
│   └── dataloader.py               # DataLoader factory + WeightedRandomSampler
├── training/
│   ├── trainer.py                  # Trainer: AMP, grad clipping, grad accumulation, resume
│   ├── callbacks.py                # EarlyStopping, ModelCheckpoint, LRLogger, WandB, TB
│   ├── metrics.py                  # MetricTracker: accuracy, F1, MAE, RMSE, R²
│   └── schedulers.py               # cosine_warmup, linear_warmup, step, plateau, onecycle
├── utils/
│   ├── cuda_utils.py               # Device helper, seed, GPU memory stats
│   └── checkpoint.py               # CheckpointManager (save/load/resume)
└── examples/
    └── train_classification.py     # Full end-to-end training run
```

## VRAM Optimization Features

| Technique                   | Config Key                      | Default                   |
| --------------------------- | ------------------------------- | ------------------------- |
| Mixed precision (bf16/fp16) | `MIXED_PRECISION`               | `true`                    |
| Gradient accumulation       | `GRADIENT_ACCUMULATION_STEPS`   | `1`                       |
| Gradient clipping           | `GRADIENT_CLIPPING`             | `1.0`                     |
| Gradient checkpointing      | `GRADIENT_CHECKPOINTING`        | `false`                   |
| GradScaler (fp16)           | auto                            | enabled with fp16         |
| torch.compile               | `compile_model=True` in Trainer | off                       |
| Pinned memory               | `PIN_MEMORY`                    | `true`                    |
| Persistent workers          | auto                            | true when num_workers > 0 |

## Quick Start

```bash
cp .env.example .env
python examples/train_classification.py
```

## Optimizer Choices

```
OPTIMIZER=adamw   # adam | adamw | sgd | rmsprop
```

## Scheduler Choices

```
SCHEDULER=cosine_warmup  # cosine | cosine_warmup | linear_warmup | step | plateau | onecycle | none
```

## Callbacks

```python
from training.callbacks import EarlyStopping, ModelCheckpoint, WandBCallback

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10),
    ModelCheckpoint(monitor="accuracy", mode="max"),
    WandBCallback(project="my-project"),
]
```
