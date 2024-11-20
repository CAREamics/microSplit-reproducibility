from pathlib import Path
from typing import Union

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

Callback = Union[EarlyStopping, LearningRateMonitor, ModelCheckpoint]

def get_callbacks(log_dir: Union[str, Path]) -> list[Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=200,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]