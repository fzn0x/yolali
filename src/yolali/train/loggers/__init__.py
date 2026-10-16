"""Pluggable training loggers."""

from yolali.train.loggers.base import Logger, MultiLogger
from yolali.train.loggers.csv import CsvLogger
from yolali.train.loggers.tensorboard import TensorBoardLogger
from yolali.train.loggers.wandb import WandbLogger

__all__ = ["CsvLogger", "Logger", "MultiLogger", "TensorBoardLogger", "WandbLogger"]
