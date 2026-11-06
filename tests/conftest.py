import random
from typing import Any

import numpy as np
import pytest
import torch

from config import DatasetName, ModelType, RunConfig, Stage
from dit import DiTConfig
from model import DiTDBlockModel

TINY = DiTConfig(depth=3, hidden_size=32, num_heads=2, image_size=8, num_classes=10)


def make_model(**overrides: Any) -> DiTDBlockModel:
    """A tiny DiT generation model (3 layers, 8×8 images, 6 sampling steps)."""
    cfg = RunConfig(
        Stage.TRAIN,
        DatasetName.CIFAR10,
        ModelType.DIT,
        image_size=8,
        num_labels=10,
        **{"num_inference_steps": 6, **overrides},
    )
    model = DiTDBlockModel(cfg, TINY)
    model.configure_model()
    return model


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Tests run on CPU with every RNG the repo uses (random, numpy, torch) seeded."""
    seed_all(0)
