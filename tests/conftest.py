import random

import numpy as np
import pytest
import torch


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Tests run on CPU with every RNG the repo uses (random, numpy, torch) seeded."""
    seed_all(0)
