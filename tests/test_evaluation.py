import json
from pathlib import Path
from typing import cast

import lightning as L
import numpy as np
import pytest
import torch
import torchmetrics
from cleanfid.utils import ResizeDataset
from conftest import make_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evaluation import (
    FID_MODES,
    CleanFID,
    FIDReference,
    SampleGrid,
    SaveResults,
    to_uint8,
)


@pytest.mark.parametrize("reference", list(FIDReference))
def test_fid_input_matches_clean_fid_on_saved_pngs(
    tmp_path: Path, reference: FIDReference
) -> None:
    images = torch.rand(2, 3, 8, 8) * 2.2 - 1.1  # includes out-of-range values
    paths = [tmp_path / f"{i}.png" for i in range(2)]
    for path, image in zip(paths, to_uint8(images).numpy(), strict=True):
        Image.fromarray(image).save(path)
    pngs = ResizeDataset(paths, FID_MODES[reference].name)
    metric = CleanFID("cifar10", 32, reference)
    metric.extractor = lambda x: x.flatten(1)  # capture the Inception input
    metric.update(images)
    expected = torch.stack([pngs[0], pngs[1]]).flatten(1)
    torch.testing.assert_close(metric.features[0], expected)


class Count(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("n", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, images: torch.Tensor) -> None:
        self.n += images.shape[0]

    def compute(self) -> torch.Tensor:
        return self.n


def test_test_stage_updates_metrics_and_saves_grid(tmp_path: Path) -> None:
    model = make_model(cfg_scale=2.0)
    model.test_metrics = torchmetrics.MetricCollection({"n": Count()}, prefix="test/")
    batches = [{"pixel_values": torch.zeros(4, 3, 8, 8), "labels": torch.arange(4)}]
    grid = SampleGrid(tmp_path / "grid.png", num_classes=3, per_class=2)
    trainer = L.Trainer(accelerator="cpu", logger=False, callbacks=[grid])
    (res,) = trainer.test(
        model, DataLoader(cast(Dataset, batches), batch_size=None), verbose=False
    )
    assert res["test/n"] == 4
    assert Image.open(tmp_path / "grid.png").size == (2 * 9 + 1, 3 * 9 + 1)


def make_fid_model(
    monkeypatch: pytest.MonkeyPatch, stats: object
) -> tuple[L.LightningModule, CleanFID]:
    monkeypatch.setattr("evaluation.get_reference_statistics", stats)
    metric = CleanFID("cifar10", 8, FIDReference.TEST)
    metric.extractor = lambda x: x.mean((2, 3))  # 3 "features" per image
    model = make_model(cfg_scale=2.0)
    model.test_metrics = torchmetrics.MetricCollection(
        {"fid_test": metric}, prefix="test/"
    )
    return model, metric


def test_results_json_has_fids_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stats = (np.zeros(3), np.eye(3))
    model, _ = make_fid_model(monkeypatch, lambda *a, **k: stats)
    batches = [{"pixel_values": torch.zeros(4, 3, 8, 8), "labels": torch.arange(4)}]
    path = tmp_path / "results.json"
    trainer = L.Trainer(
        accelerator="cpu", logger=False, callbacks=[SaveResults(path, {"seed": 7})]
    )
    trainer.test(
        model, DataLoader(cast(Dataset, batches), batch_size=None), verbose=False
    )
    results = json.loads(path.read_text())
    assert isinstance(results["fid"]["fid_test"], float)
    assert results["num_generated"] == 4 and results["config"] == {"seed": 7}


def test_missing_reference_fails_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise ConnectionError

    model, metric = make_fid_model(monkeypatch, offline)
    batches = [{"pixel_values": torch.zeros(4, 3, 8, 8), "labels": torch.arange(4)}]
    trainer = L.Trainer(accelerator="cpu", logger=False)
    with pytest.raises(ConnectionError):
        trainer.test(model, DataLoader(cast(Dataset, batches), batch_size=None))
    assert not metric.features  # raised before any sampling
