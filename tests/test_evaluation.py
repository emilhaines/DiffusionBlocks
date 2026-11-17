from pathlib import Path

import pytest
import torch
from cleanfid.utils import ResizeDataset
from PIL import Image

from evaluation import (
    FID_MODES,
    CleanFID,
    FIDReference,
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
