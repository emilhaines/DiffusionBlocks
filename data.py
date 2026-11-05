import os
from collections.abc import Callable
from functools import partial
from typing import Any

import lightning as L
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from config import DatasetName, RunConfig

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def to_rgb(image: Image.Image) -> Image.Image:
    """Module-level (not a lambda) so transforms pickle under macOS spawn workers."""
    return image.convert("RGB")


def transforms(
    examples: dict[str, Any], transform: Callable[[Image.Image], Any]
) -> dict[str, Any]:
    examples["pixel_values"] = [transform(image) for image in examples["image"]]
    return {
        "pixel_values": examples["pixel_values"],
        "labels": examples["label"],
    }


class ImageDataModule(L.LightningDataModule):
    data_name: str
    image_size: int
    num_labels: int
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def __init__(
        self,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        add_rand_aug: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers: int = (
            num_workers if num_workers is not None else (os.cpu_count() or 1)
        )
        self.collate_fn = None
        train_transforms = [
            T.Lambda(to_rgb),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transformers = [
            T.Lambda(to_rgb),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        self.post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + self.post_transforms)
        self.val_transforms = T.Compose(val_transformers + self.post_transforms)
        self.datasets: dict[str, Any] = {}
        self.train_key: str = "train"
        self.val_key: str | None = "validation"
        self.test_key: str | None = "test"

    def prepare_data(self) -> None:
        load_dataset(self.data_name, num_proc=(os.cpu_count() or 2) // 2)

    def setup_dataset(self, data: DatasetDict) -> DatasetDict:
        return data

    def setup(self, stage: str | None = None) -> None:
        data = load_dataset(self.data_name)
        assert isinstance(data, DatasetDict)
        data = self.setup_dataset(data)
        train_data = data[self.train_key].with_transform(
            partial(transforms, transform=self.train_transforms)
        )
        self.datasets["train"] = train_data
        if self.val_key is not None:
            val_data = data[self.val_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["val"] = val_data
            self.val_dataloader = self._val_dataloader
        if self.test_key is not None:
            test_data = data[self.test_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["test"] = test_data
            self.test_dataloader = self._test_dataloader

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def _val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

    def _test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )


class CIFAR10DataModule(ImageDataModule):
    """For generation: pixels scaled to [-1, 1] (EDM's sigma_data = 0.5) with
    horizontal flips only. Random crops would teach the model padded borders."""

    data_name = "uoft-cs/cifar10"
    image_size = 32
    num_labels = 10

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # like CIFAR-100, no validation split; the test stage (sample grid, FID) is
        # not implemented yet, so main.py skips it
        self.val_key = None
        self.test_key = None
        self.train_transforms = T.Compose(
            [T.Lambda(to_rgb), T.RandomHorizontalFlip(), *self.post_transforms]
        )
        self.val_transforms = T.Compose([T.Lambda(to_rgb), *self.post_transforms])

    def setup_dataset(self, data: DatasetDict) -> DatasetDict:
        return data.rename_columns({"img": "image"})


class CIFAR100DataModule(ImageDataModule):
    data_name = "uoft-cs/cifar100"
    image_size = 32
    num_labels = 100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = None
        self.test_key = "test"
        # ref: https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/main/data_loader.py#L62
        train_transforms = [
            T.Lambda(to_rgb),
            T.Resize([self.image_size, self.image_size]),
            T.RandomCrop(self.image_size, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(to_rgb),
            T.Resize([self.image_size, self.image_size]),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)

    def setup_dataset(self, data: DatasetDict) -> DatasetDict:
        data = data.remove_columns(["coarse_label"])
        data = data.rename_columns({"img": "image", "fine_label": "label"})
        return data


class TinyImageNetDataModule(ImageDataModule):
    data_name = "zh-plus/tiny-imagenet"
    image_size = 64
    num_labels = 200
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = "valid"
        self.test_key = "valid"
        train_transforms = [
            T.Lambda(to_rgb),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(to_rgb),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)


DATA_MODULES: dict[DatasetName, type[ImageDataModule]] = {
    DatasetName.CIFAR10: CIFAR10DataModule,
    DatasetName.CIFAR100: CIFAR100DataModule,
    DatasetName.TINY_IMAGENET: TinyImageNetDataModule,
}


def load_data(cfg: RunConfig) -> ImageDataModule:
    return DATA_MODULES[cfg.data_name](
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        add_rand_aug=cfg.add_rand_aug,
    )
