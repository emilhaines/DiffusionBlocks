import pickle
from dataclasses import fields

from PIL import Image

from config import DatasetName, ModelType, RunConfig, Stage, build_parser, parse_args
from data import DATA_MODULES, to_rgb
from model import MODELS


def test_cli_strings_parse_to_enums_and_defaults_come_from_runconfig() -> None:
    cfg = parse_args(
        ["train", "cifar100", "--model_type", "dblock", "--num_blocks", "1", "--debug"]
    )
    assert (cfg.stage, cfg.data_name, cfg.model_type) == (
        Stage.TRAIN,
        DatasetName.CIFAR100,
        ModelType.DBLOCK,
    )
    assert cfg.num_blocks == 1 and cfg.debug and cfg.lr == RunConfig.lr
    assert cfg.to_dict()["model_type"] == "dblock"


def test_every_cli_flag_is_a_runconfig_field_and_registries_are_complete() -> None:
    dests = {a.dest for a in build_parser()._actions} - {"help"}
    assert dests == {f.name for f in fields(RunConfig)} - {"image_size", "num_labels"}
    assert set(MODELS) == set(ModelType) and set(DATA_MODULES) == set(DatasetName)


def test_to_rgb_pickles_for_spawned_workers() -> None:
    assert pickle.loads(pickle.dumps(to_rgb))(Image.new("L", (2, 2))).mode == "RGB"
