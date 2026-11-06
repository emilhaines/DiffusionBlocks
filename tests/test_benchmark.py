import pytest
from conftest import make_model

from scripts.benchmark import measure_flops, measure_training_memory


def test_layer_flops_scale_with_layers_per_block_but_the_rest_does_not() -> None:
    one, three = (measure_flops(make_model(num_blocks=b)) for b in (1, 3))
    assert one.layers / three.layers == pytest.approx(3)
    assert one.total / three.total < 3  # the stem and head stay


def test_one_block_per_step_saves_activations_and_gradients_not_weights() -> None:
    one, three = (
        measure_training_memory(make_model(num_blocks=b), batch_size=4, steps=6)
        for b in (1, 3)
    )
    assert one.weights == three.weights and one.adamw == three.adamw
    assert three.activations < one.activations
    assert three.gradients < one.gradients
    assert one.allocator_peak is None  # the CPU has no allocator reading
