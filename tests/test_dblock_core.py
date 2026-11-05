import numpy as np
import torch
from scipy.stats import norm

from dblock_modules import BlockPartition, EDMCoefficients, edm_loss_weight

P = BlockPartition(num_blocks=3, gamma=0.05)


def test_boundaries_have_equal_probability_mass() -> None:  # §3.3
    n = P.noise
    mass = np.diff(norm.cdf((np.log(P.boundaries) - n.p_mean) / n.p_std))
    np.testing.assert_allclose(mass, mass[0])


def test_interval_expands_unexpanded_block_and_stays_in_range() -> None:  # App. C
    b = P.boundaries
    for block in range(3):
        lo, hi = P.interval(block)
        assert b[0] <= lo <= b[2 - block] < b[3 - block] <= hi <= b[-1]


def test_samples_inside_interval_and_block_0_is_highest_noise() -> None:
    for block in range(3):
        lo, hi = P.interval(block)
        s = P.sample_sigmas(block, 1000)
        assert lo <= s.min() and s.max() <= hi
        assert P.majority_block(s) == block


def test_inference_steps_spread_evenly_over_blocks() -> None:
    """App. H: 50 steps split 17/16/17 over 3 blocks."""
    counts = P.block_of(P.inference_sigmas(50)).bincount()
    assert counts.tolist() == [17, 16, 17]


def test_edm_weight_is_inverse_c_out_squared() -> None:
    sigma = torch.logspace(-2.5, 1.9, 20, dtype=torch.float64)
    c = EDMCoefficients.from_sigma(sigma)
    torch.testing.assert_close(
        edm_loss_weight(sigma) * c.c_out**2, torch.ones_like(sigma)
    )


def test_layer_assignment_is_contiguous() -> None:
    assert P.layer_assignment(12) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
