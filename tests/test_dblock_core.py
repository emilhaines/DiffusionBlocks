import numpy as np
from scipy.stats import norm

from dblock_modules import BlockPartition

P = BlockPartition(num_blocks=3, gamma=0.05)


def test_block_sigmas_have_equal_probability_mass() -> None:  # §3.3
    n = P.noise
    mass = np.diff(norm.cdf((np.log(P.block_sigmas) - n.p_mean) / n.p_std))
    np.testing.assert_allclose(mass, mass[0])


def test_sigma_range_expands_block_and_stays_in_range() -> None:  # App. C
    b = P.block_sigmas
    for block in range(3):
        lo, hi = P.get_block_sigma_range(block)
        assert b[0] <= lo <= b[2 - block] < b[3 - block] <= hi <= b[-1]


def test_inference_steps_spread_evenly_over_blocks() -> None:
    """App. H: 50 steps split 17/16/17 over 3 blocks."""
    counts = P.get_block_idx(P.get_discrete_sigmas(50)).bincount()
    assert counts.tolist() == [17, 16, 17]
