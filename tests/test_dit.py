import torch
from conftest import TINY

from dit import DiT


def test_null_condition_ignores_the_label() -> None:
    model = DiT(TINY)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, c_noise, null = torch.randn(4, 3, 8, 8), torch.randn(4), torch.zeros(4)
    a = model.forward_block(x, c_noise, torch.tensor([0, 1, 2, 3]), null, [0])
    b = model.forward_block(x, c_noise, torch.tensor([9, 8, 7, 6]), null, [0])
    torch.testing.assert_close(a, b)
