from __future__ import annotations

import torch

from tests.test_last_rd_shapes import make_last_rd_planner


def test_last_rd_has_no_lazy_linear_modules():
    planner = make_last_rd_planner()
    assert not any(isinstance(module, torch.nn.LazyLinear) for module in planner.modules())
