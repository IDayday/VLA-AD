import torch

from navsim.agents.recogdrive.stage3_reference_cache import (
    Stage3ReferenceCache,
    select_coherent_reference,
)


def _row(base: float):
    return {
        "scalar": base,
        "ep": base + 1,
        "ttc": base + 2,
        "quality": base + 3,
        "nc": base + 4,
        "dac": base + 5,
        "ddc": base + 6,
    }


def test_reference_components_come_from_one_trajectory() -> None:
    selected = select_coherent_reference(
        _row(10.0),
        _row(20.0),
        gt_feasible=True,
        stage2_feasible=True,
    )
    assert selected["selected_source"] == "stage2"
    assert selected["selected"] == _row(20.0)
    assert selected["gt_ddc"] == 16.0

    cache = Stage3ReferenceCache(
        payload={"metadata": {"benchmark": "navsim_v1"}, "records": {"scene": selected}},
        benchmark="navsim_v1",
    )
    batch = cache.get(["scene"], "cpu", torch.float32)
    assert batch.scalar.item() == 20.0
    assert batch.ep.item() == 21.0
    assert batch.ddc.item() == 26.0
    assert batch.gt_ddc.item() == 16.0
    assert batch.selected_source_code.item() == 2


def test_reference_cache_missing_token_is_an_error() -> None:
    record = select_coherent_reference(_row(1.0), _row(2.0), gt_feasible=True, stage2_feasible=True)
    cache = Stage3ReferenceCache(payload={"metadata": {}, "records": {"known": record}})
    try:
        cache.get(["missing"], "cpu", torch.float32)
    except KeyError as error:
        assert "Dynamic/random reference fallback is forbidden" in str(error)
    else:
        raise AssertionError("missing reference token must fail")
