import pytest

from scripts.bench2drive.validate_b2d_report_reward import (
    build_report_states,
    parse_efficiency_percentages,
    parse_efficiency_smoothness_log,
    record_is_infraction_free,
)


def _record():
    return {
        "save_name": "route_0",
        "scores": {
            "score_route": 100.0,
            "score_penalty": 1.0,
        },
        "infractions": {
            "collisions_vehicle": [],
            "red_light": [],
            "route_dev": [],
            "vehicle_blocked": [],
            "route_timeout": [],
            "min_speed_infractions": [
                "Average speed is 120.0% of the surrounding traffic's one",
                "Average speed is 1200.0% of the surrounding traffic's one",
                "Average speed is 80.0% of the surrounding traffic's one",
            ],
        },
    }


def test_efficiency_parser_applies_official_1000_percent_filter():
    values = parse_efficiency_percentages(_record()["infractions"]["min_speed_infractions"])
    assert values == (120.0, 80.0)


def test_route_state_adapter_preserves_strict_success_and_efficiency():
    record = _record()
    states = build_report_states((record,))
    assert len(states) == 1
    assert states[0].success
    assert states[0].driving_score_percentage == pytest.approx(100.0)
    assert states[0].efficiency_percentage == pytest.approx(100.0)

    record["infractions"]["red_light"] = ["red light"]
    assert not record_is_infraction_free(record)
    assert not build_report_states((record,))[0].success


def test_efficiency_parser_rejects_unparseable_official_message():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_efficiency_percentages(("missing percentage",))


def test_official_postprocessing_log_is_parsed_on_report_scales():
    efficiency, smoothness = parse_efficiency_smoothness_log(
        "Driving Efficiency=137.51\nDriving Smoothness=0.3771\n"
    )
    assert efficiency == pytest.approx(137.51)
    assert smoothness == pytest.approx(37.71)
