from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from scripts.eval_last_vla_cot_corruption_pdm import build_metrics_summary
from scripts.last_vla_v2.pdm_scoring_utils import PDM_COMPONENT_KEYS, TrajectoryScorer


def test_corruption_proxy_metrics_schema_has_nullable_pdm_fields():
    args = Namespace(
        score_mode="proxy",
        zero_all_cot=True,
        zero_geometry_cot=False,
        zero_dynamic_cot=False,
        zero_ego_cot=False,
        zero_action_refine_cot=False,
        zero_coarse_prior=False,
        use_raw_vlm_context_ablation=False,
        drop_vlm_summary=False,
    )
    planner = SimpleNamespace(
        config=SimpleNamespace(
            last_vla_cot_bottleneck_mode=True,
            last_vla_raw_vlm_context_to_dit=False,
        )
    )
    scorer = TrajectoryScorer("proxy")
    summary = build_metrics_summary(
        args=args,
        planner=planner,
        scorer=scorer,
        score_results=[
            {
                "score": 0.3,
                "trajectory_l1": 0.2,
                "proxy_score": 0.3,
                **{key: None for key in PDM_COMPONENT_KEYS},
            }
        ],
        l1_values=[0.2],
        num_rows=1,
    )

    assert summary["score_mode"] == "proxy"
    assert summary["proxy_scoring_active"] is True
    assert summary["pdm_scoring_active"] is False
    assert summary["corruption_mode"] == "zero_all_cot"
    assert summary["target_teacher_tokens_disabled_in_eval"] is True
    assert abs(summary["trajectory_l1"] - 0.2) < 1e-6
    assert summary["mean_trajectory_l1"] == 0.2
    assert abs(summary["proxy_score_mean"] - 0.3) < 1e-6
    for key in PDM_COMPONENT_KEYS:
        assert summary[key] is None
