"""The fixed eta encoding is intentional; nonfinite core/optimizer data fails."""
import copy
import pytest
import torch
from scripts.process_improvement.qualify_grpo_partition32 import compare_states


def state():
    return dict(step=4,physical_scores=1024,full_student={
        'eta.eta_logit':torch.tensor([float('inf')]),
        'old_policy.eta.eta_logit':torch.tensor([float('inf')]),
        'weight':torch.ones(2)},optimizer=dict(state={0:dict(exp_avg=torch.ones(2))}))


def test_exact_native_eta_constant_is_not_nonfinite_core_error():
    left=state();assert compare_states(left,copy.deepcopy(left),False)==0.
    assert compare_states(left,copy.deepcopy(left),True)==0.


@pytest.mark.parametrize('key',['weight','eta.eta_logit','old_policy.eta.eta_logit'])
def test_nan_or_inf_cannot_hide_as_matching_values(key):
    left=state();left['full_student'][key].fill_(float('nan') if 'eta' in key else float('inf'))
    with pytest.raises(AssertionError):compare_states(left,copy.deepcopy(left),False)


def test_fixed_eta_mismatch_and_optimizer_inf_fail():
    left=state();right=copy.deepcopy(left);right['full_student']['eta.eta_logit'].fill_(0)
    with pytest.raises(AssertionError):compare_states(left,right,False)
    left['optimizer']['state'][0]['exp_avg'].fill_(float('inf'))
    with pytest.raises(AssertionError):compare_states(left,copy.deepcopy(left),False)
