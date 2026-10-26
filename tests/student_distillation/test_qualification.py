from types import SimpleNamespace
import numpy as np
import pytest
import torch
from student_distillation.qualification import native_replay, check_input_gradients, nested_equal
from student_distillation.noise import make_noise_tape
from test_native_bridge_replay import fixture


def test_qualification_traces_real_native_method_at_all_five_steps():
    head,bridge,hidden,history,status=fixture()
    post=head.feature_encoder(hidden).detach()
    batch={'history12':history,'status8':status,'base_post':post}
    result=native_replay(bridge,post,batch,make_noise_tape(55,'cpu'))
    assert result['all_steps_exact'] and len(result['trace'])==5
    assert result['trajectory'].shape==(1,8,3)
    gradients=check_input_gradients(bridge,batch,make_noise_tape(55,'cpu'))
    assert len(gradients['steps'])==5
    assert all(s['post_gradient_norm']>0 for s in gradients['steps'])


def test_independent_recovery_comparison_checks_optimizer_tensors_and_rng():
    value={'adapter':{'p':torch.ones(3)},'optimizer':{'exp_avg':torch.arange(3.)},'RNG':np.arange(4,dtype=np.uint8)}
    other={'adapter':{'p':torch.ones(3)},'optimizer':{'exp_avg':torch.arange(3.)},'RNG':np.arange(4,dtype=np.uint8)}
    assert nested_equal(value,other)
    other['optimizer']['exp_avg'][1]=8
    assert not nested_equal(value,other)


def test_student_metadata_scope_keeps_all_five_means_and_restores_native_flags():
    from student_distillation.qualification import student_forward_native_parity
    head,bridge,hidden,history,status=fixture()
    post=head.feature_encoder(hidden).detach()
    batches=[{'token':f'fixture{i}','history12':history,'status8':status,'base_post':post} for i in range(64)]
    class Noise:
        def tape(self,token,j,device):return make_noise_tape(123,device)
    assert not torch.are_deterministic_algorithms_enabled()
    result=student_forward_native_parity(bridge,batches,Noise())
    assert result['status']=='PASS' and result['scenes']==64
    assert result['bitwise'] and result['time_indices']==[0,1,2,3,4]
    assert not torch.are_deterministic_algorithms_enabled()
