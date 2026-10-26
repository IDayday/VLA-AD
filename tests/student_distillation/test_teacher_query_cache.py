from copy import deepcopy
import builtins
import numpy as np
import pytest
import torch
from test_native_bridge_replay import fixture
from student_distillation.noise import make_noise_tape
from student_distillation.teacher_query_cache import (cached_teacher_query,clear_teacher_query_cache,
    teacher_query_cache_statistics,teacher_query_cache_debug_last,validate_cache_receipt,
    native_runtime_identity,cache_runtime_identity,SCHEMA,QUALIFIED)


def setup():
    head,bridge,hidden,history,status=fixture()
    post=head.feature_encoder(hidden).detach()
    context=bridge.encode_context(history,status)
    tape=make_noise_tape(26,'cpu')
    identity={'feature_manifest_sha256':'a'*64,'targets_manifest_sha256':'b'*64,
              'teacher_bundle_sha256':'c'*64,'teacher_target':'gain1'}
    return head,bridge,post,context,tape,identity


def query(values,j,**kw):
    head,bridge,post,context,tape,identity=values
    options=dict(token='real_fixture_token',training_seed=20261027,noise_bank=0,identity=identity,engineering=True)
    options.update(kw)
    return cached_teacher_query(bridge,post,context,tape,j,**options)


def exact(a,b):assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def test_prefix_all_steps_reverse_warm_last_floor_and_rng():
    values=setup();head,bridge,post,context,tape,_=values
    with torch.no_grad():
        baseline=bridge.states_from_post(post,context,tape)
        rng=torch.get_rng_state().clone()
        for j in range(5):
            state,mean,hit=query(values,j)
            assert not hit
            exact(state,baseline['states'][j]);exact(mean,baseline['means'][j])
        assert teacher_query_cache_statistics(bridge)['native_mean_queries']==5
        snapshot=teacher_query_cache_debug_last(bridge,engineering=True)
        exact(snapshot['states'][5],baseline['before_final_clip'])
        for j in reversed(range(5)):
            state,mean,hit=query(values,j)
            assert hit;exact(state,baseline['states'][j]);exact(mean,baseline['means'][j])
            state.zero_();mean.zero_()
            state2,mean2,_=query(values,j)
            exact(state2,baseline['states'][j]);exact(mean2,baseline['means'][j])
        assert torch.equal(rng,torch.get_rng_state())
        clear_teacher_query_cache(bridge)
        for j in reversed(range(5)):
            state,mean,hit=query(values,j)
            assert hit == (j<4)
            exact(state,baseline['states'][j]);exact(mean,baseline['means'][j])
        assert teacher_query_cache_statistics(bridge)['native_mean_queries']==10


def test_lru_namespace_tape_context_and_bounded_capacity():
    v=setup();bridge=v[1]
    with torch.no_grad():
        assert query(v,0,max_entries=2)[2] is False
        assert query(v,0,max_entries=2,token='other')[2] is False
        assert query(v,0,max_entries=2)[2] is True # refresh LRU
        assert query(v,0,max_entries=2,noise_bank=1)[2] is False
        assert query(v,0,max_entries=2,token='other')[2] is False
        assert teacher_query_cache_statistics(bridge)['entries']==2
        identity=dict(v[-1],teacher_target='full')
        assert query(v,0,max_entries=2,identity=identity)[2] is False
        changed=list(v);changed[4]=v[4].clone();changed[4][0,0,0,0]+=.1
        assert query(changed,0,max_entries=2,identity=identity)[2] is False
        changed=list(v);changed[3]=(v[3][0]+.01,v[3][1])
        assert query(changed,0,max_entries=2,identity=identity)[2] is False
        assert teacher_query_cache_statistics(bridge)['entries']<=2


def test_frozen_no_grad_C_only_checks_and_parameter_mutation():
    v=setup()
    with pytest.raises(ValueError,match='no-grad'):query(v,0)
    with torch.no_grad():
        with pytest.raises(ValueError,match='D and H-only'):query(v,0,identity=dict(v[-1],state_source='student'))
        with pytest.raises(ValueError,match='1..500000'):query(v,0,max_entries=500001)
        query(v,0)
        next(v[0].parameters()).add_(.01)
        with pytest.raises(ValueError,match='parameters/gradients changed'):query(v,0)


def test_cache_never_opens_files_and_receipt_guard_is_preloaded():
    v=setup()
    with torch.no_grad():
        with pytest.raises(ValueError,match='Preload'):query(v,0,engineering=False,receipt='not_a_path_to_open')
    # Cache math and metadata are RAM-only, even on its first real cold key.
    def forbidden(*a,**kw):raise AssertionError('cache opened a file')
    old=builtins.open;builtins.open=forbidden
    try:
        with torch.no_grad():
            query(v,0);query(v,4);assert query(v,2)[2]
    finally:builtins.open=old


def test_receipt_validation_requires_complete_evidence_and_current_runtime():
    v=setup()
    proof=dict(real_scenes=64,time_indices=list(range(5)),training_seeds=[20261027,20261028],noise_banks=list(range(5)))
    for name in ('cold_bytewise','warm_bytewise','reverse_order_bytewise','eviction_bytewise',
                 'last_step_floor_and_unclipped_initial_exact','rng_unchanged','head_unchanged',
                 'returned_values_do_not_alias_cache','no_gradient_or_optimizer_updates'):proof[name]=True
    receipt={'schema':SCHEMA,'status':QUALIFIED,'policy':'C_ONLY_FROZEN_TEACHER_PREFIX','cache_module_sha256':'d'*64,
             'native_source_sha256':{'student_distillation/native_bridge.py':'e'*64},
             'cache_runtime_sha256':cache_runtime_identity(),'native_runtime_sha256':native_runtime_identity(v[1]),
             'feature_manifest_sha256':'a'*64,'proof':proof}
    assert validate_cache_receipt(receipt)['cache_module_sha256']=='d'*64
    with torch.no_grad():
        query(v,2,engineering=False,receipt=receipt)
        assert query(v,1,engineering=False,receipt=receipt)[2]
    broken=deepcopy(receipt);broken['proof']['warm_bytewise']=False
    with pytest.raises(ValueError,match='warm_bytewise'):validate_cache_receipt(broken)
