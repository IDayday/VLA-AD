import numpy as np
from navsim.agents.recogdrive.process_improvement.statistics import summarize_delta,paired_bootstrap,holm_adjust,local_noise_bootstrap


def test_truncate_after_scene_noise_mean_before_training_seed_mean():
    old=np.ones((2,2,2,7))*.5
    new=old.copy()
    new[0,0,:,-1]=[.8,.2]  # Sampling variation cancels before truncation.
    new[1,0,:,-1]=[.7,.7]
    new[0,1,:,-1]=[.4,.4]
    result=summarize_delta(old,new)
    assert abs(result['M_plus']-.05)<1e-10
    assert abs(result['M_minus']-.025)<1e-10
    assert abs(result['G']-(result['M_plus']-result['M_minus']))<1e-10
    assert result['single_output_drop_probability']>result['material_drop_fraction']


def test_identical_policy_has_exact_zero_paired_change():
    old=np.ones((4,5,7))*.8
    new=np.stack([old,old,old])
    result=paired_bootstrap(old,new,['a','a','b','c'],repetitions=50)
    assert result['point']['G']==0
    assert result['CI95']['G']==[0.,0.]
    assert result['parent_count']==3


def test_holm_multiplicity_monotonic():
    adjusted=holm_adjust({'a':.01,'b':.04,'c':.2})
    assert adjusted=={'a':.03,'b':.08,'c':.2}


def test_local_noise_bootstrap_keeps_comparisons_paired():
    raw=np.random.default_rng(1).normal(size=(8,2,16,1))
    values=np.concatenate([raw,raw,-raw],axis=-1)
    result=local_noise_bootstrap(values,['a']*3+['b']*3+['c']*2,repetitions=100)
    assert result['mean'][0]==result['mean'][1]
    assert result['CI95'][0][0]==result['CI95'][0][1]
    assert abs(result['CI95'][0][0]+result['CI95'][1][2])<1e-10
