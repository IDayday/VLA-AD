import numpy as np


def test_linear_harm_bootstrap_retains_parent_and_paired_noise():
    from navsim.agents.recogdrive.process_improvement.statistics import paired_linear_bootstrap
    old=np.zeros((5,3,7));old[:,0]=.1;old[:,1]=.4;old[:,2]=.9
    new=old+.025
    result=paired_linear_bootstrap(old,new,['A','A','B','B','B'],repetitions=1000)
    assert abs(result['point']['G']-.025)<1e-14
    np.testing.assert_allclose(result['CI95']['G'],[.025,.025],atol=1e-14)
    assert result['parent_count']==2 and result['training_seeds']==1


def test_risk_difference_does_not_create_a_training_seed_ensemble():
    from navsim.agents.recogdrive.process_improvement.statistics import risk_difference_bootstrap
    baseline=np.full((2,2,7),.5);control=np.stack([baseline,baseline]);new=control.copy()
    new[0,:,:,-1]=0.;new[1,:,:,-1]=1.
    result=risk_difference_bootstrap(baseline,control,new,['A','B'],repetitions=32)
    assert result['point']['G']==0 and result['point']['M_minus']==.25
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


def test_shared_inference_seeds_do_not_gain_precision_by_repeating_a_policy():
    # Actual evaluation uses the SAME five semantic noise IDs for every
    # training policy. Repeating one policy cannot create fifteen noise IDs.
    old=np.zeros((2,5,7));new=old.copy();new[:,:,-1]=[0,0,1,1,1]
    single=paired_bootstrap(old,new[None],['a','b'],repetitions=3000)
    repeated=paired_bootstrap(old,np.stack([new]*3),['a','b'],repetitions=3000)
    assert single['CI95']['G']==[.2,1.]
    # Repeating the arithmetic mean can change its last binary digit only.
    np.testing.assert_allclose(repeated['CI95']['G'],single['CI95']['G'],rtol=0,atol=np.finfo(np.float64).eps)


def test_shared_noise_is_preserved_in_method_risk_contrasts():
    from navsim.agents.recogdrive.process_improvement.statistics import risk_difference_bootstrap
    baseline=np.zeros((2,5,7));control=np.stack([baseline]*3)
    policy=baseline.copy();policy[:,:,-1]=[0,0,1,1,1]
    result=risk_difference_bootstrap(baseline,control,np.stack([policy]*3),['a','b'],repetitions=3000)
    np.testing.assert_allclose(result['CI95']['G'],[.2,1.],rtol=0,atol=np.finfo(np.float64).eps)


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
