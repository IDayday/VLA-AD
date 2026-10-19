import numpy as np
from scripts.opd.analyze import bootstrap,compare


def test_paired_identity_has_zero_gain_interval():
    # Heterogeneous scores and unequal log sizes exercise shared paired draws.
    rng=np.random.default_rng(19)
    base=rng.uniform(.05,.95,(11,5,7));new=np.repeat(base[None],3,axis=0)
    logs=['a']*7+['b']*3+['c'];stable=np.ones(11,dtype=bool)
    result=bootstrap(new,base,logs,stable,reps=80)
    for k in ['G','M_plus','M_minus']:
        assert result[k]['estimate']==0 and result[k]['lower95']==0 and result[k]['upper95']==0


def test_identical_methods_have_zero_paired_contrast():
    rng=np.random.default_rng(23);base=rng.uniform(.05,.95,(13,5,7));new=rng.uniform(.05,.95,(3,13,5,7))
    result=compare(new,new.copy(),base,['a']*9+['b']*4,np.ones(13,dtype=bool),reps=80)
    assert all(v['estimate']==v['lower95']==v['upper95']==0 for v in result.values())
