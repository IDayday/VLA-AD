import numpy as np,pytest
from navsim.agents.recogdrive.opd.gating import advantage_gate

def score(x):
    a=np.ones((5,7));a[:,-1]=x;return a

@pytest.mark.parametrize('il,rl,category,gate',[(.2,.95,'A',1),(.95,.2,'B',0),(.95,.96,'C',0),(.2,.3,'D',1)])
def test_T20_categories(il,rl,category,gate):
    v=advantage_gate(score(il),score(rl));assert v['category']==category and v['gate']==gate

def test_T20_safety_and_uncertainty():
    il=score(.5);rl=score(.9);rl[2,0]=0
    assert advantage_gate(il,rl)['gate']==0
    with pytest.raises(ValueError):advantage_gate(il[:1],rl[:1])
    assert advantage_gate(score(.9),score(.9))['gate']==0
