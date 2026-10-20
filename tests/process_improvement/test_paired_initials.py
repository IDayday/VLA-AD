"""A warm seed must never be compared with another seed's initial policy."""
import numpy as np
import pytest
from scripts.process_improvement import analyze


def fixture(monkeypatch):
    values={name:np.full((3,2,7),v) for name,v in [('warm42',.2),('warm43',.8),('new42',.25),('new43',.85)]}
    rows=[dict(token=str(i),driver_record=str(i//2),log_name=str(i)) for i in range(3)]
    monkeypatch.setattr(analyze,'predictions',lambda name,split,limit=None:(values[name],rows))
    monkeypatch.setattr(analyze,'analysis_source',lambda split:{'unit_only':True})
    monkeypatch.setattr(analyze,'write',lambda *args:None)


def test_before_after_retains_each_initial_policy(monkeypatch):
    fixture(monkeypatch)
    result=analyze.compare(['new42','new43'],['warm42','warm43'],'unit',repetitions=30,result_name='unit_paired')
    assert result['point']['G']==pytest.approx(.05)
    assert result['point']['M_minus']==0.
    assert result['paired_baselines']==['warm42','warm43'] and result['baseline'] is None
    wrong=analyze.compare(['new42','new43'],'warm42','unit',repetitions=30,result_name='unit_fixed')
    assert wrong['point']['G']==pytest.approx(.35)


def test_paired_initials_require_complete_pairing_and_unique_identity(monkeypatch):
    fixture(monkeypatch)
    with pytest.raises(ValueError,match='one baseline'):
        analyze.compare(['new42','new43'],['warm42'],'unit',result_name='bad')
    with pytest.raises(ValueError,match='explicit result'):
        analyze.compare(['new42','new43'],['warm42','warm43'],'unit')
