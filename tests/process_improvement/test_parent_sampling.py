from scripts.process_improvement.correct_diagnostic_sampling import parent_sample


def rows():return [dict(driver_record=parent,token=f'{parent}{i}') for parent in ['A','B','C'] for i in range(7)]


def test_parent_hash_round_robin_spreads_across_records():
    result=parent_sample(rows(),6,'train')
    assert len({r['token'] for r in result})==6
    for parent in ['A','B','C']:assert sum(r['driver_record']==parent for r in result)==2
    assert len({r['driver_record'] for r in result[:3]})==3


def test_parent_sampling_is_order_invariant_and_does_not_filter_by_reward():
    original=rows();first=parent_sample(original,8,'cal')
    modified=[dict(r,reward=999 if i%2 else -999) for i,r in enumerate(reversed(original))]
    second=parent_sample(modified,8,'cal')
    assert [r['token'] for r in first]==[r['token'] for r in second]
