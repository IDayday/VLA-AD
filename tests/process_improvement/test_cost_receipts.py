"""Standalone auxiliary costs must not vanish or double-count shard summaries."""
import json
from scripts.process_improvement.summarize_costs import diagnostic_receipts


def test_standalone_nested_and_sharded_costs_each_count_once(tmp_path):
    values={
        'grid/complete_0.json':dict(status='COMPLETE',budget=dict(physical_scores=10),seconds=1),
        'grid/results.json':dict(status='COMPLETE',budget=dict(physical_scores=10),seconds=1),
        'suffix/model/results.json':dict(status='COMPLETE',budget=dict(physical_scores=3),seconds=2),
        'dynamic/results.json':dict(status='COMPLETE',budget=dict(physical_scores=7),seconds=3)}
    for name,value in values.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    rows=diagnostic_receipts(tmp_path)
    assert sum((r['budget'] or {}).get('physical_scores',0) for r in rows)==20
    assert next(r for r in rows if r['name']=='grid' and r['shard']=='results')['budget'] is None
    assert any(r['name']=='suffix/model' and r['budget']['physical_scores']==3 for r in rows)
