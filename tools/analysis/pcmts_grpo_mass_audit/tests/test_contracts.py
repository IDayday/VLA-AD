"""Scientific contracts: synthetic fixtures are tests only, never reported data."""
import sys,inspect,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common_mass import *
from build_raw import deduplicate,structured
from select_pools import score_select,pareto_select,geometry_select,mass_select,select,rank_pareto,matched_pairs
from analyze_B import candidate_statistics,union_metrics,representatives

def fixture(n=24):
    traj=np.zeros((n,8,3));traj[:,:,0]=np.linspace(1,8,8);traj[:,:,1]=np.arange(n)[:,None]*.1
    records=[dict(candidate_id=f'{i:04d}',source='ddv2',all_source_tags=['ddv2']) for i in range(n)]
    scores=np.ones((n,7));scores[:,6]=np.linspace(.90,1,n);return traj,records,scores

def test_01_streams_and_processes_disjoint():
    seeds=[random_seed(t,s,g) for t in ['token1','token2'] for s in ['A','B','C'] for g in range(128)]
    assert len(set(seeds))==len(seeds)
    ids=[sample_id('t',s,g,m,d) for s in ['A','B','C'] for g in range(2) for m in range(8) for d in range(6)]
    assert len(set(ids))==len(ids)
def test_02_raw_has_no_A_B_inputs():
    import build_raw
    text=inspect.getsource(build_raw);assert 'cache/rollouts/A' not in text and 'cache/rollouts/B' not in text
def test_03_shared_candidate_ids():
    t,r,s=fixture();before=digest(r)
    for m in METHODS:select(t,r,s,.9,np.full(24,1024),1024,np.zeros(24),m)
    assert digest(r)==before
def test_04_score_signature_isolated():
    assert list(inspect.signature(score_select).parameters)==['eligible','pdms','ids','geometry']
def test_05_pareto_signature_isolated():
    assert 'prob' not in inspect.getsource(pareto_select) and 'd_gt' not in inspect.getsource(pareto_select)
def test_06_gt_signature_no_sampling():
    assert list(inspect.signature(geometry_select).parameters)==['eligible','pdms','ids','geometry','d_gt','radius']
def test_07_mass_A_only():
    assert list(inspect.signature(mass_select).parameters)==['eligible','pdms','ids','geometry','hits_A','n_A','pmin']
def test_08_B_freeze_gate():
    import analyze_B,select_pools
    source=inspect.getsource(analyze_B.scene_task)
    assert source.index("frozen['selection_hashes']")<source.index("cache/rollouts/B")
    assert 'cache/rollouts/B' not in inspect.getsource(select_pools)
def test_09_cross_source_dedup():
    t=fixture()[0][0];items=[(t,dict(source='ddv2',checkpoint_hash='x')),(t.copy(),dict(source='drivor',checkpoint_hash='y'))]
    a,r=deduplicate(items,'t');assert len(a)==1 and r[0]['all_source_tags']==['ddv2','drivor']
    b,u=deduplicate(items[::-1],'t');assert np.array_equal(a,b) and r==u
def test_10_natural_sample_repetition_counted():
    t,r,s=fixture();b=np.repeat(t[:1],1024,0);bs=np.repeat(s[:1],1024,0)
    stats,_,_=candidate_statistics(distance(t[:1],b),bs,s[:1]);assert stats['hit_count_B'][0]==1024
def test_11_union_not_sum():
    near=np.ones((16,1024),bool);m=union_metrics(near,near);assert m['PoolMass']==1 and sum(m['marginal_coverage'])==1 and m['marginal_coverage'][1:]==[0]*15
def test_12_quality_same_candidate():
    near=np.array([[1,0],[0,1]],bool);quality=~near
    assert union_metrics(near,quality,group_size=2)['QualityMatchedPoolMass']==0
def test_13_fallback_truth():
    t,r,s=fixture();hits=np.zeros(24,int);hits[:3]=31
    p=select(t,r,s,.9,hits,1024,np.ones(24)*2,'grpo_mass_pc');assert p['strict_count']==3
    assert sum(x['selection_level']==1 for x in p['items'])==13
    assert all(x['strict_qualified']==x['probability_qualified'] for x in p['items'])
def test_14_no_duplicate_padding():
    t,r,s=fixture(3);p=select(t,r,s,.9,np.ones(3)*1024,1024,np.zeros(3),'score')
    assert p['operational_count']==3 and len(set(p['operational_indices']))==3 and p['status']=='DATA_INSUFFICIENT'
def test_15_source_attribution_symmetric():
    from analyze_B import source_fractions
    f=source_fractions([dict(all_source_tags=['ddv2','IL_native_C'])],[0]);assert f['source_ddv2_fraction']==f['source_IL_native_C_fraction']==.5
def test_16_geometry_units_and_origin():
    t=fixture()[0][0];a=structured(t,'test');assert len(a)==128
    for x,r in a:
        assert x.shape==(8,3) and np.max(abs(x[:,2]))<=np.pi
        assert np.max(np.linalg.norm(x[:,:2]-t[:,:2],axis=1))<=r['perturbation_parameters']['amplitude_max_point_m']+1e-8
    assert cfg()['representation']==dict(frame='ego_local_x_forward_y_left',units='meters_radians',points=8,interval_s=.5,horizon_s=4,includes_t0=False)
def test_17_pdms_points():
    t,r,s=fixture();near=np.zeros((1,1024));bs=np.repeat(s[:1],1024,0)
    stats,_,_=candidate_statistics(near,bs,s[:1]);assert abs(stats['local_mean_PDMS'][0]-90)<1e-10
def test_18_ERROR_not_zero():
    t,r,s=fixture();s[0]=np.nan;p=select(t,r,s,.9,np.ones(24)*1024,1024,np.zeros(24),'score');assert 0 not in p['operational_indices']
def test_19_real_native_state_frozen():
    a=read(OUT/'manifests/sampler_identity.json');assert a['weight_and_buffer_hash_before']==a['weight_and_buffer_hash_after'] and a['optimizer_updates']==0
def test_20_old_results_unchanged():
    for path,h in read(OUT/'audits/protected_artifact_hashes.json').items():assert sha(ROOT/path)==h,path
def test_21_global_sampling_once():
    import analyze_B
    source=inspect.getsource(analyze_B.scene_task)
    tree=ast.parse(source);function=tree.body[0]
    # Global metric assignment is outside the method/variant loops.
    assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='glob' for t in n.targets) for n in function.body)
def test_22_joint_denominators_and_zero_local():
    t,r,s=fixture();d=np.ones((2,1024));d[0,:20]=0;bs=np.repeat(s[:1],1024,0);bs[:10,0]=0
    stats,near,q=candidate_statistics(d,bs,s[:2]);assert stats['p_B'][0]==20/1024 and stats['p_B_safe'][0]==10/1024
    assert stats['local_safe_fraction'][0]==.5 and np.isnan(stats['local_mean_PDMS'][1])
def test_23_hash_stable_across_shards():
    assert random_seed('token','A',12)==random_seed('token','A',12)
    assert sample_id('token','A',12,2,3)!=sample_id('token','A',12,3,3)
def test_24_native_parity_real_eight_scenes():
    a=read(OUT/'manifests/sampler_identity.json');assert len(a['checks'])==8
    assert all(r['trajectory_max_abs']==0 and r['advantage_loss_max_abs']<=1e-8 for r in a['checks'])
    b=read(OUT/'audits/batch_native_parity.json');assert len(b['checks'])==8 and all(x['native_batched_max_abs']==0 for x in b['checks'])
def test_25_maximum_cardinality_matching():
    t,r,s=fixture(4);s[:,6]=[.9000,.9025,.9024,.9050]
    m=matched_pairs([0,1],[2,3],r,s);assert len(m['pairs'])==2 and all(x['PDMS_gap_points']<=.25+1e-8 for x in m['pairs'])
def test_26_native_threshold_31_hits():
    t,r,s=fixture();hits=np.full(24,30);p=select(t,r,s,.9,hits,1024,np.zeros(24),'grpo_mass_pc');assert p['strict_count']==0
    hits[:16]=31;p=select(t,r,s,.9,hits,1024,np.zeros(24),'grpo_mass_pc');assert p['strict_count']==16
def test_27_pareto_uses_all_fronts():
    assert np.array_equal(rank_pareto(np.array([[3,3,3],[2,2,2],[1,1,1]])),[1,2,3])
def test_28_near_representatives_not_modes():
    t,r,s=fixture(3);t[1]=t[0];assert representatives(t,.02)==2

def test_29_resume_preserves_original_pre_B_seal():
    from select_pools import seal_selection
    path=OUT/'manifests/selection_frozen.json';before=sha(path);frozen=read(path)
    returned=seal_selection(frozen['selection_hashes'],999999)
    assert returned['frozen_at']==frozen['frozen_at'] and sha(path)==before
