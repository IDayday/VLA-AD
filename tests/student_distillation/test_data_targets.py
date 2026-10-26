import json
from pathlib import Path
import numpy as np
import pytest
import torch
from student_distillation.asset_identity import canonical_sha, load_library_bundle
from student_distillation.build_targets import score_selection, compact_actual_shift, verify_historical_matrices
from student_distillation.calibration import build_train_behavior_panel, deterministic_log_coverage_indices
from student_distillation.feature_cache import validate_arrays


def test_score_units_strict_boundary_and_base_first_ties():
    q = np.zeros((4, 9), np.float64)
    q[:, 0] = 90
    q[0, 1] = 91
    q[1, 1] = np.nextafter(91., 100.)
    q[2, 1] = 90
    q[3, 1:3] = 95
    choice, gain, keep = score_selection(q)
    assert choice.tolist() == [1, 1, 0, 1]
    assert keep.tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match='Saved choice'):
        score_selection(q, np.zeros(4, np.int64))
    with pytest.raises(ValueError, match='FP64'):
        score_selection(q.astype(np.float32))


def test_historical_all_eleven_audit_is_explicitly_not_local_labels():
    result = verify_historical_matrices()
    assert result['gain1_repair_count'] == 31511
    assert len(result['matrix_hashes']) == 11
    assert result['identity'].startswith('HISTORICAL_PPU_ONLY')


def test_local_bundle_never_falls_back(tmp_path):
    p = tmp_path / 'BUNDLE.json'
    with pytest.raises(FileNotFoundError, match='not ready'):
        load_library_bundle(p)
    p.write_text(json.dumps({'status': 'FROZEN', 'teacher_origin': 'PPU'}))
    with pytest.raises(ValueError, match='local_refit'):
        load_library_bundle(p)


def test_compact_target_requires_actual_roles_identical_and_native_padding():
    from interface_teacher.native_cap import apply
    base = torch.randn(1, 5, 384)
    image = torch.tensor([[False, True, True, False, False]])
    text = torch.tensor([[False, False, False, True, False]])
    requested = torch.full((1, 2, 384), .03)
    result = apply(base, image, text, requested, torch.tensor([.02], dtype=torch.float64))
    shift, rms, cap = compact_actual_shift(base, image, text, result)
    assert shift.shape == (384,) and shift.dtype == np.float32
    assert rms <= cap + max(1e-7, cap * 1e-4)
    unequal = dict(result)
    unequal['actual_role_shifts'] = result['actual_role_shifts'].clone()
    unequal['actual_role_shifts'][0, 1, 0] += .001
    with pytest.raises(ValueError, match='identical'):
        compact_actual_shift(base, image, text, unequal)


def test_train_panel_largest_remainder_tie_is_log_bucket_then_hash():
    rows = [{'token': f't{i}', 'log_id': f'l{i//3}', 'base_mean_points': [60, 80, 95][i % 3]} for i in range(12)]
    panel = build_train_behavior_panel(rows, 5)
    assert panel['panel_size'] == 5
    assert [(s['log_id'], s['base_bucket']) for s in panel['strata'] if s['quota']] == [('l0',0),('l0',1),('l0',2),('l1',0),('l1',1)]
    assert panel['tokens_sha256'] == canonical_sha(panel['tokens'])
    assert panel == build_train_behavior_panel(rows, 5)


def test_mu_calibration_uses_repaired_train_and_log_coverage():
    rows = [{'token': f't{i}', 'log_id': f'l{i//3}', 'target_is_zero': i % 3 == 0} for i in range(12)]
    ix = deterministic_log_coverage_indices(rows, 4, repaired_only=True)
    assert {rows[i]['log_id'] for i in ix} == {'l0','l1','l2','l3'}
    assert all(not rows[i]['target_is_zero'] for i in ix)


def test_cache_state_order_and_layout_are_validated():
    arrays = {'base_post': np.zeros((4,384),np.float32),
              'image_mask': np.array([0,1,0,0],bool), 'text_mask': np.array([0,0,1,0],bool),
              'attention_mask': np.array([0,1,1,0],np.int64),
              'token_ids': np.arange(4,dtype=np.int64), 'position_ids': np.arange(4,dtype=np.int64),
              'state24': np.zeros(24,np.float32)}
    assert validate_arrays(arrays) == 4
    arrays['state24'][20] = 1
    with pytest.raises(ValueError, match='Command'):
        validate_arrays(arrays)


def test_mmap_conversion_local_library_and_compact_target_join(tmp_path):
    import gzip
    import csv
    from student_distillation.asset_identity import sha256_file
    from student_distillation.feature_cache import convert_population
    from student_distillation.build_targets import build_population
    from student_distillation.data import SceneDataset, batch_scene
    repo = tmp_path / 'repo'
    (repo / 'manifests').mkdir(parents=True)
    raw_row = dict(token='scenea', session='log-a', log_name='log-a-0', index='11', raw_rel='log.pkl', frame_index='3', metric_rel='log-a/scenea.pkl')
    with gzip.open(repo / 'manifests/train.csv.gz','wt') as f:
        writer = csv.DictWriter(f,fieldnames=list(raw_row))
        writer.writeheader(); writer.writerow(raw_row)
    source = tmp_path / 'source'
    source.mkdir()
    post = np.arange(5*384,dtype=np.float32).reshape(5,384) / 1000
    im = np.array([0,1,1,0,0],bool)
    tx = np.array([0,0,0,1,0],bool)
    np.savez_compressed(source/'scene.npz', Base_interface=post,image_mask=im,text_mask=tx,
                        token_ids=np.arange(5,dtype=np.int64),attention_mask=(im|tx).astype(np.int64),
                        position_ids=np.arange(5,dtype=np.int64),Base_quality=np.ones((5,7),np.float64)*.8)
    record={'token':'scenea','npz':'scene.npz','sha256':sha256_file(source/'scene.npz')}
    (source/'INDEX.jsonl').write_text(json.dumps(record)+'\n')
    out = tmp_path / 'features'
    spec = convert_population(source/'INDEX.jsonl','train',out,frozen_repo=repo,state_reader=lambda row,p: np.zeros(24,np.float32))
    checkpoint_sha='a'*64
    features = out/'MANIFEST.json'
    features.write_text(json.dumps({'schema':'nrkd_native_post_mmap_v1','preserve_native_layout':True,
                                   'source_checkpoint_sha256':checkpoint_sha,'populations':{'train':spec}}))
    local = tmp_path/'local'
    local.mkdir()
    basis=np.eye(384,48,dtype=np.float64)
    np.savez(local/'space.npz',basis=basis,center=np.zeros(384,np.float64),D0=np.float64(.1))
    np.savez(local/'operator.npz',coefficients=np.ones(48,np.float64)*.001)
    def ref(p): return {'path':str(p),'sha256':sha256_file(p)}
    config={'id':'NEW_LOCAL_D_C8_double','K':8,'space':ref(local/'space.npz'),'raw_shift_factor':2.,'cap_factor':2.,
            'experts':[{'method':'centroid_D0','alpha':2.,'cap_D0':2.,'parameters':ref(local/'operator.npz')} for _ in range(8)]}
    (local/'library.json').write_text(json.dumps(config))
    q=np.full((1,9),80.,np.float64);q[0,1]=82
    np.savez(local/'selection.npz',tokens=np.array(['scenea']),q_points=q,choice=np.array([1]),indices=np.array([11]))
    selection=ref(local/'selection.npz');selection['units']='points'
    bundle={'status':'FROZEN_LOCAL_TEACHER','teacher_origin':'local_refit','fit_population':'train',
            'source_checkpoint_sha256':checkpoint_sha,'artifact_root':str(local),'library':ref(local/'library.json'),
            'selections':{'train':selection}}
    (local/'BUNDLE.json').write_text(json.dumps(bundle))
    targets=tmp_path/'targets'
    spec,identity=build_population(features,local/'BUNDLE.json','train',targets)
    manifest=targets/'MANIFEST.json'
    manifest.write_text(json.dumps({'schema':'nrkd_compact_targets_v1','teacher_origin':'local_refit','teacher_identity':identity,'populations':{'train':spec}}))
    ds=SceneDataset({'feature_manifest':str(features),'targets_manifest':str(manifest)},'train','gain1')
    item=ds[0]
    assert torch.equal(item['base_post'],torch.from_numpy(post))
    assert item['metadata']['original_choice']==1 and not item['metadata']['target_is_zero']
    assert set(item)=={'base_post','image_mask','text_mask','state24','history12','status8','teacher_actual_shift','metadata'}
    batch=batch_scene(item,'cpu')
    assert batch['base_post'].shape==(1,5,384)
    # Reading a second scene after mutation cannot modify the frozen mmap.
    item['base_post'].zero_()
    assert torch.equal(ds[0]['base_post'],torch.from_numpy(post))
    label_free=SceneDataset(features,'train',None)
    assert not label_free[0]['metadata']['target_available']


def test_explicit_local_runtime_supports_new_selected_operator_family(tmp_path):
    from student_distillation.asset_identity import sha256_file
    basis=np.eye(384,48,dtype=np.float64)
    np.savez(tmp_path/'space.npz',basis=basis,center=np.zeros(384,np.float64),D0=np.float64(.1))
    np.savez(tmp_path/'op.npz',coefficients=np.ones(48,np.float64)*.001)
    def ref(p): return {'path':str(p),'sha256':sha256_file(p)}
    config={'id':'new_local13_family','K':8,'space':ref(tmp_path/'space.npz'),'raw_shift_factor':2.,'cap_factor':2.,
            'experts':[{'method':'new_valid_fitted_family','alpha':2.,'cap_D0':2.,'parameters':ref(tmp_path/'op.npz')} for _ in range(8)]}
    config['experts'][0]['cap_D0']=1.
    (tmp_path/'library.json').write_text(json.dumps(config))
    bundle={'status':'FROZEN_LOCAL_TEACHER','teacher_origin':'local_refit','fit_population':'train','source_checkpoint_sha256':'a'*64,
            'artifact_root':str(tmp_path),'library':ref(tmp_path/'library.json')}
    path=tmp_path/'BUNDLE.json';path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='runtime_predictor'):
        load_library_bundle(path)
    runtime=tmp_path/'local_runtime.py'
    runtime.write_text('''from interface_teacher.bank import Library
class LocalLibrary(Library):
 def __init__(self,*,bundle,config,space,parameters,artifact_root):
  self.config=config;self.basis=space['basis'];self.center=space['center'];self.D0=float(space['D0']);self.parameters=parameters;self.networks={}
 def _coefficient(self,slot,v):
  return self.parameters[slot]['coefficients'].copy()
''')
    bundle['runtime_predictor']=dict(ref(runtime),**{'class':'LocalLibrary'})
    path.write_text(json.dumps(bundle))
    loaded=load_library_bundle(path)
    assert loaded['identity']['runtime_predictor_sha256']==sha256_file(runtime)
    assert loaded['library'].name=='new_local13_family'
    assert loaded['cap_in_d0']==4. and loaded['student_cap_in_d0']==4.
    assert loaded['per_slot_cap_in_d0']==[2.,4.,4.,4.,4.,4.,4.,4.]
    base=torch.ones(1,4,384,dtype=torch.float32)
    image=torch.tensor([[0,1,0,0]],dtype=torch.bool);text=torch.tensor([[0,0,1,0]],dtype=torch.bool)
    actual=loaded['library'].targets(base,image,text,[1])
    shift,_,_=compact_actual_shift(base,image,text,actual)
    assert np.any(shift)


def test_feature_mmap_cache_is_bounded_without_invalidating_returned_views(tmp_path):
    from collections import OrderedDict
    from student_distillation.data import FeatureStore
    store=FeatureStore.__new__(FeatureStore)
    store.root=tmp_path;store._arrays={};store._shard_order=OrderedDict();store._mmap_cache_shards=1
    shards=[]
    for i in range(2):
        a=np.full((2,384),i+1,np.float32)
        path=tmp_path/f'post{i}.bin';path.write_bytes(a.tobytes())
        shards.append({'arrays':{'base_post':{'path':str(path),'dtype':np.dtype(np.float32).str,'shape':list(a.shape)}}})
    store.spec={'shards':shards}
    first=store._array(0,'base_post')[:1]
    second=store._array(1,'base_post')[:1]
    assert list(store._shard_order)==[1] and len(store._arrays)==1
    assert np.all(first==1) and np.all(second==2)
