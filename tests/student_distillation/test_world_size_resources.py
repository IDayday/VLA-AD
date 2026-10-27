"""Rank-count metadata and admission change; the global objective stays fixed."""
import copy
import json
from pathlib import Path
import sys

import pytest

from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import EXPERIMENTS, SEEDS, Registry, load_config
from student_distillation.qualification import validate_ddp_execution
from student_distillation.run_suite import Controller, child_command, load_resources
from student_distillation.train import QUALIFICATION_CHECKS, formal_execution, verify_formal_config, verify_qualification


def resource_fixture(tmp_path, world):
    for name, payload in (('grant', {'task': 'test_task'}), ('policy', {'source': str(tmp_path / 'live_policy.json')})):
        atomic_json(tmp_path / (name + '.json'), payload)
    value = {'schema': 'native_round1_resources_v1', 'status': 'READY_FOR_FORMAL_DISPATCH', 'world_size': world,
             'authorization': {'path': str(tmp_path / 'grant.json'), 'sha256': sha256(tmp_path / 'grant.json')},
             'resource_policy': {'path': str(tmp_path / 'policy.json'), 'sha256': sha256(tmp_path / 'policy.json')},
             'allocations': {}}
    for n, run in enumerate(f'{arm}_seed{seed}' for arm in EXPERIMENTS for seed in SEEDS):
        value['allocations'][run] = {'host': 'permanent_host' + str(n % 2), 'ssh_target': 'local',
            'python': sys.executable, 'workdir': str(tmp_path), 'gpu_uuids': [f'GPU-{n}-{i}' for i in range(world)]}
    path = tmp_path / 'resources.json'
    atomic_json(path, value)
    return path, value


@pytest.mark.parametrize('world', [2, 4])
def test_eight_allocations_launch_actual_rank_count_and_preserve_full_budget(tmp_path, world):
    path, value = resource_fixture(tmp_path, world)
    allocations = load_resources(path, tmp_path, world)
    controller = Controller(tmp_path, Registry(tmp_path), allocations)
    assert controller.world_size == world
    assert len({uuid for allocation in allocations.values() for uuid in allocation['gpu_uuids']}) == 8 * world
    command = child_command('student_distillation.train', ['--resume', 'auto'], controller.world_size)
    assert f'--nproc_per_node={world}' in command
    admission = json.loads((tmp_path / 'RESOURCE_ALLOCATION.json').read_text())
    assert admission['world_size'] == world and admission['global_batch_size'] == 64
    assert admission['microbatch_per_device'] == 1 and admission['gradient_accumulation_steps'] == 64 // world
    root = Path(__file__).resolve().parents[2] / 'configs/student_distillation'
    for arm in EXPERIMENTS:
        cfg = load_config(root / (arm + '.yaml'), root / 'common.yaml')
        cfg['runtime'] = {'world_size': world}
        cfg['training']['gradient_accumulation_steps'] = 64 // world
        verify_formal_config(cfg)
        assert cfg['training']['epochs'] == 40 and cfg['data']['train_scenes'] == 85813
        assert cfg['training']['total_optimizer_updates'] == 53640
        assert formal_execution(cfg)['gradient_accumulation_steps'] * world == 64
    value.pop('world_size')
    atomic_json(path, value)
    assert len(next(iter(load_resources(path, tmp_path).values()))['gpu_uuids']) == world


def test_resource_width_mismatch_overlap_and_protection_are_rejected(tmp_path):
    path, base = resource_fixture(tmp_path, 2)
    with pytest.raises(ValueError, match='world_size differs'):
        load_resources(path, tmp_path, 4)
    for mutation, message in (
        (lambda v: v.update(world_size=4), 'world_size differs'),
        (lambda v: next(iter(v['allocations'].values()))['gpu_uuids'].append('GPU-extra'), 'uniformly'),
        (lambda v: list(v['allocations'].values())[1]['gpu_uuids'].__setitem__(0, 'GPU-0-0'), 'overlap'),
        (lambda v: v.update(protected_gpu_uuids=['GPU-0-0']), 'protected'),
        (lambda v: next(iter(v['allocations'].values())).update(world_size=4), 'world_size differs'),
    ):
        value = copy.deepcopy(base)
        mutation(value)
        atomic_json(path, value)
        with pytest.raises(ValueError, match=message):
            load_resources(path, tmp_path)


def ddp_receipt(world):
    return {'schema': 'nrkd_ddp_tail_check_v1', 'status': 'PASS', 'world_size': world,
            'global_slots': 64, 'real_scene_count': 53, 'zero_weight_padded_slots': 11,
            'native_microbatch': 1, 'gradient_accumulation_steps': 64 // world, 'DDP_adapter_only': True}


@pytest.mark.parametrize('world', [2, 4])
def test_qualification_tail_proof_must_match_rank_count_and_accumulation(world):
    proof = ddp_receipt(world)
    execution = validate_ddp_execution(proof, world)
    assert execution['world_size'] == world and execution['gradient_accumulation_steps'] == 64 // world
    with pytest.raises(ValueError, match='world_size differs'):
        validate_ddp_execution(proof, 6 - world)
    for key, value in [('gradient_accumulation_steps', 8), ('real_scene_count', 52), ('DDP_adapter_only', False)]:
        with pytest.raises(ValueError, match='53 real of 64 slots'):
            validate_ddp_execution(dict(proof, **{key: value}), world)


def qualification_fixture(tmp_path, world):
    keys = ('teacher_bundle', 'feature_manifest', 'targets_manifest', 'calibration', 'checkpoint')
    for key in keys:
        (tmp_path / key).write_text(key)
    paths = {key: str(tmp_path / key) for key in keys}
    paths['qualification_receipt'] = str(tmp_path / 'QUALIFICATION.json')
    cfg = {'paths': paths, 'runtime': {'world_size': world}, 'training': {'gradient_accumulation_steps': 64 // world}}
    identities = {'code_files': {}, 'files': {key: {'sha256': sha256(paths[key])} for key in keys[:-1]}}
    receipt = {'schema': 'nrkd_code_checks_v1', 'status': 'CODE_CHECKS_PASS', 'formal_asset_qualification': True,
               'code_sha256': {}, 'identities': {key + '_sha256': sha256(paths[key]) for key in keys},
               'execution': formal_execution(cfg), 'checks': {}}
    for check in QUALIFICATION_CHECKS:
        artifact = tmp_path / (check + '.json')
        payload = {'status': 'PASS'}
        if check == 'dit_input_gradient':
            payload['student_forward_native_parity'] = {'status': 'PASS', 'scenes': 64, 'time_indices': list(range(5)), 'bitwise': True}
        if check == 'next_update_recovery':
            payload['active_transition_recovery'] = {key: {'status': 'PASS', 'trained_Adam_state': True,
                'transition_loss_active': True, 'independent_loads': 2} for key in ('teacher', 'student')}
        if check == 'ddp_tail_objective':
            payload = ddp_receipt(world)
        atomic_json(artifact, payload)
        receipt['checks'][check] = {'status': 'PASS', 'artifact': {'path': str(artifact), 'sha256': sha256(artifact)}}
    atomic_json(paths['qualification_receipt'], receipt)
    return cfg, identities, receipt


@pytest.mark.parametrize('world', [2, 4])
def test_training_admission_cannot_substitute_other_world_size_qualification(tmp_path, world):
    cfg, identities, receipt = qualification_fixture(tmp_path, world)
    assert verify_qualification(cfg, identities)['execution']['world_size'] == world
    other = copy.deepcopy(cfg)
    other['runtime']['world_size'] = 6 - world
    other['training']['gradient_accumulation_steps'] = 64 // (6 - world)
    with pytest.raises(ValueError, match='different DDP world_size'):
        verify_qualification(other, identities)
    artifact = Path(receipt['checks']['ddp_tail_objective']['artifact']['path'])
    atomic_json(artifact, ddp_receipt(6 - world))
    receipt['checks']['ddp_tail_objective']['artifact']['sha256'] = sha256(artifact)
    atomic_json(cfg['paths']['qualification_receipt'], receipt)
    with pytest.raises(ValueError, match='DDP tail qualification must match'):
        verify_qualification(cfg, identities)
