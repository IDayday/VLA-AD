"""Generate local native inputs only; never reads a teacher expert or choice."""
from pathlib import Path
import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
import socket
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('/mnt/project/recogdrive-teacher2-a800-20261026')
FROZEN = Path('/mnt/project/-recogdrive-representation-teacher-2')
sys.path[:0] = [str(LEGACY / 'local'), str(FROZEN)]
from cache_common import read, write, sha, stamp


def save_npz(path, arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for key, value in arrays.items():
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asarray(value), allow_pickle=False)
            z.writestr(key + '.npy', b.getvalue())
    with np.load(temp, allow_pickle=False) as z:
        for key, value in arrays.items():
            assert z[key].tobytes() == np.asarray(value).tobytes(), key
    temp.replace(path)


def ground_truth(row):
    """Reuse frozen Scene/target builder; future data is only an optimizer label."""
    from types import SimpleNamespace
    from vlm_core import raw_log
    from navsim.common.dataclasses import Scene
    from model.recogdrive_features import TrajectoryTargetBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    data = raw_log(row['raw_path'])
    i = int(row['frame_index'])
    frames = data[i-3:i+9]
    assert len(frames) == 12 and frames[3]['token'] == row['token']
    # Scene.get_future_trajectory reads only global ego poses. Reuse its exact
    # implementation without loading unused map, camera, lidar or annotations.
    scene = Scene(scene_metadata=SimpleNamespace(num_history_frames=4, num_future_frames=8),
                  map_api=None, frames=[SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in frames])
    return TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=.5)).compute_targets(scene)['trajectory'].float().numpy()


def main(args):
    import numpy as np
    import torch
    from interface_teacher.runtime import bootstrap, Runtime
    task = read(args.manifest)
    population = task['population']
    rows = task['rows'][args.rank::args.world_size]
    if args.limit:
        rows = rows[:args.limit]
    assert rows
    private = ROOT / 'native_base' / population / 'workers' / args.manifest.stem / ('rank%03d' % args.rank)
    private.mkdir(parents=True, exist_ok=True)
    lock = (private / 'WORKER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = read(args.paths)
    bootstrap(args.paths, private)
    engine = Runtime('raw')
    torch.set_num_threads(1)
    model = engine.backbone.model
    language = model.language_model.model
    actual = dict(torch=str(torch.__version__), torch_git=torch.version.git_version,
                  cudnn=torch.backends.cudnn.version(), language_impl=language._attn_implementation,
                  language_classes=sorted({type(x.self_attn).__name__ for x in language.layers}),
                  language_layers=len(language.layers), vision_layers=len(model.vision_model.encoder.layers),
                  vision_flash_count=sum(x.attn.use_flash_attn for x in model.vision_model.encoder.layers),
                  TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
                  VLM_dtype=str(next(model.parameters()).dtype), head_dtype=str(next(engine.agent.action_head.parameters()).dtype))
    expected = read(LEGACY / 'local_rebase_20261027/CONTRACT.json')['numerical_protocol']
    assert actual == expected, (actual, expected)
    write(private / 'NUMERICS.json', dict(actual=actual, host=socket.gethostname(), worker_sha256=sha(__file__),
                                         manifest_sha256=sha(args.manifest), teacher_parameters_used=False))
    import vlm_core
    original_prepare = vlm_core.prepare
    layout = {}
    def capture(backbone, features):
        inputs, prompt = original_prepare(backbone, features)
        layout.update({k: inputs[k][0].detach().cpu().numpy().copy() for k in ['input_ids', 'attention_mask', 'position_ids']})
        return inputs, prompt
    vlm_core.prepare = capture
    versions = [(p, p._version) for p in list(engine.agent.parameters()) + list(engine.backbone.parameters())]
    started = time.monotonic()
    completed = 0
    for source in rows:
        population = source.get('population', task['population'])
        assert population in ('train', 'validation')
        token = source['token']
        directory = ROOT / 'native_base' / population / 'cases' / token
        done = directory / 'COMPLETE.json'
        if done.exists():
            old = read(done)
            assert old['token'] == token and sha(directory / 'BASE_INPUTS.npz') == old['arrays_sha256']
            completed += 1
            continue
        policy = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['recogdrive_no_routing_round1_20261026']
        assert policy['status'] == 'ACTIVE_TASK_BOUNDED_GRANT'
        assert any(socket.gethostname() in [h, h+'-worker-0'] for h in policy['temporary_hosts'])
        assert not (ROOT / 'STOP_REQUESTED.json').exists()
        row = dict(source['row'])
        with torch.no_grad():
            features, hidden, post, image, text, ids = engine.scene(row)
        values = dict(Base_interface=post[0].float().cpu().numpy(), image_mask=image.cpu().numpy(),
                      text_mask=text.cpu().numpy(), token_ids=ids.cpu().numpy(),
                      attention_mask=layout['attention_mask'], position_ids=layout['position_ids'])
        for key in ['history_trajectory', 'status_feature', 'high_command_one_hot']:
            values[key] = features[key].float().cpu().numpy()
        source_hash = None
        if population == 'train':
            source_path = Path(source['Base_npz'])
            if not source_path.is_absolute():
                source_path = LEGACY / 'cache' / population / source_path
            assert sha(source_path) == source['Base_npz_sha256']
            source_hash = source['Base_npz_sha256']
            with np.load(source_path, allow_pickle=False) as z:
                for key in ['Base_interface', 'image_mask', 'text_mask', 'token_ids', 'attention_mask', 'position_ids']:
                    assert values[key].tobytes() == z[key].tobytes(), (token, key, 'Native cache identity differs')
                values['Base_trajectories'] = np.array(z['Base_trajectories'], copy=True)
                values['Base_quality'] = np.array(z['Base_quality'], copy=True)
            values['hidden_FP32'] = hidden[0].float().cpu().numpy()
            values['GT'] = ground_truth(row)
        values['state24'] = np.concatenate([values['history_trajectory'].reshape(-1), values['status_feature'], values['high_command_one_hot']]).astype(np.float32)
        assert values['state24'].shape == (24,)
        save_npz(directory / 'BASE_INPUTS.npz', values)
        record = dict(status='COMPLETE_NATIVE_BASE_ONLY', time=stamp(), token=token,
                      global_index=int(row['index']), session=row['session'], population=population,
                      arrays_sha256=sha(directory / 'BASE_INPUTS.npz'), source_cache_sha256=source_hash,
                      full_tokens=len(ids), valid_tokens=int((image | text).sum()),
                      Base_interface_sha256=hashlib.sha256(values['Base_interface'].tobytes()).hexdigest(),
                      original_source=row, native_numerics=actual, teacher_parameters_or_choices_used=False,
                      original_hidden_included=population=='train', future_GT_only_optimizer_label=population=='train')
        write(done, record)
        assert all(not p.requires_grad and p._version == v and p.grad is None for p, v in versions)
        completed += 1
        write(private / 'LIVE.json', dict(status='GENERATING_NATIVE_BASE_INPUTS', time=stamp(), completed=completed,
                                          expected=len(rows), rank=args.rank, world_size=args.world_size,
                                          last_token=token, elapsed_seconds=time.monotonic()-started))
    write(private / 'COMPLETE.json', dict(status='COMPLETE', time=stamp(), scenes=completed,
                                        manifest_sha256=sha(args.manifest), worker_sha256=sha(__file__),
                                        elapsed_seconds=time.monotonic()-started, numerics=actual))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--paths', default=str(LEGACY/'paths.local.json'))
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        import traceback
        write(ROOT/'native_base'/('FAILED_'+uuid.uuid4().hex+'.json'), dict(time=stamp(), rank=args.rank, traceback=traceback.format_exc()))
        raise
