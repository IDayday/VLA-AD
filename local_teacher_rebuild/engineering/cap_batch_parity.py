"""Compare singleton and nine-candidate native caps on actual local Base inputs."""
import argparse
import pathlib
import sys
import json
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--features', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); sys.path.insert(0, args.repo)
    from interface_teacher.native_cap import apply
    from student_distillation.data import SceneDataset, batch_scene
    from student_distillation.asset_identity import atomic_json, sha256_file
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data = SceneDataset(args.features, 'train', None)
    rng = np.random.default_rng(20261027)
    ratios = np.asarray([0., .1, 1.-1e-7, 1., 1.+1e-7, 2., 4., 100., 1e-5])
    cases = []
    for i in np.linspace(0, len(data) - 1, 32, dtype=int):
        b = batch_scene(data[int(i)], 'cuda:0')
        caps = np.linspace(.01, .32, 9, dtype=np.float64)
        q = rng.normal(size=(9, 384))
        q = q / np.sqrt(np.square(q).mean(-1, keepdims=True)) * (caps * ratios)[:, None]
        shifts = np.repeat(q.astype(np.float32)[:, None], 2, axis=1)
        shifts = torch.as_tensor(shifts, device='cuda:0'); limits = torch.as_tensor(caps, device='cuda:0')
        batched = apply(b['base_post'], b['image_mask'], b['text_mask'], shifts, limits)
        errors = []
        for k in range(9):
            single = apply(b['base_post'], b['image_mask'], b['text_mask'], shifts[k:k+1], limits[k:k+1])
            equal = {key: torch.equal(batched[key][k:k+1], single[key])
                     for key in ('actual_role_shifts', 'interfaces', 'scales', 'actual_RMS', 'cap_RMS')}
            rms_error = abs(float(batched['actual_RMS'][k]) - float(single['actual_RMS'][0]))
            if not all(equal[key] for key in ('actual_role_shifts', 'interfaces', 'scales', 'cap_RMS')):
                raise AssertionError(f'Native cap batch changed FP32 teacher targets: {b["token"]}, {k}, {equal}')
            if not np.isclose(float(batched['actual_RMS'][k]), float(single['actual_RMS'][0]), rtol=1e-10, atol=1e-12):
                raise AssertionError('Native post-rounding RMS differs beyond the specified verification tolerance')
            errors.append({'candidate': k, 'equal': equal, 'RMS_maxabs': rms_error})
        cases.append({'token': b['token'], 'candidates': errors})
    report = {'status': 'PASS_REAL_NATIVE_CAP_SINGLETON_VS_NINE', 'scenes': len(cases),
              'candidate_shapes': [1, 9], 'ratios': ratios.tolist(), 'cases': cases,
              'native_cap_sha256': sha256_file(pathlib.Path(args.repo) / 'interface_teacher/native_cap.py'),
              'feature_manifest_sha256': sha256_file(args.features),
              'max_RMS_abs_difference': max(c['RMS_maxabs'] for x in cases for c in x['candidates']),
              'all_FP32_actual_shifts_and_complete_interfaces_bitwise': True, 'formal_teacher_parameters_used': False}
    atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
