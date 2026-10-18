"""Native ReCogDrive observations, sampler and GRPO objective.

The current-front image and four ego-history states are reconstructed through
the repository's AgentInput/feature builders. Captured tokens are exactly the
tokens passed to the action head, before its 1536-to-384 projection.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import contextlib
import copy
from functools import lru_cache
from pathlib import Path
import pickle
from types import SimpleNamespace
import random
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
from .common import *


def configure():
    require_host()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_agent(online=True, checkpoint=RL):
    configure()
    agent = ReCogDriveAgent(TrajectorySampling(time_horizon=4, interval_length=.5),
        vlm_path=str(VLM), checkpoint_path=str(checkpoint), cam_type='single',
        vlm_type='internvl', dit_type='small', vlm_size='small', sampling_method='ddim',
        cache_hidden_state=not online, cache_mode=False, grpo=False,
        use_expert_features=False, allow_random_init=False)
    agent.initialize(); agent.eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    verified = 0
    for name, value in agent.action_head.state_dict().items():
        if name.startswith('action_aware_aux_head.'):
            assert not agent.action_head.config.use_action_aware_aux
            continue
        key = 'agent.action_head.' + name
        if key not in saved or not torch.equal(value.cpu(), saved[key]):
            raise ValueError(f'Checkpoint restore mismatch: {key}')
        verified += 1
    agent._effect_verified_tensors = verified
    return agent


@lru_cache(maxsize=2)
def read_log(path):
    with open(path, 'rb') as f: return pickle.load(f)


def observation(row):
    # Do not read future frames into AgentInput, even though log pickle has them.
    frames = read_log(row['metadata_path'])[row['start_idx']:row['start_idx']+4]
    frames = [dict(f, lidar_path=f.get('lidar_path') or '') for f in frames]
    sensors = SensorConfig.build_no_sensors(); sensors.cam_f0 = [3]
    return AgentInput.from_scene_dict_list(frames, Path(row['sensor_root']), 4,
                                          sensors, load_image_path=True)


def features_for(agent, row):
    result = {}
    for builder in agent.get_feature_builders():
        result.update(builder.compute_features(observation(row)))
    expected = {'history_trajectory', 'high_command_one_hot', 'status_feature', 'image_path_tensor'}
    if set(result) != expected:
        raise ValueError(f'Unreviewed observation keys: {set(result)}')
    return {k: v.unsqueeze(0) for k, v in result.items()}


class CapturedTokens(Exception): pass


def encode(agent, features, grad=False):
    payload = {}
    def capture(module, inputs):
        payload['tokens'] = inputs[0]
        raise CapturedTokens()
    hook = agent.action_head.feature_encoder.register_forward_pre_hook(capture)
    try:
        with torch.set_grad_enabled(grad):
            agent.forward(features)  # Native prompt/tokenization/vision path; no targets.
    except CapturedTokens:
        pass
    finally:
        hook.remove()
    if 'tokens' not in payload: raise RuntimeError('Native head input not captured')
    return payload['tokens']


def action_input(features, device='cuda'):
    history = features['history_trajectory'].to(device=device, dtype=torch.float32).reshape(-1, 4, 3)
    status = features['status_feature'].to(device=device, dtype=torch.float32)
    command = features['high_command_one_hot'].to(device=device, dtype=torch.float32)
    return BatchFeature(dict(history_trajectory=history, his_traj=history.reshape(-1, 12),
        status_feature=status, high_command_one_hot=command,
        state=torch.cat((status, history.reshape(-1, 12)), dim=1)))


def predict(head, tokens, inputs, token, seed, stream='inference'):
    head.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed_for(token, seed, stream)); torch.cuda.manual_seed(seed_for(token, seed, stream))
        return head.get_action(tokens, inputs, deterministic=False)['pred_traj'].float().cpu().numpy()[0]


def configure_grpo(head, rows):
    """Keep the repository's sampler, group normalization, BC and reward weights."""
    from .scoring import Scoring
    config = GRPOConfig()
    for key, value in vars(config).items():
        if key not in ('metric_cache_path', 'reference_policy_checkpoint', 'scorer_config'):
            setattr(head, key, value)
    head.metric_cache_loader = SimpleNamespace(metric_cache_paths={r['token']: Path(r['metric_cache_path']) for r in rows})
    reward = Scoring(progress_weight=config.scorer_config.progress_weight)
    head.simulator = reward.simulator; head.train_scorer = reward.scorer
    # GRPO's old_policy is the fixed IL BC reference, not an invented old logprob.
    reference = copy.deepcopy(head)
    reference._safe_load_reference_policy(str(IL))
    reference.eval(); reference.requires_grad_(False)
    head.old_policy = reference
    head.config.grpo = True
    return head


def trainable_head(head):
    return [p for name, p in head.named_parameters() if p.requires_grad and
            not name.startswith(('old_policy.', 'action_aware_aux_head.')) and name != 'eta']


def state_digest(module):
    import hashlib
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
