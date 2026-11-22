"""Build an exact Stage 1 feature and GT cache shard from log/token rows."""
import argparse
import time
from pathlib import Path

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.training.dataset import dump_feature_target_to_pickle


def write_cache(path, values):
    temporary = path.with_suffix(".tmp")
    dump_feature_target_to_pickle(temporary, values)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for line in args.manifest.read_text().splitlines():
        log, token = line.split("\t")
        groups.setdefault(log, []).append(token)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    feature = ReCogDriveFeatureBuilder(
        model_type="internvl", checkpoint_path=str(args.stage1), device="cuda:0",
        cache_hidden_state=True, cache_mode=True, use_expert_features=False, use_jepa=False, use_vggt=False,
    )
    target = TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=.5))
    names = (feature.get_unique_name() + ".gz", target.get_unique_name() + ".gz")
    sensor_config = SensorConfig.build_all_sensors(include=[0, 1, 2, 3])
    done = produced = 0
    start = time.monotonic()
    print(f"START rows={sum(map(len, groups.values()))} logs={len(groups)}", flush=True)
    with torch.inference_mode():
        for log, tokens in groups.items():
            pending = []
            for token in tokens:
                output = args.output / log / token
                complete = all((output / name).is_file() for name in names)
                if args.existing is not None:
                    previous = args.existing / log / token
                    complete |= all((previous / name).is_file() for name in names)
                if complete:
                    done += 1
                    continue
                pending.append(token)
            if not pending:
                continue
            scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                       has_route=True, log_names=[log], tokens=pending)
            loader = SceneLoader(args.logs, args.sensors, scene_filter, sensor_config, load_image_path=True)
            if set(loader.tokens) != set(pending):
                raise RuntimeError(f"{log}: missing tokens {set(pending) - set(loader.tokens)}")
            for token in loader.tokens:
                scene = loader.get_scene_from_token(token)
                output = args.output / log / token
                output.mkdir(parents=True, exist_ok=True)
                write_cache(output / names[0], feature.compute_features(scene.get_agent_input()))
                write_cache(output / names[1], target.compute_targets(scene))
                produced += 1
                done += 1
                if produced % 25 == 0:
                    print(f"PROGRESS done={done} produced={produced} seconds={time.monotonic()-start:.2f}", flush=True)
    print(f"COMPLETE done={done} produced={produced} seconds={time.monotonic()-start:.2f} "
          f"peak_mib={torch.cuda.max_memory_reserved()//2**20}", flush=True)


if __name__ == "__main__":
    main()
