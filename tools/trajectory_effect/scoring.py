"""Formal NAVSIM v1 PDMS, with candidate-independent reference normalization.

Only the experiment wrapper batches simulation. The canonical scorer itself is
unchanged. Each candidate's aggregation uses exactly its [reference, candidate]
pair, never the other candidates' progress. Scalar parity is an admission gate.
"""
from dataclasses import asdict
from functools import lru_cache
import lzma
import pickle
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex, EgoAreaIndex
from .common import FIELDS


@lru_cache(maxsize=8)
def load_cache(path):
    with lzma.open(path, 'rb') as f: return pickle.load(f)


class Scoring:
    def __init__(self, progress_weight=5.0):
        self.sampling = TrajectorySampling(num_poses=40, interval_length=.1)
        self.simulator = PDMSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, PDMScorerConfig(progress_weight=progress_weight))

    def scalar(self, cache, trajectory):
        r = pdm_score(cache, Trajectory(np.asarray(trajectory, dtype=np.float64)),
                      self.sampling, self.simulator, self.scorer)
        return np.array([asdict(r)[f] for f in FIELDS], dtype=np.float64)

    def batch(self, cache, trajectories):
        tr = np.asarray(trajectories, dtype=np.float64)
        if tr.ndim != 3 or tr.shape[1:] != (8, 3) or not np.isfinite(tr).all() or not len(tr):
            raise ValueError(f'Invalid 4s/0.5s ego rear-axle trajectory array: {tr.shape}')
        ego = cache.ego_state
        states = [get_trajectory_as_array(cache.trajectory, self.sampling, ego.time_point)]
        states += [get_trajectory_as_array(transform_trajectory(Trajectory(t), ego),
                                           self.sampling, ego.time_point) for t in tr]
        simulated = self.simulator.simulate_proposals(np.stack(states), ego)
        sc = self.scorer
        naive = sc.score_proposals(simulated, cache.observation, cache.centerline,
                                   cache.route_lane_ids, cache.drivable_area_map).copy()
        multiply = sc._multi_metrics.prod(axis=0)
        gated = sc._progress_raw * multiply
        denominator = np.maximum(gated[0], gated[1:])
        progress = np.where(denominator > sc._config.progress_distance_threshold,
                            gated[1:] / np.maximum(denominator, 1e-30),
                            (multiply[1:] != 0).astype(np.float64))
        weighted = sc._weighted_metrics[:, 1:].copy()
        weighted[WeightedMetricIndex.PROGRESS] = progress
        score = (weighted * sc._config.weighted_metrics_array[:, None]).sum(0)
        score = score / sc._config.weighted_metrics_array.sum() * multiply[1:]
        values = np.stack([sc._multi_metrics[MultiMetricIndex.NO_COLLISION, 1:],
                           sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1:], progress,
                           weighted[WeightedMetricIndex.TTC], weighted[WeightedMetricIndex.COMFORTABLE],
                           weighted[WeightedMetricIndex.DRIVING_DIRECTION], score], axis=1)
        # Infraction indices are native scorer diagnostics, not future model inputs.
        offroad = sc._ego_areas[1:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        first_dac = np.where(offroad.any(1), offroad.argmax(1) * .1, np.inf)
        return dict(scores=values, naive_batch_scores=naive[1:],
                    simulated_states=simulated[1:], progress_raw_m=sc._progress_raw[1:].copy(),
                    reference_progress_raw_m=float(sc._progress_raw[0]),
                    first_collision_s=sc._collision_time_idcs[1:].copy() * .1,
                    first_dac_s=first_dac, first_ttc_check_failure_s=sc._ttc_time_idcs[1:].copy() * .1)
