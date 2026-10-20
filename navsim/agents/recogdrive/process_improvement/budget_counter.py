"""Count logical requests, actual candidate scoring, and exact duplicate reuse."""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np

_WORKER_SCORER = None


def score_chunk(job):
    """A spawn worker owns its own mutable scorer and path-indexed metric cache."""
    global _WORKER_SCORER
    from scripts.opd.scoring import Scoring, load_cache
    if _WORKER_SCORER is None:
        _WORKER_SCORER = Scoring()
    offset, path, trajectories = job
    values = _WORKER_SCORER.batch(load_cache(path), trajectories)['scores']
    return offset, hashlib.sha256(trajectories.tobytes()).hexdigest(), values


@dataclass
class BudgetCounter:
    logical_scores: int = 0
    physical_scores: int = 0
    score_cache_hits: int = 0
    reference_simulations: int = 0

    def state_dict(self):
        return asdict(self)


class CanonicalScorer:
    """One mutable NAVSIM scorer per process; engineering errors propagate.

    Exact trajectory bytes (not rounded coordinates) identify reusable labels.
    Zero PDMS is a legal observation. Nonfinite/malformed model output is an
    engineering failure in this experiment and invalidates the affected update.
    """
    def __init__(self, counter=None, progress_weight=5.0, workers=0, metric_paths=None):
        from scripts.opd.scoring import Scoring
        self.scorer = Scoring(progress_weight)
        self.counter = counter if counter is not None else BudgetCounter()
        self.scene = None
        self.cache = {}
        self.metric_paths = metric_paths or {}
        self.pool = None
        self.workers = workers
        if workers:
            if progress_weight != 5.0:
                raise ValueError('Parallel scoring is qualified only for canonical PDMS')
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None

    def score(self, scene_token, metric_cache, trajectories):
        array = np.asarray(trajectories, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (8, 3) or not len(array) or not np.isfinite(array).all():
            raise ValueError('Invalid model trajectory; no reward substitution is permitted')
        if self.scene != scene_token:
            self.cache.clear()
            self.scene = scene_token
        self.counter.logical_scores += len(array)
        keys = [hashlib.sha256(t.tobytes()).digest() for t in array]
        unseen = {}
        for key, trajectory in zip(keys, array):
            if key not in self.cache:
                unseen[key] = trajectory
        if unseen:
            trajectories = np.stack(list(unseen.values()))
            simulations = 1
            if self.pool and len(trajectories) >= 32:
                if scene_token not in self.metric_paths:
                    raise ValueError('Parallel scorer has no bound metric path for scene')
                jobs = []
                for indices in np.array_split(np.arange(len(trajectories)), self.workers):
                    if len(indices):
                        jobs.append((int(indices[0]), self.metric_paths[scene_token], trajectories[indices].copy()))
                returned = list(self.pool.map(score_chunk, jobs))
                result = np.empty((len(trajectories), 7), dtype=np.float64)
                for (offset, path, requested), (returned_offset, returned_sha, values) in zip(jobs, returned):
                    if returned_offset != offset or returned_sha != hashlib.sha256(requested.tobytes()).hexdigest():
                        raise RuntimeError('Asynchronous scorer candidate identity mismatch')
                    result[offset:offset+len(values)] = values
                simulations = len(jobs)
            else:
                result = self.scorer.batch(metric_cache, trajectories)['scores']
            if result.shape != (len(unseen), 7) or not np.isfinite(result).all():
                raise FloatingPointError('Scoring failed; distinguish this from a legal zero score')
            self.cache.update(zip(unseen, result))
            self.counter.physical_scores += len(unseen)
            self.counter.reference_simulations += simulations
        self.counter.score_cache_hits += len(array)-len(unseen)
        return np.stack([self.cache[key] for key in keys])
