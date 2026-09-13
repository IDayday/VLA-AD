"""Check observed token, future timestamps and GT geometry against original logs."""
from common_mass import *
import pickle,concurrent.futures,functools
from pyquaternion import Quaternion
@functools.lru_cache(maxsize=4)
def log(name):
    with open(Path('/mnt/navsim/trainval_navsim_logs/trainval')/(name+'.pkl'),'rb') as f:return pickle.load(f)
def check(s):
    frames=log(s['log'])[s['frame_start']:s['frame_start']+12];assert len(frames)==12
    origin=frames[3];assert origin['token']==s['token'];theta=Quaternion(*origin['ego2global_rotation']).yaw_pitch_roll[0];xy=np.array(origin['ego2global_translation'][:2]);result=[]
    for f in frames[4:]:
        delta=np.array(f['ego2global_translation'][:2])-xy;yaw=Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]-theta
        result.append([np.cos(theta)*delta[0]+np.sin(theta)*delta[1],-np.sin(theta)*delta[0]+np.cos(theta)*delta[1],np.arctan2(np.sin(yaw),np.cos(yaw))])
    error=float(np.max(np.abs(np.asarray(result)-s['gt'])));assert error<1e-5,(s['token'],error)
    stamps=[f['timestamp'] for f in frames[3:]];dt=np.diff(stamps);assert (dt>0).all(),(s['token'],dt)
    # Native Scene.get_future_trajectory uses sequential frame poses, then
    # returns TrajectorySampling(4 s, .5 s). Raw log timestamps can have gaps;
    # report that data limitation instead of silently changing the native GT.
    regular=bool(np.allclose(dt,500000,rtol=0,atol=25000))
    assert s['token'] in Path(s['metric_cache_path']).parts and s['log'] in Path(s['metric_cache_path']).parts
    return dict(token=s['token'],log=s['log'],max_GT_coordinate_error=error,dt_microseconds=dt.tolist(),raw_log_timestamp_regular=regular,observed_current_token_matches=True,metric_cache_log_token_path_matches=True,trajectory_includes_t0=False)
def main():
    with concurrent.futures.ProcessPoolExecutor(16) as pool:checks=list(pool.map(check,scenes()))
    flagged=[r['token'] for r in checks if not r['raw_log_timestamp_regular']]
    save(OUT/'audits/coordinate_and_token_identity.json',dict(status='PASS_NATIVE_FRAME_CONTRACT_WITH_TIMESTAMP_FLAGS' if flagged else 'PASS',scene_count=len(checks),checks=checks,timestamp_flagged_tokens=flagged,timestamp_flagged_scene_count=len(flagged),GT_definition='exact existing NAVSIM frame-indexed GT, nominal .5s; not relabeled as exact wall-clock interpolation',frame='ego x forward, y left',units='meters/radians',future_indices='4:12 after four observed frames; implicit origin at index3',native_definition_path='/mnt/project/VLA-AD-worktrees/a5-epdms-stage3/navsim/common/dataclasses.py:311',completed_at=utc()))
    print('1000 native GT coordinate/token contracts checked; timestamp flags',len(flagged),flagged,flush=True)
if __name__=='__main__':main()
