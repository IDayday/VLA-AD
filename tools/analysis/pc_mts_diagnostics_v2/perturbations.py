"""Amplitude-controlled, coherent trajectory deformations with explicit families."""
from v2_common import *

def perturb(traj,amplitude,family,direction,noise_seed,seeded_shape=False):
    a=np.asarray(traj,dtype=np.float64);u=np.arange(1,9)/8;rng=np.random.default_rng(noise_seed)
    forward=np.c_[np.cos(a[:,2]),np.sin(a[:,2])];lateral=np.c_[-np.sin(a[:,2]),np.cos(a[:,2])]
    if family=='lateral':delta=direction*u[:,None]**2*lateral
    elif family=='progress':delta=direction*u[:,None]**2*forward
    elif family=='curvature':
        angle=direction*u**2*.10;xy=a[:,:2];delta=np.c_[np.cos(angle)*xy[:,0]-np.sin(angle)*xy[:,1],np.sin(angle)*xy[:,0]+np.cos(angle)*xy[:,1]]-xy
    elif family=='endpoint':
        angle=rng.uniform(-np.pi,np.pi);delta=direction*u[:,None]**2*np.array([np.cos(angle),np.sin(angle)])
    elif family=='spline':delta=direction*(u**2*np.sin(np.pi*u+rng.uniform(-.4,.4)))[:,None]*lateral
    else:raise ValueError(family)
    if seeded_shape:delta*= (1+.1*rng.uniform(-1,1)*np.sin(np.pi*u))[:,None]
    norm=np.linalg.norm(delta,axis=1).max()
    if norm<1e-12:delta=direction*u[:,None]**2*lateral;norm=np.linalg.norm(delta,axis=1).max()
    delta*=amplitude/norm;out=a.copy();out[:,:2]+=delta
    before=np.diff(np.vstack([np.zeros((1,2)),a[:,:2]]),axis=0);after=np.diff(np.vstack([np.zeros((1,2)),out[:,:2]]),axis=0)
    change=np.arctan2(np.sin(np.arctan2(after[:,1],after[:,0])-np.arctan2(before[:,1],before[:,0])),np.cos(np.arctan2(after[:,1],after[:,0])-np.arctan2(before[:,1],before[:,0])))
    change*=np.minimum(np.linalg.norm(before,axis=1)/.2,1.);out[:,2]=np.arctan2(np.sin(a[:,2]+change),np.cos(a[:,2]+change))
    assert np.linalg.norm(out[:,:2]-a[:,:2],axis=1).max()<=amplitude+1e-9
    return out

def bank(traj,token,stream,heldout=True,seeded_shape=False):
    fingerprint=digest(np.asarray(traj).tolist())
    if heldout:
        settings=[(a,f,d) for a in CFG['heldout_amplitudes_m'] for f in CFG['heldout_families'] for d in CFG['heldout_directions']]
    else:settings=[(a,f,d) for a in CFG['selection_amplitudes_m'] for f,d in [('lateral',1),('curvature',-1)]]
    return np.stack([perturb(traj,a,f,d,seed(token,stream+'_'+fingerprint,i),seeded_shape) for i,(a,f,d) in enumerate(settings)])
