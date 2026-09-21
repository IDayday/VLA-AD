"""Fixed smooth local trajectory-neighbourhood probes, not accident probabilities."""
from common import *
from scipy.interpolate import CubicHermiteSpline

def deform(trajectory,lateral,longitudinal,initial_speed=0.):
 a=np.asarray(trajectory,dtype=np.float64);poses=np.vstack([np.zeros(3),a]);times=np.arange(len(poses))*.5;T=times[-1]
 velocities=np.gradient(poses[:,:2],times,axis=0);speeds=np.linalg.norm(velocities,axis=1);speeds[0]=initial_speed
 deriv=speeds[:,None]*np.column_stack([np.cos(poses[:,2]),np.sin(poses[:,2])]);s=CubicHermiteSpline(times,poses[:,:2],deriv)
 t=times[1:];u=t/T;b=16*u*u*(1-u)**2;db=32*u*(1-u)*(1-2*u)/T
 v=s(t,1);acc=s(t,2);norm=np.linalg.norm(v,axis=1);tangent=np.column_stack([np.cos(a[:,2]),np.sin(a[:,2])]);normal=np.column_stack([-tangent[:,1],tangent[:,0]])
 dt=(acc-tangent*np.sum(tangent*acc,axis=1)[:,None])/np.maximum(norm[:,None],1e-6);dt[norm<1e-6]=0;dn=np.column_stack([-dt[:,1],dt[:,0]])
 delta=b[:,None]*(longitudinal*tangent+lateral*normal)
 dv=db[:,None]*(longitudinal*tangent+lateral*normal)+b[:,None]*(longitudinal*dt+lateral*dn)
 out=a.copy();out[:,:2]+=delta;heading=np.arctan2((v+dv)[:,1],(v+dv)[:,0]);moving=np.linalg.norm(v+dv,axis=1)>1e-6;out[moving,2]=heading[moving]
 assert np.max(abs(out[-1,:2]-a[-1,:2]))<1e-12
 if lateral==0 and longitudinal==0:assert np.max(abs(np.arctan2(np.sin(out[:,2]-a[:,2]),np.cos(out[:,2]-a[:,2]))))<1e-10
 return out

def parameters(token,stream,n=8):
 rng=np.random.default_rng(seed(cfg()['seed'],token,stream));return rng.uniform(-1,1,size=(n,2))*np.array([.05,.20])

def dynamics_ok(d,baseline):
 keys={'peak_speed':'max_speed_mps','peak_acceleration':'max_acceleration_mps2','peak_jerk':'max_jerk_mps3','max_yaw_rate':'max_yaw_rate_rps'};mask=np.ones(len(d['peak_speed']),bool)
 for field,limit in keys.items():
  maxval=cfg()['perturbations']['dynamics'][limit];bound=np.maximum(maxval,baseline[field])
  mask &= np.isfinite(d[field])&(d[field]<=bound+1e-8)
 return mask
