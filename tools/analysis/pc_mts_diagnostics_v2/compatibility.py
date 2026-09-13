"""Independent Q-to-R calibration and continuous distance measurements."""
from v2_common import *
from sklearn.covariance import LedoitWolf

def fit(R,Q):
    qr=np.sort(distance(Q,R),axis=1)[:,:CFG['knn']].mean(1)
    x=np.asarray(R)[...,:2].reshape(len(R),16);lw=LedoitWolf().fit(x);cov=lw.covariance_
    floor=CFG['covariance_eigenvalue_floor_m2'];jitter=max(0.,floor-float(np.linalg.eigvalsh(cov).min()));cov=cov+np.eye(16)*jitter;precision=np.linalg.inv(cov)
    delta=np.asarray(Q)[...,:2].reshape(len(Q),16)-lw.location_;qm=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',delta,precision,delta),0))
    return dict(R=np.asarray(R),qr=qr,qm=qm,mean=lw.location_,precision=precision,shrinkage=float(lw.shrinkage_),jitter=jitter)

def position(a,cal):
    d=np.sort(distance(a,cal['R']),axis=1)[:,:CFG['knn']].mean(1)
    delta=np.asarray(a)[...,:2].reshape(len(a),16)-cal['mean'];maha=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',delta,cal['precision'],delta),0))
    return dict(policy_distance_knn=d,r_knn=d/max(np.median(cal['qr']),CFG['distance_ratio_denominator_floor_m']),q_holdout=np.searchsorted(np.sort(cal['qr']),d,side='right')/len(cal['qr'])*100,mahalanobis=maha,maha_ratio=maha/max(np.median(cal['qm']),1e-8),maha_percentile=np.searchsorted(np.sort(cal['qm']),maha,side='right')/len(cal['qm'])*100)
