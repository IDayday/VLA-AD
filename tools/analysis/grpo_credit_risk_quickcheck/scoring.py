"""Archived native scoring with exposed simulator and progress diagnostics."""
from common import *
import lzma,pickle,dataclasses,copy
install_runtime()
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.scoring.fast_pdm_scorer import FastPDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex,MultiMetricIndex,WeightedMetricIndex,BBCoordsIndex
NAMES=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']
FIELDS=['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance','score']
SAMPLING=TrajectorySampling(num_poses=40,interval_length=.1)
def load_cache(row):
 with lzma.open(row['metric_cache_path'],'rb') as f:return pickle.load(f)
def score(cache,trajectories,training=False,fast=False,margins=False):
 # Independent objects prevent mutable scorer state leaking between calls.
 simulator=PDMSimulator(SAMPLING);scorer=(FastPDMScorer if fast else PDMScorer)(SAMPLING,PDMScorerConfig(progress_weight=10. if training else 5.,ttc_weight=5.,comfortable_weight=2.))
 results=pdm_score_batch_same_cache(cache,np.asarray(trajectories,dtype=np.float64),SAMPLING,simulator,scorer,use_exact_array_conversion=True)
 metrics=np.asarray([[float(getattr(r,n)) for n in FIELDS] for r in results])
 states=scorer._states.copy();mult=scorer._multi_metrics.prod(axis=0);raw=scorer._progress_raw.copy();gated=raw*mult;den=np.maximum(gated[0],gated[1:])
 speed=np.linalg.norm(states[...,3:5],axis=-1);acc=np.linalg.norm(states[...,5:7],axis=-1);jerk=np.linalg.norm(np.diff(states[...,5:7],axis=1)/.1,axis=-1)
 signed=[]
 from shapely.geometry import Point
 for coords in scorer._ego_coords:
  p=cache.centerline.project([Point(*coords[0,BBCoordsIndex.CENTER]),Point(*coords[-1,BBCoordsIndex.CENTER])]);signed.append(float(p[1]-p[0]))
 diagnostic=dict(raw_progress=raw[1:],signed_centerline_progress=np.asarray(signed[1:]),gated_progress=gated[1:],
  baseline_raw_progress=np.full(len(results),raw[0]),baseline_gated_progress=np.full(len(results),gated[0]),
  ep_denominator=den,low_progress_branch=den<=scorer._config.progress_distance_threshold,
  end_speed=speed[1:,-1],peak_speed=speed[1:].max(1),peak_acceleration=acc[1:].max(1),peak_jerk=jerk[1:].max(1),
  mean_acceleration=acc[1:].mean(1),mean_jerk=jerk[1:].mean(1),
  max_yaw_rate=np.abs(states[1:,:,StateIndex.ANGULAR_VELOCITY]).max(1),
  ttc_infraction_time=scorer._ttc_time_idcs[1:]*.1)
 if margins:diagnostic.update(nominal_margins(cache,scorer))
 return metrics,diagnostic,states[1:]

def nominal_margins(cache,scorer):
 import shapely
 from shapely.geometry import Point
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer
 polys=scorer._ego_polygons[1:];n,h=polys.shape;closest=np.full(n,np.inf);times=np.full(n,-1);objects=np.full(n,'',object);geometric_overlap=np.zeros(n,bool)
 for j in range(h):
  occ=cache.observation[j];ids=[i for i,t in enumerate(occ.tokens) if cache.observation.red_light_token not in t]
  if not ids:continue
  other=np.asarray(occ._geometries,dtype=object)[ids]
  distances=shapely.distance(polys[:,j,None],other[None,:]);best=distances.argmin(1);mins=distances[np.arange(n),best];mask=mins<closest
  closest[mask]=mins[mask];times[mask]=j;objects[mask]=np.asarray(occ.tokens,dtype=object)[np.asarray(ids)[best[mask]]]
  geometric_overlap|=shapely.intersects(polys[:,j,None],other[None,:]).any(1)
 dr=cache.drivable_area_map;ids=dr.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK,SemanticMapLayer.INTERSECTION,SemanticMapLayer.DRIVABLE_AREA,SemanticMapLayer.CARPARK_AREA])
 area=shapely.union_all(np.asarray(dr._geometries,dtype=object)[ids]);boundary=area.boundary
 road=np.full(n,np.inf);roadtime=np.full(n,-1);outside_any=np.zeros(n,bool)
 for i in range(n):
  for j in range(h):
   p=polys[i,j];outside=p.difference(area)
   if not outside.is_empty and outside.area>1e-10:
    pts=shapely.get_coordinates(outside);depth=max(float(shapely.distance(shapely.points(pts),boundary).max(initial=0)),outside.representative_point().distance(boundary))
    val=-depth;outside_any[i]=True
   else:val=p.distance(boundary)
   if val<road[i]:road[i]=val;roadtime[i]=j
 closest[~np.isfinite(closest)]=np.nan;road[~np.isfinite(road)]=np.nan
 return dict(object_clearance_m=closest,object_closest_time_s=times*.1,object_closest_token=objects,
  geometric_overlap=geometric_overlap,closest_object_ignored_initial_collision=np.asarray([t in cache.observation.collided_track_ids for t in objects]),
  road_margin_m=road,road_closest_time_s=roadtime*.1,footprint_outside_drivable=outside_any)

def parity(row,arrays):
 cache=load_cache(row);checks=[]
 for training in [False,True]:
  batch,diag,_=score(cache,arrays,training)
  fast,fdiag,_=score(cache,arrays,training,fast=True)
  singles=[]
  for tr in arrays:
   sc=PDMScorer(SAMPLING,PDMScorerConfig(progress_weight=10. if training else 5.,ttc_weight=5.,comfortable_weight=2.))
   r=pdm_score(cache,Trajectory(np.asarray(tr,dtype=np.float64)),SAMPLING,PDMSimulator(SAMPLING),sc)
   singles.append([getattr(r,n) for n in FIELDS])
  error=float(np.max(abs(batch-np.asarray(singles))));ferror=float(np.max(abs(batch-fast)))
  # Adding duplicated/different peers cannot change the first candidate's score.
  alone,ad,_=score(cache,arrays[:1],training);companions,cd,_=score(cache,np.concatenate([arrays,arrays[::-1]]),training)
  companion_error=max(float(abs(alone[0]-batch[0]).max()),float(abs(companions[:len(arrays)]-batch).max()))
  assert max(error,ferror,companion_error)<=cfg()['tolerances']['scalar_batch'],(row['token'],training,error,ferror,companion_error)
  assert np.max(abs(diag['ep_denominator']-cd['ep_denominator'][:len(arrays)]))<=1e-8
  expected=diag['gated_progress']/np.maximum(diag['ep_denominator'],1e-30);expected=np.where(diag['low_progress_branch'],(batch[:,0]*batch[:,1]>0).astype(float),expected)
  assert np.max(abs(expected-batch[:,2]))<=1e-8
  checks.append(dict(training_weights=training,scalar_batch_error=error,fast_standard_error=ferror,companion_error=companion_error,raw_progress_exact=True,baseline_raw_progress=float(diag['baseline_raw_progress'][0])))
 return checks
