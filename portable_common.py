"""Private path adapter for the unmodified first-batch model/operator snapshots."""
from pathlib import Path
import os,sys,json,hashlib,time,random,datetime as dt
ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT/'vendor/upstream'
COMPONENTS=['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance']
sys.dont_write_bytecode=True
def retry(function):return function()
def read(path):
    path=Path(path);value=json.loads(path.read_text())
    if path.resolve()==(ROOT/'configs/repair_final_baseline.json').resolve():
        paths=json.loads(Path(os.environ['RECOGDRIVE_PORTABLE_PATHS']).read_text())
        value=dict(value,checkpoint=str(Path(paths['checkpoint']).expanduser().resolve()),vlm=str(Path(paths['vlm']).expanduser().resolve()))
    return value
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def setup(name):
    base=Path(os.environ.get('RECOGDRIVE_PORTABLE_WORKDIR',str(ROOT/'runtime')))/name;base.mkdir(parents=True,exist_ok=True)
    for k in ['HF_HOME','HF_MODULES_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR']:
        p=base/k.lower();p.mkdir(exist_ok=True);os.environ[k]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    config=os.environ.get('RECOGDRIVE_PORTABLE_PATHS')
    if config:
        paths=read(config)
        if paths.get('maps'):os.environ['NUPLAN_MAPS_ROOT']=paths['maps']
    os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0';sys.path.insert(0,str(UPSTREAM))
def seed(token,j=0):
    from seeds import scene_seed
    return scene_seed(token,j)
def reseed(value):
    from seeds import reseed as run
    return run(value)
def compatibility():
    import numpy as np
    if int(np.__version__.split('.')[0])<2:sys.modules.setdefault('numpy._core.numeric',np.core.numeric)
def quality(v):return 'failure' if v<.7 else 'high_quality' if v>.9 else 'intermediate'
