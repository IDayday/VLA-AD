"""One-time V3 string serialization repair; all numerical measurements stay bit-identical."""
from common_v3 import *
def main():
    rows=[]
    for token in tokens('all'):
        path=OUT/'cache/learnability'/f'{token}.npz'
        # Only task-generated, local V3 arrays are read with pickle enabled.
        with np.load(path,allow_pickle=True) as a:
            if a['query_id'].dtype.kind!='O':continue
            oldsha=sha(path);meta=json.loads(str(a['metadata']));arrays={k:a[k] for k in a.files if k!='metadata'}
        numerical={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items() if k!='query_id'}
        assert all(isinstance(v,str) for v in arrays['query_id']);arrays['query_id']=arrays['query_id'].astype(str);npz(path,meta,**arrays)
        with np.load(path,allow_pickle=False) as a:
            assert all(hashlib.sha256(a[k].tobytes()).hexdigest()==h for k,h in numerical.items())
        rows.append(dict(token=token,old_sha256=oldsha,new_sha256=sha(path),all_numeric_arrays_bit_identical=True))
    save(OUT/'manifests/C_serialization_repair.json',dict(reason='pandas string column was exported as object array; convert only query_id to numpy unicode so pickle is unnecessary',repaired_files=len(rows),records=rows,rerun_inference=False))
if __name__=='__main__':main()
