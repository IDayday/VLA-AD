"""Lossless columnar publication of larger tables; local CSVs remain untouched."""
from common_fd import *

def main():
    rows=[]
    for path in sorted((OUT/'metrics').glob('*.csv')):
        if path.stat().st_size<=3*1024**2:
            rows.append(dict(source=str(path.relative_to(ROOT)),published=str(path.relative_to(ROOT)),sha256=sha(path)))
            continue
        frame=pd.read_csv(path);dest=path.with_suffix('.parquet')
        frame.to_parquet(dest,index=False,compression='zstd')
        pd.testing.assert_frame_equal(frame,pd.read_parquet(dest))
        rows.append(dict(source=str(path.relative_to(ROOT)),published=str(dest.relative_to(ROOT)),source_sha256=sha(path),sha256=sha(dest),rows=len(frame),columns=list(frame.columns)))
    save(OUT/'manifests/metric_storage.json',rows)
    print('Published table formats',len(rows),flush=True)

if __name__=='__main__':main()
