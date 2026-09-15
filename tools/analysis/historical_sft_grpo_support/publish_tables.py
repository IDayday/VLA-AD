"""Lossless tabular publication without committing large repeated-text CSVs."""
from common_support import *

def main():
    manifest=[]
    for name in ['scene_metrics','teacher_scene_metrics','teachers']:
        source=OUT/'metrics'/f'{name}.csv';dest=source.with_suffix('.parquet')
        frame=pd.read_csv(source);frame.to_parquet(dest,index=False)
        pd.testing.assert_frame_equal(frame,pd.read_parquet(dest))
        manifest.append(dict(csv_path=str(source),csv_hash=sha(source),published_parquet_path=str(dest),parquet_hash=sha(dest),rows=len(frame),columns=list(frame.columns),roundtrip='exact pandas table equality',csv_retained_locally=True))
    save(OUT/'audits/published_tables.json',manifest)
    print('PUBLISHED TABLES',len(manifest),flush=True)

if __name__=='__main__':main()
