"""Bounded metadata search; exact scores and recipe names do not establish provenance."""
import subprocess
from v2_common import *
def main():
    roots=[ROOT/'outputs',ROOT/'docs',ROOT/'reports',Path('/mnt/project/VLA-AD_last_vla_dev/outputs'),Path('/mnt/project/VLA-AD_last_vla_dev/docs'),Path('/mnt/project/VLA-AD-worktrees'),Path('/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl/outputs')]
    cmd=['rg','--files','--no-ignore','-g','*.md','-g','train_args.json','-g','config.yaml','-g','manifest.json','-g','*summary*.csv','-g','*summary*.json','-g','!**/metric_cache*/**','-g','!**/features/**','-g','!**/deps/**','-g','!**/site-packages/**','-g','!**/.git/**','-g','!**/reproduction_diagnostics/**']+[str(r) for r in roots if r.exists()]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0;paths=r.stdout.splitlines();terms=['85.8','86.7','90.3','score_mts','pareto_mts','gt_distance','stage3','grpo'];matches=[];inspected=skipped=0
    for name in paths:
        p=Path(name)
        if str(p).startswith(str(OUT)):continue
        if p.stat().st_size>8_000_000:skipped+=1;continue
        text=p.read_text(errors='replace');inspected+=1;hits={term:[] for term in terms}
        for line_number,line in enumerate(text.splitlines(),1):
            for term in terms:
                if term.lower() in line.lower() and len(hits[term])<3:hits[term].append(dict(line=line_number,text=line[:600]))
        if any(hits.values()):matches.append(dict(path=str(p),sha256=sha(p),matches={k:v for k,v in hits.items() if v}))
    save(OUT/'manifests/historical_search_audit.json',dict(roots=list(map(str,roots)),command=cmd,terms=terms,metadata_inventory=len(paths),inspected=inspected,skipped_over8MB=skipped,matches=matches,limitations='Metadata and reports only; trajectory caches and installed dependencies excluded. Filename or approximate score alone is not accepted as a causal-chain match.'))
    print('Historical metadata',len(paths),'inspected',inspected,'matches',len(matches))
if __name__=='__main__':main()
