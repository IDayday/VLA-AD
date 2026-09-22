"""Read-only official-version and historical-run provenance; no model update."""
from pathlib import Path
import ast,dataclasses,datetime,hashlib,json,subprocess,sys,urllib.request
import yaml

ROOT=Path(__file__).resolve().parents[3]
ARCHIVE=Path('/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl')
CODE=ARCHIVE/'official_recogdrive'
REL='navsim/agents/recogdrive/recogdrive_diffusion_planner.py'
PIN='8a200077601ef33414469cfbe9fd095e322cccb6'
REVISIONS=['30212249d90bc6cd6248390365dd232d1c85c7d4','1bc43060d3d6af5ed03bf46cbd1ac227e7cb38b3','e3d5810a3529271c5a546c2532d29ae390bc650c',PIN]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ReCogDrive-reward-provenance-audit'})
    with urllib.request.urlopen(req,timeout=30) as f:return f.read()
def weights(source):
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GRPOConfig')
    calls=[n for n in ast.walk(cls) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='PDMScorerConfig']
    assert len(calls)==1
    return {k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
def main():
    history=json.loads(fetch('https://api.github.com/repos/xiaomi-research/recogdrive/commits?path=navsim%2Fagents%2Frecogdrive%2Frecogdrive_diffusion_planner.py&per_page=100'))
    dates={c['sha']:c['commit']['committer']['date'] for c in history};dates[PIN]='2026-03-13T04:10:21Z'
    versions=[]
    for revision in REVISIONS:
        url=f'https://raw.githubusercontent.com/xiaomi-research/recogdrive/{revision}/{REL}';source=fetch(url)
        versions.append(dict(commit=revision,date=dates[revision],url=url,sha256=hashlib.sha256(source).hexdigest(),weights=weights(source)))
    assert [r['weights']['progress_weight'] for r in versions]==[30,30,10,10]
    assert versions[-1]['sha256']==sha(CODE/REL)
    immutable=[]
    for rel in [REL,'navsim/agents/recogdrive/recogdrive_agent.py','navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py']:
        blob=subprocess.check_output(['git','show',PIN+':'+rel],cwd=CODE)
        assert blob==(CODE/rel).read_bytes();immutable.append(dict(path=str(CODE/rel),sha256=sha(CODE/rel),equals_upstream_commit=True))
    run=ARCHIVE/'outputs/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z'
    exp=run/'navsim_exp/training_recogdrive_agent'/run.name
    config_path=exp/'code/hydra/config.yaml';config=yaml.safe_load(config_path.read_text())
    overrides=yaml.safe_load((exp/'code/hydra/overrides.yaml').read_text())
    assert config['agent']['grpo'] and 'scorer' not in config
    assert not any('scorer' in k or 'reward' in k or 'progress_weight' in k for k in config['agent'])
    assert not any('scorer' in k or 'progress_weight' in k for k in overrides)
    sys.path.insert(0,str(CODE))
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig,PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    cfg=make_recogdrive_config('small',action_dim=3,action_horizon=8,input_embedding_dim=384,sampling_method='ddim',grpo=True)
    scorer=PDMScorer(TrajectorySampling(time_horizon=4,interval_length=.1),cfg.grpo_cfg.scorer_config)
    assert list(scorer._config.weighted_metrics_array)==[10.,5.,2.,0.]
    original=exp/'lightning_logs/version_0/checkpoints/epoch=8-step=11970.ckpt'
    watcher=ARCHIVE/'outputs/pdms_eval_stage3_ckpt_watcher_20260609T093852Z/ckpts/epoch8_step11970.ckpt'
    hashes=[sha(p) for p in [original,watcher]];assert hashes[0]==hashes[1]=='b51951abbba86e9661fc82f85406d09ae26c6eff6aa3651a40462363866fb326'
    result=dict(status='PASS',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),official_versions=versions,immutable_source_files=immutable,run_path=str(run),run_config_path=str(config_path),run_config_sha256=sha(config_path),run_agent=config['agent'],run_overrides=overrides,command_log_sha256=sha(run/'commands.log'),root_scorer_present=False,runtime_factory_grpo_weights=dataclasses.asdict(cfg.grpo_cfg.scorer_config),runtime_train_scorer_array=scorer._config.weighted_metrics_array.tolist(),default_scorer_weights=dataclasses.asdict(PDMScorerConfig()),checkpoint_paths=[str(original),str(watcher)],checkpoint_hashes=hashes,checkpoint_hyperparameters_note='Checkpoint lacks hyper_parameters and hparams.yaml is empty; attribution uses launcher, saved Hydra config, unchanged official agent/planner/scorer, and exact checkpoint identity. Not inferred from tensor values.',local_git_is_shallow=True,no_rollout_or_metric_changes=True,optimizer_updates=0,script_sha256=sha(__file__))
    dest=ROOT/'outputs/grpo_component_diversity/audits/reward_weight_provenance_20260922.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',official_EP_history=[30,30,10,10],runtime_training_EP=10,evaluation_EP=5,checkpoint_match=hashes[0])))
if __name__=='__main__':main()
