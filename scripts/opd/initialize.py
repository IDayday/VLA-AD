"""Initialize a fresh, isolated reproducibility root using verified real weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,xml.etree.ElementTree as ET
from scripts.opd.common import *

def main():
    import torch,yaml
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,read_state
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--dispatch-inputs',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);c=config();protocol=yaml.safe_load((SOURCE/'configs/opd/protocol.yaml').read_text())
    assert c['official_conditioning'] and protocol['version']==2
    from scripts.opd.splits import BUNDLE,verify_versions
    if (BUNDLE/'source_versions.json').exists():verify_versions()
    weights={}
    for label,key in [('IL','student_init'),('RL','teacher_init'),('COPY','teacher_init')]:
        head=build_head(c[key],device='cpu',frozen=True);value=sha(c[key]);expected=protocol['baseline']['RL' if label=='COPY' else label]['sha256']
        assert value==expected,(label,value,expected)
        weights[label]=dict(path=c[key],sha256=value,load_report=head._opd_load_report);del head
    value=sha(Path(c['vlm'])/'model.safetensors');assert value==protocol['baseline']['VLM']['sha256']
    weights['VLM']=dict(path=c['vlm'],sha256=value,deployment_shared=True,limitation='Head-only checkpoints cannot prove historical VLM identity')
    il={k.removeprefix('agent.action_head.'):v for k,v in read_state(c['student_init']).items()}
    rl={k.removeprefix('agent.action_head.'):v for k,v in read_state(c['teacher_init']).items()}
    active={k:v for k,v in rl.items() if not k.startswith('old_policy.')}
    reference={k.removeprefix('old_policy.'):v for k,v in rl.items() if k.startswith('old_policy.')}
    assert il.keys()==active.keys()==reference.keys()
    assert all(v.shape==active[k].shape and v.dtype==active[k].dtype for k,v in il.items())
    difference=[k for k,v in il.items() if not torch.equal(v,active[k])]
    ref_difference=[k for k,v in il.items() if not torch.equal(v,reference[k])]
    assert difference and not ref_difference
    write(ROOT/'audit/RL_reference_comparison.json',dict(passed=True,IL_active_tensors=len(il),RL_active_tensors=len(active),
        changed_active_tensors=len(difference),changed_names=difference,RL_old_policy_exactly_IL=True,
        missing=[],unexpected=[],shape_mismatch=[],teacher_source='Actual RL active head; never old_policy'))
    if a.verify:
        existing=read(ROOT/'audit/checkpoints.json');assert all(existing[k]['sha256']==v['sha256'] for k,v in weights.items())
        print('REAL_WEIGHT_IDENTITIES_VERIFIED',ROOT);return
    if (ROOT/'audit/checkpoints.json').exists():raise FileExistsError('Fresh initialization requires an unused OPD_RUN_ROOT')
    write(ROOT/'audit/checkpoints.json',weights);write(ROOT/'protocol.registration.json',protocol)
    write(ROOT/'calibration/registration.json',dict(status='LOCKED_BEFORE_CALIBRATION_RUN',
        rule='E3 tiny100; fixed16 hash-selected fit scenes in largest token-length bucket. Median OPD/aux gradient ratios times0.1(anchor)/0.05(GT), clips[.001,1]/[1e-6,.05]. Equal two LR slots1e-5/3e-5,100 updates each, cal-only selection.',
        formal_updates=1000,formal_train_seeds=[42,43,44],bootstrap_replicates=2000,primary_material_PDMS_gain=.002))
    write(ROOT/'analysis/case_rule_registration.json',dict(registered_before_formal_results=True,
        rule='Hash OPD_case_v1 first eligible, trainseed42, draw nearest scene mean E3-minus-IL, retain every nonempty category',
        thresholds=dict(repair=.05,regression=-.05,protected_E3=-.05,protected_E5=-.02,teacher_gap=.1,teacher_high=.8),source_sha256=sha(SOURCE/'scripts/opd/figures.py'),owner=identity()))
    subprocess.run([PYTHON,'scripts/opd/prepare_data.py','--manifest-only'],cwd=SOURCE,check=True)
    subprocess.run([PYTHON,'scripts/opd/inputs.py','manifest'],cwd=SOURCE,check=True)
    xml=ROOT/'tests/cpu.xml';xml.parent.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'tests/cpu.stdout','w') as log:
        subprocess.run([PYTHON,'-m','pytest','tests/opd','-q','--junitxml='+str(xml)],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
    element=ET.parse(xml).getroot();suites=list(element) if element.tag=='testsuites' else [element]
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert not counts['failures'] and not counts['errors'];write(ROOT/'tests/cpu.json',dict(passed=True,**counts,junit_sha256=sha(xml)))
    write(ROOT/'initialization.json',dict(owner=identity(),config_sha256=digest(c),protocol_registration_sha256=sha(ROOT/'protocol.registration.json'),
        status='INPUTS_REQUIRED',scope='Real strict weights and CPU tests; no G0/G1 or research-performance claim'))
    if a.dispatch_inputs:
        from scripts.opd.suite import run_queue
        run_queue([(f'official_inputs_{i}',[['scripts/opd/inputs.py','extract','--split','all','--rank',str(i),'--shards','32','--with-metrics']]) for i in range(32)])
    print('INITIALIZED',ROOT)

if __name__=='__main__':main()
