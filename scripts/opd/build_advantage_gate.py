import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.opd.common import *


def main():
    from navsim.agents.recogdrive.opd.gating import advantage_gate
    cfg=config();scenes={};counts={k:0 for k in 'ABCD'}
    for row in rows('train'):
        value=read(ROOT/'evaluation/train/E0/scores'/f"{row['token']}.json")
        gate=advantage_gate(value['scores']['IL']['scores'],value['scores']['RL']['scores'],high=cfg['high_threshold'],z=cfg['gate_z'],margin=cfg['gate_margin'])
        gate['effective_OPD_gate']=float(gate['gate'] and gate['category']=='A');scenes[row['token']]=gate;counts[gate['category']]+=1
    write(ROOT/'gates/static.json',dict(config_sha256=digest(cfg),scenes=scenes,category_counts=counts,
        effective_active=sum(g['effective_OPD_gate'] for g in scenes.values()),source='navtrain only, paired5 inference seeds',
        rule='A receives OPD only if positive LCB and per-seed NC/DAC/TTC nondegradation; B/C anchor; all receive GT; D receives no fixed-teacher/anchor guidance'))
    print(counts)

if __name__=='__main__':main()
