"""Private CPU fixtures only: no production imports or real training-log reads."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('epoch_stats',Path(__file__).with_name('epoch_loss_statistics_v2.py'))
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
RUN=M.RUNS[0]

def row(step,size=64,loss=1.,zero=20):
    return {'global_step':step,'real_scenes':size,'loss':loss,'h':loss/2,'mu':2.,
            'zero_count':zero,'repair_count':size-zero,'zero_h':1.,'repair_h':3.,
            'lr':1e-4,'grad_norm':2.,'grad_clip_ratio':.5,'cap_count':.25,'actual_rms':.01,
            'nonfinite_gradients':0,'transition_scene_count':size,'step_seconds':2.,
            'time':'2026-10-27T14:30:00+00:00','source_groups':{'0':[zero,zero],'1':[3*(size-zero),size-zero]},
            'mu_by_step':[[2*size,size],[0.,0],[0.,0],[0.,0],[0.,0]]}

def seal(folder,epoch=1,updates=3,scenes=181):
    folder.mkdir(parents=True,exist_ok=True);p=folder/f'epoch_{epoch:03d}.pt';p.write_bytes(b'private CPU checkpoint fixture')
    M.atomic_json(str(p)+'.COMPLETE.json',{'status':'COMPLETE','sha256':M.sha(p),'global_step':epoch*updates,
        'progress':{'global_step':epoch*updates,'epoch_index':epoch,'global_batch_cursor':0,'real_scenes_seen':epoch*scenes}})
    return p

class EpochStatisticsTests(unittest.TestCase):
    def test_only_four_retained_runs_cancelled_seed_rejected_before_log_read(self):
        self.assertEqual(len(M.RUNS),4)
        self.assertTrue(all(run.endswith('seed20261027') for run in M.RUNS))
        with patch.object(M,'load_records',side_effect=AssertionError('cancelled seed log read')):
            with self.assertRaises(ValueError):
                M.summarize_run('/unused','/unused','H_full_seed20261028')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'schedule.json'
            M.atomic_json(p,{'runs':{run:{'state':'ALL_40_EPOCHS_COMPLETE','next_read_utc':None} for run in M.RUNS}})
            self.assertEqual(M.run_due(p)['status'],'ALL_ACTIVE_RUNS_40_EPOCHS_COMPLETE')

    def test_tail_weighting_resume_latest_and_trailing_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path=root/'input.jsonl'
            values=[row(1,loss=1),row(2,loss=99),row(2,loss=2),row(3,53,loss=4,zero=10)]
            path.write_bytes((''.join(json.dumps(r)+'\n' for r in values)+'{"global_step":4').encode())
            records,source=M.load_records(path);cp=root/'checkpoints';seal(cp)
            value=M.summarize_records(RUN,records,cp,updates=3,scenes=181)[0]
            expected=(64+2*64+4*53)/181
            self.assertEqual(value['status'],'COMPLETE');self.assertEqual(value['real_samples'],181)
            self.assertAlmostEqual(value['metrics']['loss']['mean'],expected)
            self.assertAlmostEqual(value['metrics']['weighted_functional_contribution']['mean'],expected/2)
            self.assertEqual(value['zero_target_h']['weight'],50);self.assertEqual(value['zero_target_h']['mean'],1)
            self.assertEqual(value['repair_target_h']['weight'],131);self.assertEqual(value['repair_target_h']['mean'],3)
            self.assertEqual(value['source_groups']['1']['count'],131)
            self.assertEqual(value['mu_by_step'][0]['mu_mean'],2)
            self.assertEqual(value['mu_by_step'][0]['scheduled_scene_count'],181)
            self.assertEqual(source['duplicate_global_steps'],{'2':2});self.assertGreater(source['ignored_trailing_bytes'],0)
            variance=sum(w*(x-expected)**2 for x,w in [(1,64),(2,64),(4,53)])/181
            self.assertAlmostEqual(value['metrics']['loss']['std'],math.sqrt(variance))

    def test_missing_steps_bad_tail_nonfinite_and_seal_never_false_complete(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);seal(folder)
            missing={1:row(1),3:row(3,53,zero=10)}
            v=M.summarize_records(RUN,missing,folder,updates=3,scenes=181)[0]
            self.assertEqual(v['status'],'PARTIAL');self.assertEqual(v['missing_global_steps'],[2])
            bad={1:row(1),2:row(2),3:row(3,64)}
            v=M.summarize_records(RUN,bad,folder,updates=3,scenes=181)[0]
            self.assertEqual(v['status'],'PARTIAL');self.assertEqual(len(v['record_errors']),1)
            bad[3]=row(3,53,zero=10);bad[2]['loss']=float('nan')
            self.assertEqual(M.summarize_records(RUN,bad,folder,updates=3,scenes=181)[0]['status'],'PARTIAL')
            bad[2]=row(2);(folder/'epoch_001.pt').write_bytes(b'tampered')
            self.assertEqual(M.summarize_records(RUN,bad,folder,updates=3,scenes=181)[0]['status'],'INVALID_CHECKPOINT')

    def test_logs_complete_without_seal_pending_and_idempotent_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);log=root/RUN/'logs/train_metrics.jsonl';log.parent.mkdir(parents=True)
            log.write_text(''.join(json.dumps(r)+'\n' for r in [row(1),row(2),row(3,53,zero=10)]))
            out=root/'separate_sidecar_output'
            a,_=M.summarize_run(root,out,RUN,updates=3,scenes=181)
            self.assertEqual(a['epochs'][0]['status'],'LOGS_COMPLETE_CHECKPOINT_PENDING')
            p=out/RUN/'epoch_metrics.json';first=p.read_bytes()
            M.summarize_run(root,out,RUN,updates=3,scenes=181)
            self.assertEqual(first,p.read_bytes());self.assertTrue((out/RUN/'epoch_metrics.csv').is_file())

    def test_ramp_contribution_is_not_final_lambda_times_raw_mu(self):
        with tempfile.TemporaryDirectory() as t:
            a=row(5365,loss=1);a['h']=1;a['mu']=2
            b=row(5366,loss=2);b['h']=1;b['mu']=2
            v=M.summarize_records('H_mu_teacher_seed20261027',{5365:a,5366:b},Path(t))[0]
            self.assertEqual(v['metrics']['mu']['mean'],2)
            self.assertEqual(v['metrics']['weighted_functional_contribution']['mean'],.5)
            self.assertNotEqual(v['metrics']['weighted_functional_contribution']['mean'],.5*v['metrics']['mu']['mean'])
            self.assertEqual(v['phase_update_counts'],{'ramp':2})

    def test_due_run_only_minimum_interval_phase_slowdown_and_no_poll(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'schedule.json';now=M.utc('2026-10-27T15:00:00Z')
            due=M.RUNS[0];future=M.RUNS[1]
            state={'results_root':str(root),'out':str(root/'out'),'buffer_seconds':300,
                   'runs':{due:{'next_read_utc':now.isoformat(),'last_step':0},
                           future:{'next_read_utc':(now+M.timedelta(hours=3)).isoformat(),'last_step':0}}}
            M.atomic_json(p,state)
            summary={'run':due,'complete_epochs':0,'epochs':[{'status':'PARTIAL'}]}
            obs={'last_step':100,'last_time':now.isoformat(),'median_recent_step_seconds':2}
            def fake(results,out,run):
                M.atomic_json(Path(out)/run/'epoch_metrics.json',summary);return summary,obs
            with patch.object(M,'summarize_run',side_effect=fake) as call:
                v=M.run_due(p,now=now);self.assertEqual(v['processed_runs'],[due]);self.assertEqual(call.call_count,1)
                M.run_due(p,now=now+M.timedelta(minutes=1));self.assertEqual(call.call_count,1)
            current=M.read(p);self.assertGreaterEqual((M.utc(current['runs'][due]['next_read_utc'])-now).total_seconds(),600)
            entry={'last_step':5364}
            M.next_window(entry,{'last_step':5364,'median_recent_step_seconds':1},
                          {'run':'H_mu_student_seed20261027','complete_epochs':4,'epochs':[{'status':'COMPLETE'}]},now,300)
            self.assertEqual(entry['next_epoch'],5)
            self.assertGreaterEqual((M.utc(entry['next_read_utc'])-now).total_seconds(),1341*4+1800)
            M.next_window(entry,{'last_step':5400,'median_recent_step_seconds':1,
                                'median_recent_functional_step_seconds':2},
                          {'run':'H_mu_student_seed20261027','complete_epochs':4,'epochs':[{'status':'PARTIAL'}]},now,300)
            self.assertEqual((M.utc(entry['next_read_utc'])-now).total_seconds(),(6705-5400)*2+300)

    def test_40_epochs_stop_and_explicit_actual_terminal_status(self):
        entry={};now=M.utc('2026-10-27T15:00:00Z')
        M.next_window(entry,{'last_step':53640,'median_recent_step_seconds':1},
            {'run':RUN,'complete_epochs':40,'epochs':[{'epoch':i,'status':'COMPLETE'} for i in range(1,41)]},now,300)
        self.assertIsNone(entry['next_read_utc']);self.assertEqual(entry['state'],'ALL_40_EPOCHS_COMPLETE')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'state.json';terminal=root/'actual_status.json'
            M.atomic_json(terminal,{'status':'EXITED','returncode':1})
            M.atomic_json(p,{'results_root':str(root),'out':str(root/'out'),'buffer_seconds':300,
                            'runs':{RUN:{'next_read_utc':now.isoformat()}},'terminal_status_paths':{RUN:str(terminal)}})
            with patch.object(M,'summarize_run') as call:
                def fake(*args):
                    M.atomic_json(root/'out'/RUN/'epoch_metrics.json',{})
                    return {'run':RUN,'complete_epochs':0,'epochs':[]},{'last_step':0,'median_recent_step_seconds':None}
                call.side_effect=fake
                v=M.run_due(p,now=now)
            self.assertEqual(v['status'],'STOPPED_AT_ACTUAL_TERMINAL_EVENT_WITH_PARTIALS')
            self.assertEqual(M.read(p)['runs'][RUN]['returncode'],1)

    def test_watch_sleeps_once_to_explicit_deadline_then_exits_durably(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'watch_state.json';start=M.utc('2026-10-27T15:00:00Z');deadline=start+M.timedelta(hours=2)
            M.atomic_json(p,{'status':'SCHEDULED','runs':{RUN:{'next_read_utc':deadline.isoformat()}}})
            clock=[start];waits=[]
            actual_utc=M.utc
            def now(value=None):return clock[0] if value is None else actual_utc(value)
            def sleep(seconds):waits.append(seconds);clock[0]+=M.timedelta(seconds=seconds)
            def due(path):
                self.assertGreaterEqual(clock[0],deadline)
                M.atomic_json(path,{'status':'ALL_ACTIVE_RUNS_40_EPOCHS_COMPLETE','runs':{RUN:{'next_read_utc':None}}})
                return {'status':'ALL_ACTIVE_RUNS_40_EPOCHS_COMPLETE'}
            with patch.object(M,'utc',side_effect=now),patch.object(M.time,'sleep',side_effect=sleep),patch.object(M,'run_due',side_effect=due) as call:
                M.watch(p)
            self.assertEqual(waits,[7200]);self.assertEqual(call.call_count,1)
            self.assertEqual(M.read(str(p)+'.WATCH_EXIT.json')['returncode'],0)

if __name__=='__main__':unittest.main(verbosity=2)
