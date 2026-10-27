"""CPU-only checks for descriptive association and complete paired evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import correlation_analysis as c


class CorrelationTests(unittest.TestCase):
    def test_positive_and_negative_association(self):
        for method in ('pearson', 'spearman'):
            self.assertAlmostEqual(c.correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], method)['value'], 1.)
            self.assertAlmostEqual(c.correlation([1, 2, 3, 4, 5], [10, 8, 6, 4, 2], method)['value'], -1.)

    def test_spearman_average_ties(self):
        np.testing.assert_array_equal(c.average_ranks([4, 1, 1, 3]), [4., 1.5, 1.5, 3.])
        value = c.correlation([1, 1, 2, 3, 3], [3, 3, 2, 1, 1], 'spearman')
        self.assertAlmostEqual(value['value'], -1.)

    def test_constant_or_too_few_points_not_fabricated(self):
        self.assertIsNone(c.correlation([1, 1, 1], [90, 92, 94])['value'])
        self.assertIsNone(c.correlation([1, 2, 3], [90, 90, 90])['value'])
        self.assertIsNone(c.correlation([1, 2], [90, 95])['value'])
        with self.assertRaises(ValueError):
            c.correlation([1, float('nan'), 3], [90, 92, 94])

    def test_correlation_pairs_are_order_invariant(self):
        x, y = [4, 2, 1, 3, 5], [80, 90, 93, 82, 75]
        permutation = [3, 0, 4, 2, 1]
        for method in ('pearson', 'spearman'):
            a = c.correlation(x, y, method)['value']
            b = c.correlation([x[i] for i in permutation], [y[i] for i in permutation], method)['value']
            self.assertAlmostEqual(a, b)

    def test_bootstrap_preserves_paired_test_log_population(self):
        rows = [{name: float(5-i) for name in c.LOSS_COLUMNS} for i in range(5)]
        logs = [{k: [float((i+1)*count), 50.*count, count] for k, count in [('a', 2), ('b', 3)]} for i in range(5)]
        result = c.bootstrap_correlations(rows, logs, replicates=80)
        for col in c.LOSS_COLUMNS:
            for method in ('pearson', 'spearman'):
                ci = result['intervals'][col][method]['conditional_Test_log_95ci']
                np.testing.assert_allclose(ci, [-1., -1.], atol=1e-12)
        bad = copy.deepcopy(logs); bad[-1]['b'][2] += 1
        with self.assertRaisesRegex(ValueError, 'per-log scene population'):
            c.bootstrap_correlations(rows, bad, replicates=10)

    def test_epoch_mean_is_true_scene_weighted_and_checkpoint_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'epochs.json'
            rows = [{'epoch': e, 'status': 'COMPLETE', 'real_samples': 85813,
                     'observed_updates': 1341, 'nonfinite_gradients_reported': 0,
                     'missing_global_steps': [], 'metrics': {k:{'mean':v} for k,v in
                     [('loss',1/e),('h',1/e),('mu',0.)]},
                     'checkpoint': {'status':'PASS','checkpoint_sha256':str(e)}} for e in [1,4,8,12,16]]
            path.write_text(json.dumps({'epochs':rows}))
            actual=c.parse_epoch_statistics(path,[1,4,8,12,16])
            self.assertEqual(actual[16]['checkpoint_sha256'],'16')
            rows[-1]['real_samples']=85824;path.write_text(json.dumps({'epochs':rows}))
            with self.assertRaisesRegex(ValueError,'Incomplete actual training'):
                c.parse_epoch_statistics(path,[1,4,8,12,16])

    def test_real_native_verifier_and_recomputed_test_pdms(self):
        # Exercise original per-scene trajectory/seven-score verification with
        # a two-scene format fixture; production analyze is fixed at12146.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rows=[];protocol={'population':'test','scene_count':2,
                'noise_indices':[0,1,2,3,4],'checkpoint_sha256':'fixture', 'artifact_kind':'behavior',
                'mode':'student','target_version':None,'full_population':True,
                'tokens_sha256':c.digest(['t0','t1'])};p_sha=c.digest(protocol)
            keys=['score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
                  'time_to_collision_within_bound','comfort','driving_direction_compliance']
            for i in range(2):
                rows.append({'token':f't{i}','log_id':f'log{i}','protocol_sha256':p_sha,
                  'noise':[{'noise_index':j,'trajectory':[[1.,0.,0.]]*8,
                            'base_trajectory':[[0.,0.,0.]]*8,
                            'score':{k:(.8 if k=='score' else 1.) for k in keys},
                            'base_score':{k:(.7 if k=='score' else 1.) for k in keys}} for j in range(5)]})
            summary={'protocol_sha256':p_sha,'scenes':2,**{k:protocol[k] for k in
                     ['population','noise_indices','checkpoint_sha256','artifact_kind','mode','target_version','full_population']},
                     'behavior':{'scenes':2,'pdms_points':80.,'base_pdms_points':70.,'delta_points':10.}}
            def seal():
                (root/'PROTOCOL.json').write_text(json.dumps({**protocol,'protocol_sha256':p_sha}))
                (root/'SUMMARY.json').write_text(json.dumps(summary))
                (root/'scene_scores.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                (root/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','protocol_sha256':p_sha,
                    'artifacts':{name:c.sha256(root/name) for name in ['PROTOCOL.json','SUMMARY.json','scene_scores.jsonl']}}))
            seal();_,verify=c.native_tools();actual=verify(root,protocol)
            with patch.object(c,'TEST_SCENES',2):
                data=c.test_records(root,actual)
            self.assertAlmostEqual(data['test_pdms_points'],80.)
            self.assertEqual(data['tokens'],['t0','t1'])
            # Re-sealed hashes alone do not turn a missing noise into evidence.
            rows[-1]['noise'].pop();seal()
            with self.assertRaisesRegex(ValueError,'missing required behavior noise'):
                verify(root,protocol)

    def test_recomputed_summary_rejects_wrong_hashed_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);row={'token':'t0','log_id':'l',
                'noise':[{'noise_index':i,'score':{'score':.8},'base_score':{'score':.7},
                          'base_trajectory':[[0.,0.,0.]]*8} for i in range(5)]}
            (root/'scene_scores.jsonl').write_text(json.dumps(row)+'\n')
            summary={'behavior':{'pdms_points':90.,'base_pdms_points':70.,'delta_points':20.}}
            with patch.object(c,'TEST_SCENES',1), self.assertRaisesRegex(ValueError,'summary differs'):
                c.test_records(root,summary)

    def test_incomplete_inputs_do_not_emit_correlations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'request.json').write_text('{}')
            stats=[]
            for e in [1,4,8,12,16]:
                stats.append({'epoch':e,'status':'COMPLETE','real_samples':85813,
                  'observed_updates':1341,'nonfinite_gradients_reported':0,'missing_global_steps':[],
                  'metrics':{k:{'mean':v} for k,v in [('loss',1.),('h',1.),('mu',0.)]},
                  'checkpoint':{'status':'PASS','checkpoint_sha256':'missing'}})
            (root/'stats.json').write_text(json.dumps({'epochs':stats}))
            ref=lambda p:{'path':str(p),'sha256':c.sha256(p)}
            manifest={'schema':'nrkd_loss_pdms_correlation_input_v1','run_id':c.RUN_ID,
                'epochs':[1,4,8,12,16],'request':ref(root/'request.json'),
                'epoch_statistics':ref(root/'stats.json'),'results_root':str(root),
                'points':[{'epoch':e,'checkpoint':str(root/c.RUN_ID/'checkpoints'/f'epoch_{e:03d}.pt'),
                           'test_evaluation':str(root/'missing-test'),
                           'train_panel_representation':str(root/'missing-panel')} for e in [16,1,12,4,8]]}
            (root/'input.json').write_text(json.dumps(manifest))
            result=c.analyze(root/'input.json',root/'out',make_plot=False)
            self.assertEqual(result['status'],'INCOMPLETE');self.assertEqual(result['n_checkpoints'],0)
            self.assertIsNone(result['correlations']);self.assertIsNone(result['bootstrap'])
            self.assertEqual(len(result['incomplete']),5)
            self.assertFalse((root/'out/LOSS_VS_TEST_PDMS.png').exists())
            self.assertIn('没有计算或推断相关系数',(root/'out/LOSS_PDMS_CORRELATION_REPORT.md').read_text())


if __name__=='__main__':
    unittest.main()
