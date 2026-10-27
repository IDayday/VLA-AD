"""Small CPU-only fixtures; production/GPU artifacts are never used."""
import copy,json,tempfile,unittest
from pathlib import Path
import supplemental_loss_metrics as s
import correlation_analysis as c


class SupplementalTests(unittest.TestCase):
    def fixture(self,root):
        epochs=[1,4,8,12,16];rows=[];statistics=[];points=[]
        (root/'request.json').write_text('{}')
        ref=lambda p:{'path':str(p),'sha256':c.sha256(p)}
        for e in epochs:
            panel=root/f'panel{e}';test=root/f'test{e}';panel.mkdir();test.mkdir()
            summary={'representation':{'all':{'scenes':8192},'keep':{'scenes':4000,
                'h_normalized':1/e,'predicted_rms':.1/e},'repair':{'scenes':4192,
                'h_normalized':2/e,'direction_cosine':1-1/(e+1),'amplitude_ratio':.7+e/100}}}
            (panel/'SUMMARY.json').write_text(json.dumps(summary));(panel/'COMPLETE.json').write_text('{}')
            (test/'SUMMARY.json').write_text('{}');(test/'COMPLETE.json').write_text('{}')
            rows.append({'epoch':e,'checkpoint_sha256':f'sha{e}','test_pdms_points':90+e/10,
                'train_panel_representation':str(panel),'test_evaluation':str(test),
                'panel_summary_sha256':c.sha256(panel/'SUMMARY.json'),'panel_complete_sha256':c.sha256(panel/'COMPLETE.json'),
                'test_summary_sha256':c.sha256(test/'SUMMARY.json'),'test_complete_sha256':c.sha256(test/'COMPLETE.json')})
            statistics.append({'epoch':e,'status':'COMPLETE','checkpoint':{'checkpoint_sha256':f'sha{e}'},
                'zero_target_h':{'weight':39240,'mean':1/e},'repair_target_h':{'weight':46573,'mean':2/e}})
            points.append({'epoch':e,'train_panel_representation':str(panel)})
        (root/'stats.json').write_text(json.dumps({'epochs':statistics}))
        manifest={'run_id':c.RUN_ID,'epochs':epochs,'points':points,'request':ref(root/'request.json'),
                  'epoch_statistics':ref(root/'stats.json')};(root/'input.json').write_text(json.dumps(manifest))
        analysis={'status':'COMPLETE','n_checkpoints':5,'rows':rows,'manifest':ref(root/'input.json')}
        (root/'analysis.json').write_text(json.dumps(analysis));return manifest,analysis

    def test_components_and_expected_correlation_sign(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);r=s.analyze(root/'input.json',root/'analysis.json',root/'out')
            self.assertEqual(len(r['rows']),5)
            self.assertAlmostEqual(r['correlations']['fixed_keep_h_normalized']['spearman']['value'],-1.)
            self.assertAlmostEqual(r['correlations']['fixed_repair_direction_cosine']['spearman']['value'],1.)
            self.assertAlmostEqual(r['rows'][0]['fixed_repair_mean_amplitude_deviation_from_one'],.29)
            self.assertTrue((root/'out/LOSS_COMPONENTS.csv').exists())

    def test_incomplete_primary_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,a=self.fixture(root);a['status']='INCOMPLETE';(root/'analysis.json').write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'complete primary'):
                s.analyze(root/'input.json',root/'analysis.json',root/'out')
            self.assertFalse((root/'out').exists())

    def test_changed_summary_or_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);(root/'panel1/SUMMARY.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                s.analyze(root/'input.json',root/'analysis.json',root/'out')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,_=self.fixture(root);m['extra']='changed';(root/'input.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'different frozen manifest'):
                s.analyze(root/'input.json',root/'analysis.json',root/'out')

    def test_missing_cosine_remains_undefined_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,a=self.fixture(root)
            for row in a['rows']:
                p=Path(row['train_panel_representation'])/'SUMMARY.json';d=json.loads(p.read_text());d['representation']['repair']['direction_cosine']=None;p.write_text(json.dumps(d));row['panel_summary_sha256']=c.sha256(p)
            (root/'analysis.json').write_text(json.dumps(a));r=s.analyze(root/'input.json',root/'analysis.json',root/'out')
            metric=r['correlations']['fixed_repair_direction_cosine'];self.assertEqual(metric['n_checkpoint_pairs'],0)
            self.assertIsNone(metric['pearson']['value'])


if __name__=='__main__':unittest.main()
