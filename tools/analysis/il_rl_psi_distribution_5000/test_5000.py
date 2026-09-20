"""Scientific contracts for the enlarged read-only experiment."""
from common_5000 import *
import unittest
import analyze as historical

class Contracts(unittest.TestCase):
    def test_cohort(self):
        r=scenes();ids={s['token'] for s in r}
        old={s['token'] for s in read(OLDOUT/'manifests/scenes.json')['scenes']}
        self.assertEqual(len(ids),5000);self.assertEqual(len(r),5000)
        self.assertEqual({s['token'] for s in r if s['cohort']=='OLD1000'},old)
        self.assertEqual(sum(s['cohort']=='NEW4000' for s in r),4000)
        self.assertFalse(read(OUT/'manifests/scenes.json')['outcome_conditioned'])
    def test_frozen_protocol(self):
        identity();f=read(OUT/'manifests/protocol.json')
        self.assertEqual(sha(OUT/'manifests/scenes.json'),f['scene_manifest_sha256'])
        for path,h in f['reused_code'].items():self.assertEqual(sha(path),h,path)
        self.assertEqual(CFG['optimizer_updates'],0)
    def test_seed_identity_and_independence(self):
        values=[seed(s['token'],g) for s in scenes() for g in range(4)]
        self.assertEqual(len(set(values)),20000)
        oldcfg=yaml.safe_load((MAIN/'configs/historical_sft_grpo_support/primary.yaml').read_text())
        self.assertEqual(CFG['seed'],oldcfg['seed'])
    def test_geometry_and_pdms(self):
        t=np.zeros((16,8,3));t[8:,:,0]=2;t[:,:,2]=100
        s=np.ones((16,7));s[0,0]=0;s[0,6]=0
        r=historical.group_stats(t,s,99)
        self.assertAlmostEqual(r['mean_PDMS'],93.75)
        self.assertAlmostEqual(r['feasible_rate'],15/16)
        self.assertAlmostEqual(r['pairwise_ADE'],128/120)
        self.assertEqual(r['min_PDMS'],0);self.assertEqual(r['max_PDMS'],100)
        self.assertAlmostEqual(distance(t.mean(0)[None],np.zeros((1,8,3)))[0,0],1)
    def test_missing_not_zero_and_empty_safety(self):
        s=np.zeros((16,7));r=historical.group_stats(np.zeros((16,8,3)),s,99)
        self.assertTrue(np.isnan(r['best_safe_PDMS']));self.assertTrue(r['no_safe_group'])
    def test_ddc_is_not_reward_multiplier(self):
        s=np.ones((1,7));s[0,5]=0
        self.assertEqual(historical.train_reward(s)[0],1)
        self.assertFalse(historical.safe(s)[0])
    def test_old_cache_is_resolved_without_copy(self):
        for m in CFG['primary_models']:
            self.assertTrue(str(bank_path(m,scenes()[0]['token'])).startswith(str(OLDOUT)))
    def test_scope_does_not_write_previous_results(self):
        self.assertNotEqual(OUT,OLDOUT)
        self.assertEqual(OUT.name,'il_rl_psi_distribution_5000')
        self.assertFalse(str(OUT).startswith(str(OLDOUT)))
    def test_bootstrap_keeps_scene_denominator(self):
        r=historical.bootstrap_difference(np.array([1.,1.,np.nan]),np.array(['a','b','c']),'unit_test')
        self.assertEqual(r['n'],2);self.assertEqual(r['mean_difference'],1)
        self.assertEqual(r['ci_low'],1);self.assertEqual(r['ci_high'],1)
    def test_supplementary_teacher_protocol_frozen(self):
        f=read(OUT/'manifests/teacher_coverage_protocol.json')
        self.assertEqual(f['config_sha256'],sha(WORK/'configs/il_rl_psi_distribution_5000/teacher_coverage.yaml'))
        self.assertEqual(f['scene_hash'],sha(OUT/'manifests/scenes.json'))

if __name__=='__main__':unittest.main()
