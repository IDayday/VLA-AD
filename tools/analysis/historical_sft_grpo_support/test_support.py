import unittest
from common_support import *
from analyze import group_stats,train_reward,representatives,bootstrap_difference,safe

class SupportTests(unittest.TestCase):
    def test_group_seeds_are_disjoint(self):
        values=[seed(s['token'],g) for s in scenes() for g in range(4)]
        self.assertEqual(len(set(values)),4000)
        fitting=[seed(s['token'],g,'teacher_fitting') for s in scenes() for g in range(4)]
        self.assertFalse(set(values)&set(fitting))
    def test_exact_16_group_stats_and_unsafe_tail(self):
        t=np.zeros((16,8,3));t[:, :,0]=np.arange(16)[:,None]
        s=np.ones((16,7));s[0,0]=0;s[0,6]=0
        r=group_stats(t,s,99)
        self.assertEqual(r['min_PDMS'],0);self.assertEqual(r['max_PDMS'],100)
        self.assertEqual(r['mean_PDMS'],93.75);self.assertEqual(r['safe_count'],15)
        self.assertEqual(r['feasible_rate'],15/16);self.assertTrue(r['Hit16'])
    def test_no_safe_best_is_na_not_zero(self):
        s=np.ones((16,7));s[:,0]=0;s[:,6]=0
        r=group_stats(np.zeros((16,8,3)),s,99)
        self.assertTrue(np.isnan(r['best_safe_PDMS']));self.assertTrue(r['no_safe_group'])
    def test_heading_not_used_as_xy_distance(self):
        a=np.zeros((1,8,3));b=a.copy();b[...,2]=3
        self.assertEqual(distance(a,b)[0,0],0)
        b[...,0]=.5;self.assertEqual(distance(a,b)[0,0],.5)
    def test_natural_duplicate_rollouts_keep_frequency(self):
        t=np.zeros((16,8,3));s=np.ones((16,7));s[:8,6]=.9
        self.assertEqual(group_stats(t,s,99)['HQ_mass'],.5)
        self.assertEqual(representatives(t),1)
    def test_training_reward_is_not_standard_pdms(self):
        s=np.ones((2,7));s[:,2]=0;s[:,6]=7/12;s[1,0]=0
        self.assertAlmostEqual(train_reward(s)[0],7/17);self.assertEqual(train_reward(s)[1],0)
    def test_diagnostic_ddc_does_not_multiply_native_v1_reward(self):
        s=np.ones((1,7));s[:,5]=0
        self.assertEqual(train_reward(s)[0],1.)
        self.assertFalse(safe(s)[0])
    def test_joint_safety_requires_all_four(self):
        s=np.ones((4,7))
        for j,i in enumerate([0,1,3,5]):s[j,i]=.5
        self.assertFalse(safe(s).any())
    def test_frozen_configuration(self):
        f=read(OUT/'manifests/protocol.json')
        self.assertEqual(sha(CFG_PATH),f['config_sha256'])
        self.assertEqual(sha(OUT/'manifests/models.json'),f['models_sha256'])
        self.assertEqual(sha(OUT/'manifests/scenes.json'),f['scene_sha256'])
    def test_training_membership_never_relabelled(self):
        d=pd.read_csv(OUT/'metrics/training_membership.csv')
        self.assertEqual(len(d),1000);self.assertEqual(d.common_train.sum(),835)
        self.assertEqual(d.psi_train.sum(),835);self.assertTrue(d.a5_train.all() and d.v6_train.all())
    def test_teacher_rows_resolve_actual_tensor(self):
        for s in scenes()[:8]:
            z=np.load(OUT/'cache/teachers'/f"{s['token']}.npz")
            rows=json.loads(str(z['metadata']))['rows'];self.assertEqual(len(rows),len(z['trajectories']))
            for j,r in enumerate(rows):self.assertEqual(digest(z['trajectories'][j].tolist()),r['trajectory_hash'])
    def test_full_native_entry_parity_and_no_update(self):
        for m in CFG['primary_models']:
            a=read(OUT/'audits'/f'sampling_{m}.json');self.assertEqual(a['status'],'PASS')
            self.assertEqual(a['state_before'],a['state_after']);self.assertEqual(a['optimizer_updates'],0)
            for r in a['checks']:self.assertEqual(r['native_entry_vs_extraction_max_abs'],0)
    def test_bootstrap_pairs_not_pooled_teachers(self):
        r=bootstrap_difference([2,2,2,2],['a','a','b','b'],'test')
        self.assertEqual(r['mean_difference'],2);self.assertEqual(r['ci_low'],2);self.assertEqual(r['log_count'],2)

if __name__=='__main__':unittest.main()
