import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from metrics import *
class Tests(unittest.TestCase):
    def make(self,g=16):
        t=np.zeros((g,8,3));s=np.ones((g,7));return t,s
    def test_all_identical_is_not_diverse(self):
        t,s=self.make();v=group_stats(t,s)
        self.assertEqual(v['safety_pattern_count'],1);self.assertEqual(v['safety_pattern_disagreement'],0)
        self.assertEqual(v['mean_pairwise_ADE'],0);self.assertTrue(v['zero_advantage_group'])
        self.assertEqual(v['component_effective_rank'],0)
    def test_reward_formula_units(self):
        t,s=self.make();s[:,2]=.5
        np.testing.assert_allclose(reward(s),(5+5+2)/17);np.testing.assert_allclose(reward(s,False),9.5/12)
        s[:,0]=.5;np.testing.assert_allclose(reward(s),(12/17)*.5)
    def test_DDC_blind_reward(self):
        t,s=self.make();s[:8,5]=0
        v=group_stats(t,s);self.assertTrue(v['no_reward_contrast_but_outcome_difference'])
        self.assertTrue(v['zero_advantage_group']);self.assertAlmostEqual(v['feasible_rate'],.5)
        self.assertGreater(v['safety_pattern_disagreement'],0)
    def test_collision_mask(self):
        t,s=self.make();s[:,0]=0;s[:8,3]=0
        v=group_stats(t,s);self.assertEqual(v['zero_reward_masked_fraction'],1)
        self.assertTrue(v['zero_advantage_group']);self.assertTrue(np.isnan(v['safe_EP_headroom']))
        self.assertGreater(v['masked_pair_outcome_difference'],0)
    def test_safety_variation_not_safe_exploration(self):
        t,s=self.make();s[:8,1]=0;s[:,6]=reward(s,False)
        v=group_stats(t,s);self.assertGreater(v['safety_pattern_disagreement'],0)
        self.assertEqual(v['safe_EP_headroom'],0);self.assertFalse(v['safe_EP_opportunity'])
    def test_useful_progress(self):
        t,s=self.make();s[:,2]=np.linspace(.5,1,16);s[:,6]=reward(s,False)
        v=group_stats(t,s);self.assertTrue(v['safe_EP_opportunity']);self.assertAlmostEqual(v['safe_EP_headroom'],.25)
        self.assertEqual(v['safety_pattern_disagreement'],0);self.assertGreater(v['covariance_adv_EP'],0)
    def test_positive_unsafe_denominator(self):
        t,s=self.make(4);s[:,2]=[0,0,1,1];s[3,3]=0;s[:,6]=reward(s,False)
        v=group_stats(t,s);self.assertEqual(v['positive_infeasible_rate'],1)
        self.assertAlmostEqual(v['infeasible_among_positive'],.5)
    def test_dominance_and_effective_rank(self):
        t,s=self.make(4);s[:,2]=[.1,.2,.3,.4];s[:,6]=reward(s,False)
        v=group_stats(t,s);self.assertEqual(v['dominated_sample_fraction'],.75);self.assertAlmostEqual(v['component_effective_rank'],1)
    def test_geometry_can_widen_without_outcomes(self):
        t,s=self.make();t[:,:,0]=np.arange(16)[:,None]
        v=group_stats(t,s);self.assertGreater(v['mean_pairwise_ADE'],0);self.assertEqual(v['safety_pattern_disagreement'],0)
    def test_one_safe_headroom(self):
        t,s=self.make();s[1:,0]=0
        v=group_stats(t,s);self.assertEqual(v['safe_EP_headroom'],0);self.assertTrue(np.isnan(v['safe_EP_std']))
    def test_unbiased_disagreement(self):
        t,s=self.make(8);s[:4,3]=0;v=group_stats(t,s)
        self.assertAlmostEqual(v['safety_pattern_disagreement'],16/28)
    def test_zeros_not_dropped(self):
        t,s=self.make();s[:8,0]=0;s[:,6]=reward(s,False)
        self.assertEqual(group_stats(t,s)['mean_PDMS'],50)
if __name__=='__main__':unittest.main()
