"""Meaningful gate-order, matching, weights and scoring parity checks."""
import unittest
from common_v3 import *
from static import ranks, dominators, conditional
from matching import match

class Contracts(unittest.TestCase):
    def test_far_dominator_removed_before_conditional_ranking(self):
        s=np.ones((4,7));s[:,2]=[1,.9,.8,.7];s[:,6]=[.99,.98,.97,.96]
        q=np.array([100,100,75,30]);e=q<95
        self.assertEqual(ranks(s).tolist(),[1,2,3,4]);self.assertEqual(ranks(s,e).tolist(),[-1,-1,1,2])
        self.assertTrue(dominators(s)[2,0])
    def test_conditional_strict_and_secondary_no_duplicate_or_far(self):
        s=np.ones((5,7));s[:,6]=[1,.99,.985,1,.98];q=np.array([60,70,80,100,20]);traj=np.zeros((5,8,3));traj[:, :, 0]=np.arange(5)[:,None];local=np.ones(5)
        ids,levels,e,r=conditional(s,traj,q,local,99,np.ones(5,dtype=bool))
        self.assertEqual(e.tolist(),[True,True,False,False,False]);self.assertEqual(set(ids),{0,1,2,4});self.assertEqual(len(ids),len(set(ids)));self.assertEqual(levels,[0,0,1,1])
    def test_matching_max_cardinality_and_primary_gap(self):
        a=pd.DataFrame({'PDMS':[90,90.2]},index=[0,1]);b=pd.DataFrame({'PDMS':[90.15,90.4]},index=[2,3])
        pairs=match(a,b,.25);self.assertEqual(len(pairs),2);self.assertEqual(len({j for _,j,_ in pairs}),2);self.assertTrue(all(g<=.25+1e-8 for _,_,g in pairs))
    def test_split_and_frozen_config(self):
        identity();tr=tokens('train');ho=tokens('holdout');self.assertEqual(len(tr),700);self.assertEqual(len(ho),300);self.assertFalse(set(tr)&set(ho));self.assertEqual(len(tokens('gradient')),256)
    def test_v3_write_boundary(self):
        with self.assertRaises(AssertionError):guard(V2/'metrics/no_write.json')
    def test_repeated_slots_preserve_unique_parent_weight(self):
        from train import parent_slots
        slots,w,fallback=parent_slots([3,8,17], 'test',1701,10,0)
        self.assertEqual(len(slots),16);self.assertFalse(fallback)
        for parent in [3,8,17]:self.assertAlmostEqual(float(w[slots==parent].sum()),1/3,places=6)
        slots,w,fallback=parent_slots([], 'test',1701,10,0);self.assertTrue(fallback);self.assertEqual(set(slots),{0});self.assertAlmostEqual(float(w.sum()),1,places=6)
    def test_ranking_native_parity(self):
        from select_candidate_pools import pareto_ranks
        for r in scenes()[:4]:
            s=np.load(V2/'evaluator/raw_candidates'/f"{r['token']}.npz")['trajectories'];np.testing.assert_array_equal(ranks(s),pareto_ranks(s))
    def test_native_evaluator_cached_four_scene_parity(self):
        from evaluate_cached_rollouts import init_worker, parity, score_arrays
        import lzma,pickle
        init_worker();checks=[]
        for r in scenes()[:4]:
            t=r['token'];a=np.load(V1/'rollouts/official_il'/f'{t}.npz')['reference'];checks.append(parity(r,a))
            with lzma.open(r['metric_cache_path'],'rb') as f:c=pickle.load(f)
            actual=score_arrays(c,a);saved=np.load(V1/'evaluator/rollouts/official_il'/f'{t}.npz')['reference'];self.assertLessEqual(abs(actual-saved).max(),1e-8)
        save(OUT/'manifests/scorer_parity.json',checks)
if __name__=='__main__':unittest.main(verbosity=2)
