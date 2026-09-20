import unittest
from common_matched import *
from candidates import front_ranks,select_arrays,psi_kernel
class MatchedTests(unittest.TestCase):
    def test_split(self):
        tr=scenes('train');ev=scenes('holdout')
        self.assertEqual(len(tr),3072);self.assertEqual(len(ev),5000)
        self.assertFalse({s['log'] for s in tr}&{s['log'] for s in ev})
    def test_frozen(self):identity()
    def test_real_archive_identity(self):
        import pickle,lzma
        for s in scenes()[:4]:self.assertEqual(pickle.load(lzma.open(archive(s['token']),'rb'))['token'],s['token'])
    def test_fronts(self):
        v=np.array([[1,1,1],[.5,1,1],[1,.5,1],[.2,.2,.2]])
        np.testing.assert_array_equal(front_ranks(v,np.ones(4,bool)),[0,1,1,2])
    def test_target_weighting(self):
        m,_=psi_kernel()
        for ids in [[0],[0,1],[1,0,2],[1,2,3]]:
            w=m._support_weights(ids,0,best_weight=.5,gt_weight=.2,other_weight=.3,dtype=__import__('torch').float32)
            self.assertAlmostEqual(float(w.sum()),1);self.assertTrue((w[len(ids):]==0).all())
    def test_no_score_conditioned_split(self):
        self.assertFalse(CFG['outcome_conditioned_selection'])
        self.assertEqual(CFG['optimizer_updates_GRPO'],0)
    def test_real_selections(self):
        if not (OUT/'manifests/selection_frozen.json').exists():self.skipTest('selection generation still running')
        for s in scenes()[::50]:
            data=read(OUT/'cache/selection'/f"{s['token']}.json")
            for v in data['methods'].values():
                self.assertEqual(len(v['indices']),len(set(v['indices'])));self.assertAlmostEqual(sum(v['weights']),1,places=6)
                self.assertGreater(len(v['indices']),0);self.assertLessEqual(len(v['indices']),3)
    def test_seed_independence(self):
        vals=[seed(r,'training_noise',s,m) for r in CFG['train_seeds'] for s in range(512) for m in range(8)]
        self.assertEqual(len(vals),len(set(vals)))
if __name__=='__main__':unittest.main()
