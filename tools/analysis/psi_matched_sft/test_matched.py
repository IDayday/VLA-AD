import unittest
from common_matched import *
from candidates import front_ranks,select_arrays,psi_kernel
class MatchedTests(unittest.TestCase):
    def test_execution_shards(self):
        for n in [1000,5512]:
            shards=[execution_shard(list(range(n)),r,4) for r in range(4)]
            merged=[x for rows in shards for x in rows]
            self.assertEqual(sorted(merged),list(range(n)))
            self.assertEqual(len(merged),len(set(merged)))
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
    def test_native_loss_gradient_smoke(self):
        a=read(OUT/'audits/training_smoke.json')
        self.assertEqual(a['trainable'],34329219);self.assertEqual(a['optimizer_updates'],0)
        self.assertLess(a['abs_error'],1e-7);self.assertTrue(np.isfinite(a['gradient_norm']))
    def test_scalar_batch_evaluator(self):
        a=read(OUT/'audits/initial.json')
        self.assertEqual(len(a['scoring_parity']),4)
        self.assertTrue(all(x['max_abs_error']<=1e-8 for x in a['scoring_parity']))
    def test_sampler_batching_preserves_stream(self):
        a=read(OUT/'audits/multiscene_smoke.json')
        self.assertEqual(len(a['checks']),8)
        self.assertTrue(all(x['max_abs_error']<1e-4 for x in a['checks']))
    def test_target_tables_reconstruct(self):
        p=OUT/'metrics/selected_targets.parquet'
        if not p.exists():self.skipTest('target publication still running')
        f=pd.read_parquet(p)
        for r in f.iloc[::1000].to_dict('records'):
            a=np.asarray(r['trajectory'],dtype=np.float32).reshape(8,3)
            self.assertEqual(digest(a.tolist()),r['trajectory_sha256'])
            raw=np.load(OUT/'cache/raw'/f"{r['token']}.npz")['trajectories'][r['raw_index']]
            np.testing.assert_array_equal(a,raw)
if __name__=='__main__':unittest.main()
