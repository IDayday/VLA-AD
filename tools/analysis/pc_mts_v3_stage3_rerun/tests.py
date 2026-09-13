"""Protocol and dataset safety checks, independent of training outcomes."""
from common_r import *
import unittest,torch
from torch.utils.data import DistributedSampler
class Contracts(unittest.TestCase):
    def test_frozen_protocol_and_formal_sources(self):
        identity()
        for p,h in read(OUT/'manifests/frozen.json')['source_sha256'].items():self.assertEqual(sha(ROOT/p),h)
    def test_no_holdout_in_distributed_updates(self):
        tr=tokens('train');te=set(tokens('holdout'));self.assertFalse(set(tr)&te)
        for sd in CFG['seeds']:
            for epoch in range(10):
                draws=[]
                for rank in range(8):
                    s=DistributedSampler(tr,8,rank,shuffle=True,seed=sd,drop_last=False);s.set_epoch(epoch)
                    indices=list(s);self.assertEqual(len(indices),88);draws.extend(tr[i] for i in indices)
                self.assertEqual(len(draws),704);self.assertEqual(set(draws),set(tr));self.assertFalse(set(draws)&te)
    def test_all_initializations_are_tensor_identical(self):
        for row in read(OUT/'manifests/initializations.json'):
            a=state_only(row['source']);b=state_only(row['path']);self.assertEqual(set(a),set(b))
            for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
    def test_write_boundary(self):
        for old in [V1,V2,V3]:
            with self.assertRaises(AssertionError):guard(old/'forbidden.json')
    def test_scorer_never_reads_uncommitted_or_unknown_token(self):
        import tempfile
        from score_r import completed_inputs
        with tempfile.TemporaryDirectory(dir=OUT) as td:
            p=Path(td)/'run/step';p.mkdir(parents=True)
            for name in ['known.npz','known.123.tmp.npz','unknown.npz']:(p/name).write_bytes(b'not read by discovery')
            self.assertEqual(completed_inputs(td,{'known':{}}),[p/'known.npz'])
if __name__=='__main__':unittest.main()
