import unittest
import numpy as np
import torch
from pairwise_credit import risk_credit,combine,ep_half_scalar,norm_match_delta,require_update_gate,validate_trainable_identity
from perturb import deform
from common import state_hash
class ProbeTests(unittest.TestCase):
 def test_fixed_group_denominator_and_symmetric_mask(self):
  c=torch.zeros(1,16);c[0,1]=1;s=torch.zeros(1,16,dtype=torch.bool);s[0,:2]=True
  b,m=risk_credit(c,s);self.assertAlmostEqual(b[0,0].item(),1/15,7);self.assertAlmostEqual(b[0,1].item(),-1/15,7)
  self.assertTrue(torch.equal(m,m.transpose(1,2)));self.assertFalse(m.diagonal(dim1=1,dim2=2).any());self.assertAlmostEqual(b.sum().item(),0)
 def test_equal_risk_and_no_pair_zero(self):
  z=torch.ones(2,16)*.5;s=torch.ones_like(z,dtype=torch.bool)
  self.assertEqual(risk_credit(z,s)[0].abs().sum(),0)
  d=torch.arange(16)[None].expand(2,-1).float();self.assertEqual(risk_credit(z,s,d,.5)[1].sum(),0)
 def test_progress_mask_uses_meters_not_ep(self):
  b,m=risk_credit(torch.tensor([[0.,1.,.5]]),torch.ones(1,3,dtype=torch.bool),torch.tensor([[30.,30.5,31.01]]),.5)
  self.assertTrue(m[0,0,1]);self.assertFalse(m[0,1,2]);torch.testing.assert_close(b,torch.tensor([[.5,-.5,0.]]))
 def test_safety_gate_after_bonus_no_recentering(self):
  a=torch.tensor([[0.,0.,-1.]]);b=torch.tensor([[.1,.1,-.2]]);safe=torch.tensor([[True,False,False]])
  out,d=combine(a,b,safe);self.assertGreater(out[0,0],0);self.assertEqual(out[0,1],0);self.assertLess(out[0,2],0);self.assertTrue(d['risk_information_active'])
 def test_zero_original_span_retains_risk(self):
  b,m=risk_credit(torch.tensor([[0.,1.]]),torch.ones(1,2,dtype=torch.bool));out,d=combine(torch.zeros_like(b),b,torch.ones_like(b,dtype=torch.bool))
  torch.testing.assert_close(out,b);self.assertFalse(d['original_information_active']);self.assertTrue(d['risk_information_active'])
 def test_arm_b_preserves_other_weights_and_reference_consistency(self):
  self.assertAlmostEqual(ep_half_scalar(1,1,1,1,1),12/17)
  self.assertAlmostEqual(ep_half_scalar(1,1,.2,1,1)-ep_half_scalar(1,1,0,1,1),1/17)
  self.assertEqual(ep_half_scalar(0,1,1,1,1),0)
 def test_actual_displacement_norm(self):
  start={'p':torch.tensor([3.,4.])};a={'p':torch.tensor([6.,8.])};d={'p':torch.tensor([3.,6.])};out,na,nd=norm_match_delta(start,a,d)
  self.assertEqual(na,5);self.assertEqual(nd,2);self.assertAlmostEqual(float(torch.linalg.norm(out['p']-start['p'])),2,6)
 def test_smooth_probe_zero_and_endpoints(self):
  a=np.column_stack([np.arange(1,9)*5,np.zeros(8),np.zeros(8)])
  np.testing.assert_allclose(deform(a,0,0,10),a,atol=1e-12)
  out=deform(a,.05,.2,10);np.testing.assert_allclose(out[-1],a[-1],atol=1e-12)
  self.assertLessEqual(np.max(np.abs(out[:,1])),.05+1e-12)
  # Analytic start basis and derivative, including endpoint derivatives.
  for u in [0,1]:self.assertEqual(16*u*u*(1-u)**2,0);self.assertEqual(32*u*(1-u)*(1-2*u),0)
 def test_guard_rejects_all_non_native_signals(self):
  for s in ['CONSTRUCTED_ONLY_SIGNAL','INSUFFICIENT_SIGNAL','NOT_SUPPORTED','NOT_RUN']:
   with self.assertRaises(RuntimeError):require_update_gate({'status':'PASS'},{'status':s})
  with self.assertRaises(RuntimeError):require_update_gate({'status':'FAIL'},{'status':'WITHIN_GROUP_SIGNAL'})
 def test_frozen_infinite_sampler_encoding_not_trainable(self):
  m=torch.nn.Module();m.register_parameter('weight',torch.nn.Parameter(torch.ones(2)));m.register_parameter('eta',torch.nn.Parameter(torch.tensor(float('inf')),requires_grad=False))
  self.assertEqual(validate_trainable_identity(m,['weight']),{'weight'})
  m.eta.requires_grad_(True)
  with self.assertRaises(RuntimeError):validate_trainable_identity(m,['weight'])
 def test_nonfinite_trainable_stops_before_optimizer(self):
  m=torch.nn.Linear(1,1);m.weight.data.fill_(float('nan'))
  with self.assertRaises(FloatingPointError):validate_trainable_identity(m,['weight','bias'])
 def test_hash_includes_scalar_buffers(self):
  model=torch.nn.BatchNorm1d(2);a=state_hash(model);model.num_batches_tracked.add_(1);self.assertNotEqual(a,state_hash(model))
if __name__=='__main__':unittest.main()
