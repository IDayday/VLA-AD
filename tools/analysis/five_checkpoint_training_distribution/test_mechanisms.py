"""Tests for the nontrivial geometric intervention and native target estimator."""
import unittest
import numpy as np
import torch
from common_fd import *
from teacher_audit import native_helpers,native_config,exact_legacy_sampling_expectation,weights_for

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(3206)
        self.a=self.rng.normal(size=(64,8,3));self.a[...,2]*=.1
        self.b=self.rng.normal(size=(64,8,3));self.b[...,2]*=.1
    def test_reconstruction_and_wrap(self):
        self.a[...,2]=np.pi+self.a[...,2];self.a[...,2]=wrap(self.a[...,2])
        np.testing.assert_allclose(combine(center(self.a),residual(self.a)),self.a,atol=1e-12)
        self.assertTrue(np.all(np.abs(combine(center(self.a),residual(self.b))[...,2])<=np.pi))
    def test_center_swap_preserves_within_scene_xy_distances(self):
        z=combine(center(self.b),residual(self.a))
        np.testing.assert_allclose(distance(z,z),distance(self.a,self.a),atol=1e-12)
        np.testing.assert_allclose(center(z)[:,:2],center(self.b)[:,:2],atol=1e-12)
    def test_shapley_additivity(self):
        q=self.rng.normal(size=(4,1000));a,b=shapley(*q)
        np.testing.assert_allclose(a+b,q[3]-q[0],atol=1e-12)
    def test_width_control_exact_spread(self):
        bank={m:self.a.copy() for m in MODELS};bank['grpo_9041']=self.b*3
        z=make_counterfactuals(bank)['official_il__grpo_width_only']
        self.assertAlmostEqual(spread(z),spread(bank['grpo_9041']),places=12)
        np.testing.assert_allclose(center(z)[:,:2],center(self.a)[:,:2],atol=1e-12)
        np.testing.assert_equal(z[...,2],self.a[...,2])
    def test_scores_and_empty_quality(self):
        s=np.ones((64,7));s[:,6]=.93
        self.assertAlmostEqual(score_metrics(s,94)['PDMS'],93)
        self.assertEqual(score_metrics(s,94)['Hit8'],0)
        self.assertEqual(score_metrics(s,92)['Hit8'],1)

class TeacherTests(unittest.TestCase):
    def test_exact_sampler_expectation_against_native_sampling(self):
        n=native_helpers();cfg=native_config('mts_8751');B=6000
        w=np.array([.6,.12,.1,.08,.1]);code=np.array([1,7,4,5,7]);valid=np.ones(5,dtype=bool);reward=np.array([.8,.85,.88,.9,.99])
        expected=exact_legacy_sampling_expectation(w,code,valid,reward,cfg)
        t=torch.zeros((B,5,8,3));t[:,:,:,0]=torch.arange(5)[None,:,None]
        weights=torch.tensor(w,dtype=torch.float32)[None].expand(B,-1);real=torch.ones((B,5),dtype=torch.bool)
        torch.manual_seed(7739)
        traj,sw,mask,_=n._sample_asmi_targets(t,weights,real,real,torch.tensor(code)[None].expand(B,-1),torch.tensor(reward)[None].expand(B,-1),cfg,155,True)
        observed=np.bincount(traj[:,:,0,0].numpy().astype(int).flatten(),weights=sw.numpy().flatten(),minlength=5)/B
        np.testing.assert_allclose(expected,observed,atol=.008)
        self.assertAlmostEqual(expected.sum(),1)
    def test_residual_budget_caps_proxy_not_nominal_mode_count(self):
        n=native_helpers();loss=torch.tensor([[.01,.25]]);w=torch.tensor([[.5,.5]]);source=torch.tensor([[1,7]])
        adjusted,_=n._apply_dpsi_residual_budget(loss,w,source,enabled=True,non_gt_residual_mass_cap=.35)
        proxy=adjusted*loss.sqrt()
        self.assertAlmostEqual(float(proxy[0,1]/proxy.sum()),.35,places=6)
        self.assertAlmostEqual(float(adjusted.sum()),1,places=6)
        self.assertLess(float(adjusted[0,1]),.35)
    def test_actual_record_gt_identity_and_probability_mass(self):
        for m in ARCHIVES:
            p,r,idx,t=teacher_record(m,SCENES[0]['token']);w=weights_for(m,r,idx,t)
            self.assertAlmostEqual(w['expected'].sum(),1,places=6)
            gt=w['codes']==1
            np.testing.assert_allclose(t[gt][0],np.asarray(SCENES[0]['gt']),atol=2e-6)
    def test_frozen_protocol_and_scene_identity(self):
        self.assertEqual(len(SCENES),1000);self.assertEqual(len({s['token'] for s in SCENES}),1000)
        self.assertEqual(config_identity(),sha(CONFIG))
        self.assertEqual(CFG['optimizer_updates'],0)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
