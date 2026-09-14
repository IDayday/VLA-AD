"""Tests use synthetic fixtures solely for accounting, never as experiment results."""
import unittest
import numpy as np
import pandas as pd
from analyze import METRICS, normalize_rows, boot_mean, compare, historical_summary_points

def fixture(scores):
    d=pd.DataFrame({'token':[f'{i:016x}' for i in range(len(scores))],**{m:np.ones(len(scores)) for m in METRICS}})
    d['PDMS']=scores
    return d

class AccountingTests(unittest.TestCase):
    def test_historical_watcher_named_PDMS_is_fraction(self):
        r={m:1.0 for m in METRICS};r['comfort']=r.pop('Comfort');r['PDMS']=.854536
        self.assertAlmostEqual(historical_summary_points(r)['PDMS'],85.4536)
    def test_units_and_genuine_zero(self):
        d=normalize_rows(fixture([.87,0]));self.assertEqual(d.PDMS.tolist(),[87,0]);self.assertEqual(d.zero_score.tolist(),[0,1])
    def test_error_is_not_zero(self):
        with self.assertRaises(AssertionError):normalize_rows(fixture([np.nan]))
    def test_duplicate_benchmark_token_rejected(self):
        f=fixture([.9,.8]);f['token']='0000000000000000'
        with self.assertRaises(AssertionError):normalize_rows(f)
    def test_already_scaled_input_rejected(self):
        with self.assertRaises(AssertionError):normalize_rows(fixture([90]))
    def test_cluster_bootstrap_constant_and_reproducible(self):
        a=boot_mean([2,2,2],['one','one','two'],3000,99);np.testing.assert_allclose(a,2)
        np.testing.assert_array_equal(boot_mean([1,3,8],['a','a','b'],3000,3),boot_mean([1,3,8],['a','a','b'],3000,3))
    def test_cluster_ratio_not_equal_cluster_means(self):
        # Unequal cluster sizes: interval must retain both extremes; exact point estimate stays sample weighted.
        a=normalize_rows(fixture([.5,.5,.5]));b=normalize_rows(fixture([.6,.6,.9]))
        t,z=compare(a,b,'test',pd.Series(['a','a','b'],index=a.index),'fixture')
        r=next(x for x in t if x['metric']=='PDMS');self.assertAlmostEqual(r['mean_difference'],20)
        self.assertEqual(r['n_scene_clusters'],2);self.assertLess(r['ci_low'],20);self.assertGreater(r['ci_high'],20)
    def test_safety_tail_exact_partition(self):
        a=normalize_rows(fixture([.9,.8]));f=fixture([0,.9]);f.loc[0,'NC']=0;b=normalize_rows(f)
        _,z=compare(a,b,'test',pd.Series(['a','b'],index=a.index),'fixture')
        self.assertEqual(z['new_zero_scores'],1);self.assertEqual(z['NC_regressions'],1)
        self.assertAlmostEqual(z['safety_regression_contribution'],-45)
        self.assertAlmostEqual(z['nonregression_contribution'],5);self.assertAlmostEqual(z['total_delta'],-40)
    def test_comparison_pairs_tokens_not_row_position(self):
        a=normalize_rows(fixture([.1,.5]));b=normalize_rows(fixture([.2,.6])).iloc[::-1]
        t,_=compare(a,b,'test',pd.Series(['a','b'],index=a.index),'fixture')
        self.assertAlmostEqual(t[0]['mean_difference'],10)

if __name__=='__main__':unittest.main()
