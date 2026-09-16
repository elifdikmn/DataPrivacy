import unittest
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from analysis.metrics import scores, paired_intervals, wilson_interval

class MetricsTests(unittest.TestCase):
    def test_metrics_match_reference_including_zero_predictions(self):
        y=np.array([0,0,1,1,2,2]);p=np.array([0,1,1,0,0,1])
        np.testing.assert_allclose(scores(y,p,3),[accuracy_score(y,p),f1_score(y,p,average='macro'),f1_score(y,p,average='weighted')])
    def test_pairing_identical_models_has_zero_difference(self):
        y=['a','a','b','b'];pred=['a','b','b','a']
        r=paired_intervals(y,{'baseline':pred,'same':pred},n_resamples=200)
        for m in r['paired_differences']['same_minus_baseline'].values():
            self.assertEqual(m['estimate'],0);self.assertEqual(m['ci95'],[0,0])
    def test_perfect_and_failed_models(self):
        r=paired_intervals(['a','a','b','b'],{'baseline':['b','b','a','a'],'perfect':['a','a','b','b']},n_resamples=200)
        self.assertEqual(r['models']['perfect']['accuracy']['ci95'],[1,1])
        self.assertEqual(r['paired_differences']['perfect_minus_baseline']['accuracy']['ci95'],[1,1])
    def test_wilson_matches_scipy_with_boundary_counts(self):
        for yes,n in [(0,5),(5,5),(12,16),(70,87)]:
            ci=binomtest(yes,n).proportion_ci(method='wilson')
            np.testing.assert_allclose(wilson_interval(yes,n),[ci.low,ci.high],atol=1e-14)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):paired_intervals(['a'],{'x':[]},n_resamples=200)
        with self.assertRaises(ValueError):wilson_interval(2,1)
