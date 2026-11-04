"""Sanity checks: repeatable metrics and leakage-safe statistics."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.analysis import RAW, linear_experiment, logistic_experiment, ZERO_INVALID_COLS

class TestLab04(unittest.TestCase):
    def test_housing_shapes_and_scaling(self):
        a = linear_experiment()
        self.assertEqual((a['rows'], a['columns'], a['train_size'], a['test_size']), (545, 13, 436, 109))
        self.assertEqual(a['feature_count_encoded'], 13)
        self.assertGreater(a['r2'], 0.5)
        self.assertLess(a['r2'], 1.0)
        self.assertAlmostEqual(a['mae'], 970043.4039, places=1)

    def test_diabetes_stratification_and_imputation(self):
        a = logistic_experiment()
        self.assertEqual((a['train_size'], a['test_size']), (614, 154))
        self.assertEqual((a['train_positive'], a['test_positive']), (214, 54))
        self.assertEqual(a['tp'] + a['tn'] + a['fp'] + a['fn'], 154)
        self.assertTrue(all(v > 0 for v in a['train_medians'].values()))
        data = pd.read_csv(RAW/'diabetes.csv')
        X, y = data.drop(columns='Outcome'), data['Outcome']
        tr, te, _, _ = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        for col in ZERO_INVALID_COLS:
            self.assertEqual(float(tr.loc[tr[col] != 0, col].median()), a['train_medians'][col])
        self.assertAlmostEqual(a['roc_auc'], 0.81296296296)

    def test_thresholds_monotonic_counts(self):
        rows = logistic_experiment()['thresholds']
        self.assertEqual([r['threshold'] for r in rows], [.3,.4,.5,.6])
        self.assertEqual([r['FN'] for r in rows], [11,19,27,29])
        self.assertEqual([r['FP'] for r in rows], [29,22,18,14])

if __name__ == '__main__':
    unittest.main()
