"""Paired, class-stratified percentile bootstrap for fixed test predictions.

Intervals condition on the fitted models and observed class supports. They do
not include training/selection uncertainty or dependence among related Actions.
"""
import numpy as np
from scipy.stats import norm


def scores(y, pred, n_classes):
    cm = np.bincount(y * n_classes + pred, minlength=n_classes**2).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes, dtype=float), where=denom != 0)
    return np.array([tp.sum() / cm.sum(), f1.mean(), np.average(f1, weights=cm.sum(1))])


def paired_intervals(y_true, predictions, n_resamples=2000, seed=20260915):
    if n_resamples < 100:
        raise ValueError('At least 100 bootstrap replicates are required.')
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    if not len(labels):
        raise ValueError('Test labels cannot be empty.')
    encode = {label: i for i, label in enumerate(labels)}
    y = np.array([encode[v] for v in y_true])
    encoded = {}
    for name, values in predictions.items():
        if len(values) != len(y):
            raise ValueError('Prediction length must match test labels.')
        if any(v not in encode for v in values):
            raise ValueError('Every predicted class must occur in the test label universe.')
        encoded[name] = np.array([encode[v] for v in values])
    groups = [np.flatnonzero(y == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    draws = {name: np.empty((n_resamples, 3)) for name in encoded}
    for b in range(n_resamples):
        # The SAME sampled record indices are used for every model.
        sample = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for name, pred in encoded.items():
            draws[name][b] = scores(y[sample], pred[sample], len(labels))
    names = ['accuracy', 'macro_f1', 'weighted_f1']
    def pack(point, boot):
        ci = np.quantile(boot, [.025, .975], axis=0)
        return {name: {'estimate': float(point[j]), 'ci95': [float(ci[0,j]), float(ci[1,j])]}
                for j, name in enumerate(names)}
    point = {name: scores(y, pred, len(labels)) for name, pred in encoded.items()}
    models = {name: pack(point[name], draws[name]) for name in encoded}
    deltas = {f'{name}_minus_baseline': pack(point[name]-point['baseline'], draws[name]-draws['baseline'])
              for name in encoded if name != 'baseline' and 'baseline' in encoded}
    return {'method': 'paired class-stratified percentile bootstrap', 'confidence_level': .95,
            'n_resamples': n_resamples, 'seed': seed, 'n_test': len(y),
            'limitations': 'Conditional on fixed fitted models and observed class counts; assumes independent parameter records. Excludes retraining, selection uncertainty, shared-Action dependence and distribution shift.',
            'models': models, 'paired_differences': deltas}


def wilson_interval(successes, total, confidence=.95):
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError('Require 0 <= successes <= total and total > 0.')
    z = norm.ppf((1 + confidence)/2)
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [float(max(0, center-radius)), float(min(1, center+radius))]
