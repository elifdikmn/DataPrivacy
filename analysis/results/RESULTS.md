# Corrected analysis results

|Model|Accuracy (95% CI)|Macro F1 (95% CI)|
|---|---|---|
|baseline|68.94% [67.30%, 70.66%]|0.4676 [0.4278, 0.4963]|
|word_char_balanced|76.20% [74.56%, 77.80%]|0.6423 [0.6084, 0.6777]|

## Paired improvement intervals
{
  "word_char_balanced_minus_baseline": {
    "accuracy": {
      "estimate": 0.0725712056184159,
      "ci95": [
        0.057354662504877085,
        0.08818767069840028
      ]
    },
    "macro_f1": {
      "estimate": 0.17467903563012183,
      "ci95": [
        0.13580938893613587,
        0.22313039608379945
      ]
    },
    "weighted_f1": {
      "estimate": 0.07750469516725922,
      "ci95": [
        0.06318335855141768,
        0.09264675559787681
      ]
    }
  }
}

## Interpretation
Conditional on fixed fitted models and observed class counts; assumes independent parameter records. Excludes retraining, selection uncertainty, shared-Action dependence and distribution shift.
Three candidates selected by validation macro F1. The original test set has already been inspected historically; these are exploratory estimates, not a new external evaluation.

Prediction scores are not confidence intervals. Unlabeled Other precision cannot be estimated without reviewed labels.

## Corrected clustering
2968 eligible Actions; 30478 distinct parameter–Action pairs.

Method: [SciPy bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). The implementation uses explicit class-stratified paired resampling.

Additional JSON reports contain fine-category and known-label-only intervals, plus Wilson intervals for sensitive-category recall.
