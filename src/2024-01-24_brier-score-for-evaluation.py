"""
Using the Brier score to evaluate the predictive ability of a time-to-event model

References:
    - [scikit-survival docs](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html#Time-dependent-Brier-Score)  # noqa


Todo:
  - Models:
    - M1: null model that takes in X and returns 0.5 for every case
    - M2: perfect model that takes in X and passes it to the known Weibull survival
      function to get true survival probabilities (or 1-surv_prob if we want a risk
      score)
    - M3: intermediate model that passes to very similar Weibull params, but not exact
    - M4: KM model "learned" from training data
  - set up functions for generating the data and splitting into train/test
  - Evaluate models M1-M4 on test data

"""


# todo: replicate reference above
