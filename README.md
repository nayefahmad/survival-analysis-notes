# Survival analysis notes

Notes on survival and recurrent event analysis, from several different references. 

## Contents 
1. [Generating a smooth estimate of the survival function via the hazard function](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-09_smoothing-the-km-estimate.md)
2. [Recurrent models based on Cox regression](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-08_recurrent-models-based-on-cod-regression.md)
3. [Using the lifelines library in python to fit KM curves](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-10_km-curve-lifelines.ipynb)
4. [Notes and examples from book "Applied Survival Analysis Using R", by D.F. Moore](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-09_notes-on-applied-survival-analysis-using-r.md)
5. [Confidence intervals for conditional survival estimates](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-08-03_confidence-intervals-for-conditional-survival.md)
6. [Deriving conditional survival distributions based on the Weibull distribution](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-09-20_conditional-survival-for-weibull-distribution.ipynb)

## Repo structure 

- `src` directory: code files 
- `.pre-commit-config.yaml`: config for use with `pre-commit`. It specifies what hooks to use. 
  Once this file is created, if you run `pre-commit install`, the pre-commit tool will populate the 
  `pre-commit` file in the `./.git/hooks` directory. Helpful references: 
    - [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
    - [Keep your code clean using Black & Pylint & Git Hooks & Pre-commit](https://towardsdatascience.com/keep-your-code-clean-using-black-pylint-git-hooks-pre-commit-baf6991f7376)
    - [pre-commit docs](https://pre-commit.com/#)
- `.flake8`: config for Flake8. Mainly used to specify max-line-length=88, to match [Black's default](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
- `requirements.txt`: python packages used 
- `renv` directory: files created by `renv` R package to replicate environment. Helpful 
  reference: 
  - [Introduction to renv](https://rstudio.github.io/renv/articles/renv.html)
    
### Notes 
- I use `p2j` to convert from .py files to .ipynb files ([reference](https://pypi.org/project/p2j/)). 
Unfortunately, this doesn't run the file and create outputs, so I do that manually. 

