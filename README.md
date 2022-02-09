# Survival analysis notes

Notes on survival and recurrent event analysis, from several different references. 

## Repo structure 

- `src` directory: code files 
- `.pre-commit-config.yaml`: config for use with `pre-commit`. It specifies what hooks to use. 
  Once this file is created, if you run `pre-commit install`, the pre-commit tool will populate the 
  `pre-commit` file in the `./.git/hooks` directory. Helpful references: 
    - [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
    - [Keep your code clean using Black & Pylint & Git Hooks & Pre-commit](https://towardsdatascience.com/keep-your-code-clean-using-black-pylint-git-hooks-pre-commit-baf6991f7376)
    - [pre-commit docs](https://pre-commit.com/#)
- `requirements.txt`: python packages used 
- `renv` directory: files created by `renv` R package to replicate environment. Helpful 
  reference: 
  - [Introduction to renv](https://rstudio.github.io/renv/articles/renv.html)
  - Note: my current workflow is to include a `library(<package>)` call in the 
  script I'm working on, then in the console, call `renv::hydrate`. This seems 
  to correctly install the package in the local env. 

## Contents 
1. [Generating a smooth estimate of the survival function via the hazard function](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-09_smoothing-the-km-estimate.md)
2. [Recurrent models based on Cox regression](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-08_recurrent-models-based-on-cod-regression.md)
