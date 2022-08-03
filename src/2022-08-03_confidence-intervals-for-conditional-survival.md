Confidence intervals for conditional survival estimates
================
Nayef
8/3/2022

-   [1 Overview](#overview)
    -   [1.1 References](#references)
-   [2 Libraries](#libraries)
-   [3 Example](#example)

# 1 Overview

## 1.1 References

1.  `condsurv` R package [docs](https://www.emilyzabor.com/condsurv/)
2.  `condsurv` R package [github
    page](https://github.com/zabore/condsurv)

# 2 Libraries

``` r
# library("remotes")
# install_github("zabore/condsurv")

library("condsurv")
library(dplyr)
```

    ## Warning: package 'dplyr' was built under R version 4.0.3

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(survival)
```

    ## Warning: package 'survival' was built under R version 4.0.5

# 3 Example

``` r
lung2 <- 
  mutate(
    lung,
    os_yrs = time / 365.25
  )


myfit <- survfit(Surv(os_yrs, status) ~ 1, data = lung2)

# ?conditional_surv_est
conditional_surv_est(
  basekm = myfit,
  t1 = 0.5, 
  t2 = 1
)
```

    ## $cs_est
    ## [1] 0.58
    ## 
    ## $cs_lci
    ## [1] 0.49
    ## 
    ## $cs_uci
    ## [1] 0.66
