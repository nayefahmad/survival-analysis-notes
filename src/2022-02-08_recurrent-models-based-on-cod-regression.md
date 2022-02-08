Recurrent models based on Cox regression
================
Nayef
2022-01-06

-   [1 Overview](#overview)
-   [2 Libraries](#libraries)
-   [3 Test](#test)

# 1 Overview

Reference: [A systematic comparison of recurrent event models for
application to composite
endpoints](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5755224/)

Specifically, see the Additional File section.

# 2 Libraries

``` r
library(survival)
```

    ## Warning: package 'survival' was built under R version 4.0.5

# 3 Test

``` r
x <- rnorm(100)
hist(x)
```

![](2022-02-08_recurrent-models-based-on-cod-regression_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
