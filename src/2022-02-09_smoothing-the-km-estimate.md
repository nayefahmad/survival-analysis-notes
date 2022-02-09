Smoothing the KM estimate
================
Nayef
2022-01-06

-   [1 Overview](#overview)
-   [2 Libraries](#libraries)
-   [3 Dataset: Progression-free survival of gastric
    cancer](#dataset-progression-free-survival-of-gastric-cancer)
-   [4 KM curve](#km-curve)
-   [5 Smoothed hazard functions](#smoothed-hazard-functions)
-   [6 Deriving smoothed survival estimtate from smoothed hazard
    function](#deriving-smoothed-survival-estimtate-from-smoothed-hazard-function)
-   [7 Plotting KM and smoothed survival
    estimates](#plotting-km-and-smoothed-survival-estimates)

# 1 Overview

Reference: DF Moore, *“Applied Survival Analysis Using R”*, p32.

# 2 Libraries

``` r
library(survival)
```

    ## Warning: package 'survival' was built under R version 4.0.5

``` r
library(asaur)
```

    ## Warning: package 'asaur' was built under R version 4.0.3

``` r
library(survminer)
```

    ## Warning: package 'survminer' was built under R version 4.0.5

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.0.5

    ## Loading required package: ggpubr

    ## Warning: package 'ggpubr' was built under R version 4.0.3

    ## 
    ## Attaching package: 'survminer'

    ## The following object is masked from 'package:survival':
    ## 
    ##     myeloma

``` r
library(muhaz)
```

    ## Warning: package 'muhaz' was built under R version 4.0.5

# 3 Dataset: Progression-free survival of gastric cancer

``` r
# ?gastricXelox
time_months <- gastricXelox$timeWeeks * 7/30.25
delta <- gastricXelox$delta  # 1 for death, 0 for censored 

str(gastricXelox)
```

    ## 'data.frame':    48 obs. of  2 variables:
    ##  $ timeWeeks: int  4 8 8 8 9 11 12 13 16 16 ...
    ##  $ delta    : int  1 1 1 1 1 1 1 1 1 1 ...

``` r
stopifnot(nrow(gastricXelox) == 48)
```

# 4 KM curve

``` r
km1 <- survfit(Surv(time_months, delta) ~ 1, conf.type = "log-log")

km1
```

    ## Call: survfit(formula = Surv(time_months, delta) ~ 1, conf.type = "log-log")
    ## 
    ##       n events median 0.95LCL 0.95UCL
    ## [1,] 48     32   10.3    5.79    15.3

``` r
ggsurvplot(km1, data = gastricXelox)
```

![](2022-02-09_smoothing-the-km-estimate_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# plot(km1, conf.int = T, mark = "|", xlab = "time_months", ylab = "surv prob")
```

# 5 Smoothed hazard functions

-   Uses `muhaz::pehaz` to estimate piecewise exponential hazard
    function from right-censored data.
-   `width = 5` means we split the time axis and “bin” into intervals of
    length 5 months.

``` r
# ?pehaz
# ?muhaz
hazard_1_month_bin <- pehaz(time_months, delta, width = 1)
```

    ## 
    ## max.time= 58.54545
    ## width= 1
    ## nbins= 59

``` r
hazard_5_month_bin <- pehaz(time_months, delta, width = 5)
```

    ## 
    ## max.time= 58.54545
    ## width= 5
    ## nbins= 12

``` r
hazard_smooth <- muhaz(time_months, delta, 
                       bw.smooth = 20, 
                       b.cor = "left")
hazard_smooth_auto <- muhaz(time_months, delta,
                            bw.method = "local")



plot(hazard_1_month_bin)
lines(hazard_5_month_bin, col = "tomato")
lines(hazard_smooth, col = "deepskyblue3", lwd = 2)
lines(hazard_smooth_auto, col = "blue", lwd = 2)

title("Estimated hazard functions using piecewise exponential estimation \nand kernel-based methods")
```

![](2022-02-09_smoothing-the-km-estimate_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

# 6 Deriving smoothed survival estimtate from smoothed hazard function

``` r
hazard_values <- hazard_smooth_auto$haz.est
times <- hazard_smooth_auto$est.grid
n_haz <- length(hazard_values)

smooth_surv <- exp(-cumsum(hazard_values[1:n_haz - 1] * diff(times)))
```

# 7 Plotting KM and smoothed survival estimates

``` r
plot(km1, conf.int = F, mark = "|", 
     xlab = "time_months", ylab = "surv prob", 
     xlim = c(0, 30))
lines(smooth_surv ~ times[1:(length(times) - 1)], col = "blue", lwd =2)
title("Survival estimates - KM curve and smoothed curve")
```

![](2022-02-09_smoothing-the-km-estimate_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
