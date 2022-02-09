Notes on Applied Survival Analysis Using R
================
Nayef
2/9/2022

-   [1 Overview](#overview)
-   [2 Libraries](#libraries)
-   [3 Chapter 1](#chapter-1)
    -   [3.1 Hazards for US males and
        females](#hazards-for-us-males-and-females)

# 1 Overview

Reference: DF Moore, *“Applied Survival Analysis Using R”*.

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

# 3 Chapter 1

## 3.1 Hazards for US males and females

``` r
# ?survexp.us
str(survexp.us)
```

    ##  'ratetable' num [1:110, 1:2, 1:75] 1.47e-04 1.52e-05 7.92e-06 5.51e-06 4.44e-06 ...
    ##  - attr(*, "dimnames")=List of 3
    ##   ..$ age : chr [1:110] "0" "1" "2" "3" ...
    ##   ..$ sex : chr [1:2] "male" "female"
    ##   ..$ year: chr [1:75] "1940" "1941" "1942" "1943" ...
    ##  - attr(*, "type")= num [1:3] 2 1 4
    ##  - attr(*, "cutpoints")=List of 3
    ##   ..$ : num [1:110] 0 365 730 1096 1461 ...
    ##   ..$ : NULL
    ##   ..$ : Date[1:75], format: "1940-01-01" "1941-01-01" ...
    ##  - attr(*, "summary")=function (R)  
    ##   ..- attr(*, "srcref")= 'srcref' int [1:8] 1 9 9 3 9 3 1 9
    ##   .. ..- attr(*, "srcfile")=Classes 'srcfilecopy', 'srcfile' <environment: 0x000000001f856d08>
