# Longley Linear Regression
Stephanie Stallworth  
April 21, 2017  



###**Executive Summary**

Linear regression was developed in the field of statistics and is studied as a model for understanding the relationship between input and output numeric variables.  More specifically, it is a linear model that assumes a linear relationship between the input variables (x) and the single output variable (y).  It assumes that y can be calcuated from a linear combination of the input variables (x). As such, both the input values (x) and the output value (y) are numeric. 

In addition to being a statisical algorithm, linear regression is also used in machine learning for making predictions.  There are a number of techniques that could be used to prepare (or train) the linear regression equation from the data, four of which I demonstrate in this analysis:

1. Ordinary Least Squares Regression (OLS)    
2. Stepwize Linear Regression   
3. Principal Component Regression (PCR)  
4. Partial Least Squares Regression (PLS)  


###**PreProcessing**
I'll be applying these techniques to the `longley` dataset, which describes 7 economic variables observed from 1947 to 1962 used to predict the number of people employed yearly. 


```r
# Load statistical packages
library(pls)

# Load data
data(longley)
```

```r
# Examine structure and variable types
str(longley)
```

```
'data.frame':	16 obs. of  7 variables:
 $ GNP.deflator: num  83 88.5 88.2 89.5 96.2 ...
 $ GNP         : num  234 259 258 285 329 ...
 $ Unemployed  : num  236 232 368 335 210 ...
 $ Armed.Forces: num  159 146 162 165 310 ...
 $ Population  : num  108 109 110 111 112 ...
 $ Year        : int  1947 1948 1949 1950 1951 1952 1953 1954 1955 1956 ...
 $ Employed    : num  60.3 61.1 60.2 61.2 63.2 ...
```

```r
# View first few lines
head(longley)
```

```
     GNP.deflator     GNP Unemployed Armed.Forces Population Year Employed
1947         83.0 234.289      235.6        159.0    107.608 1947   60.323
1948         88.5 259.426      232.5        145.6    108.632 1948   61.122
1949         88.2 258.054      368.2        161.6    109.773 1949   60.171
1950         89.5 284.599      335.1        165.0    110.929 1950   61.187
1951         96.2 328.975      209.9        309.9    112.075 1951   63.221
1952         98.1 346.999      193.2        359.4    113.270 1952   63.639
```

```r
# Summarize the data set
summary(longley)
```

```
  GNP.deflator         GNP          Unemployed     Armed.Forces  
 Min.   : 83.00   Min.   :234.3   Min.   :187.0   Min.   :145.6  
 1st Qu.: 94.53   1st Qu.:317.9   1st Qu.:234.8   1st Qu.:229.8  
 Median :100.60   Median :381.4   Median :314.4   Median :271.8  
 Mean   :101.68   Mean   :387.7   Mean   :319.3   Mean   :260.7  
 3rd Qu.:111.25   3rd Qu.:454.1   3rd Qu.:384.2   3rd Qu.:306.1  
 Max.   :116.90   Max.   :554.9   Max.   :480.6   Max.   :359.4  
   Population         Year         Employed    
 Min.   :107.6   Min.   :1947   Min.   :60.17  
 1st Qu.:111.8   1st Qu.:1951   1st Qu.:62.71  
 Median :116.8   Median :1954   Median :65.50  
 Mean   :117.4   Mean   :1954   Mean   :65.32  
 3rd Qu.:122.3   3rd Qu.:1958   3rd Qu.:68.29  
 Max.   :130.1   Max.   :1962   Max.   :70.55  
```

```r
# Visualize the data set 
x<-longley[ ,1:7]

par(mfrow=c(1,7))
  for(i in 1:7) {
  boxplot(x[,i], main=names(longley)[i])
  }
```

![](Longley_Linear_Regression_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

### **Linear Regression 1: Ordinary Least Squares Regression**
When there is more than one input variable, Ordinary Least Squares regression (OLS) can be used to estimate the values of the coefficients.  This regression is a linear model that seeks to find a set of coefficients for a line/hyper-plane that minimizes the sum of the squared errors. In other words, given a regression line through the data, it calculates the distance from each data point to the regression line, squares it, and sums all of the squared errors together. *This* is the quantity that ordinary least squares seeks to minimize.



```r
# Fit model
olsFit<-lm(Employed ~.,longley)

# Summarize the fit
summary(olsFit)
```

```

Call:
lm(formula = Employed ~ ., data = longley)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.41011 -0.15767 -0.02816  0.10155  0.45539 

Coefficients:
               Estimate Std. Error t value Pr(>|t|)    
(Intercept)  -3.482e+03  8.904e+02  -3.911 0.003560 ** 
GNP.deflator  1.506e-02  8.492e-02   0.177 0.863141    
GNP          -3.582e-02  3.349e-02  -1.070 0.312681    
Unemployed   -2.020e-02  4.884e-03  -4.136 0.002535 ** 
Armed.Forces -1.033e-02  2.143e-03  -4.822 0.000944 ***
Population   -5.110e-02  2.261e-01  -0.226 0.826212    
Year          1.829e+00  4.555e-01   4.016 0.003037 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.3049 on 9 degrees of freedom
Multiple R-squared:  0.9955,	Adjusted R-squared:  0.9925 
F-statistic: 330.3 on 6 and 9 DF,  p-value: 4.984e-10
```

```r
# Make predictions
olsPredictions<-predict(olsFit, longley)

# Summarize accuracy
olsMSE<-mean((longley$Employed - olsPredictions)^2)
print(olsMSE)
```

```
[1] 0.0522765
```

###**Linear Regression 2: Stepwize Linear Regression**
Stepwize Linear Regression is a method that makes use of linear regression to discover which subset of attributes in the dataset result in the best performing model. It is step-wise because each iteration of the method makes a change to the set of attributes and creates a model to evaluate the performance of the set.


```r
# Fit model
base<-lm(Employed ~., longley)

# Summarize the fit
summary(base)
```

```

Call:
lm(formula = Employed ~ ., data = longley)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.41011 -0.15767 -0.02816  0.10155  0.45539 

Coefficients:
               Estimate Std. Error t value Pr(>|t|)    
(Intercept)  -3.482e+03  8.904e+02  -3.911 0.003560 ** 
GNP.deflator  1.506e-02  8.492e-02   0.177 0.863141    
GNP          -3.582e-02  3.349e-02  -1.070 0.312681    
Unemployed   -2.020e-02  4.884e-03  -4.136 0.002535 ** 
Armed.Forces -1.033e-02  2.143e-03  -4.822 0.000944 ***
Population   -5.110e-02  2.261e-01  -0.226 0.826212    
Year          1.829e+00  4.555e-01   4.016 0.003037 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.3049 on 9 degrees of freedom
Multiple R-squared:  0.9955,	Adjusted R-squared:  0.9925 
F-statistic: 330.3 on 6 and 9 DF,  p-value: 4.984e-10
```

```r
# Perform step-wise feature selection
slrFit<-step(base)
```

```
Start:  AIC=-33.22
Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces + Population + 
    Year

               Df Sum of Sq     RSS     AIC
- GNP.deflator  1   0.00292 0.83935 -35.163
- Population    1   0.00475 0.84117 -35.129
- GNP           1   0.10631 0.94273 -33.305
<none>                      0.83642 -33.219
- Year          1   1.49881 2.33524 -18.792
- Unemployed    1   1.59014 2.42656 -18.178
- Armed.Forces  1   2.16091 2.99733 -14.798

Step:  AIC=-35.16
Employed ~ GNP + Unemployed + Armed.Forces + Population + Year

               Df Sum of Sq    RSS     AIC
- Population    1   0.01933 0.8587 -36.799
<none>                      0.8393 -35.163
- GNP           1   0.14637 0.9857 -34.592
- Year          1   1.52725 2.3666 -20.578
- Unemployed    1   2.18989 3.0292 -16.628
- Armed.Forces  1   2.39752 3.2369 -15.568

Step:  AIC=-36.8
Employed ~ GNP + Unemployed + Armed.Forces + Year

               Df Sum of Sq    RSS     AIC
<none>                      0.8587 -36.799
- GNP           1    0.4647 1.3234 -31.879
- Year          1    1.8980 2.7567 -20.137
- Armed.Forces  1    2.3806 3.2393 -17.556
- Unemployed    1    4.0491 4.9077 -10.908
```

```r
# Summarize selected model
summary(slrFit)
```

```

Call:
lm(formula = Employed ~ GNP + Unemployed + Armed.Forces + Year, 
    data = longley)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.42165 -0.12457 -0.02416  0.08369  0.45268 

Coefficients:
               Estimate Std. Error t value Pr(>|t|)    
(Intercept)  -3.599e+03  7.406e+02  -4.859 0.000503 ***
GNP          -4.019e-02  1.647e-02  -2.440 0.032833 *  
Unemployed   -2.088e-02  2.900e-03  -7.202 1.75e-05 ***
Armed.Forces -1.015e-02  1.837e-03  -5.522 0.000180 ***
Year          1.887e+00  3.828e-01   4.931 0.000449 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.2794 on 11 degrees of freedom
Multiple R-squared:  0.9954,	Adjusted R-squared:  0.9937 
F-statistic: 589.8 on 4 and 11 DF,  p-value: 9.5e-13
```

```r
# Make predictions
slrPredictions<-predict(slrFit, longley)

# Summarize accuracy
slrMSE<-mean((longley$Employed - slrPredictions)^2)
print(slrMSE)
```

```
[1] 0.05366753
```

###**Linear Regression 3: Principal Component Regression**
Principal Component Regression (PCR) creates a linear regression model using the outputs of a Principal Component Analysis (PCA) to estimate the coefficients of the model. PCR is useful when the data has highly correlated predictors.



```r
# Fit model
pcrFit<-pcr(Employed ~ ., data = longley, valdiation = "cv")

# Summarize the fit
summary(pcrFit)
```

```
Data: 	X dimension: 16 6 
	Y dimension: 16 1
Fit method: svdpc
Number of components considered: 6
TRAINING: % variance explained
          1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
X           64.96    94.90    99.99   100.00   100.00   100.00
Employed    78.42    89.73    98.51    98.56    98.83    99.55
```

```r
# Make predictions
pcrPredictions<-predict(pcrFit, longley, ncomp = 6)

# Summarize accuracy
pcrMSE<-mean((longley$Employed - pcrPredictions)^2)
print(pcrMSE)
```

```
[1] 0.0522765
```

###**Linear Regression 4: Partial Least Squares Regression**
Partial Least Squares (PLS) Regression creates a linear model of the data in a transformed projection of problem space. Like PCR, PLS is appropriate for data with highly-correlated predictors.


```r
# Fit model
plsFit<-plsr(Employed ~., data = longley, validation = "CV")

# Summarize the fit
summary(plsFit)
```

```
Data: 	X dimension: 16 6 
	Y dimension: 16 1
Fit method: kernelpls
Number of components considered: 6

VALIDATION: RMSEP
Cross-validated using 10 random segments.
       (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
CV           3.627    1.461    1.007   0.5194   0.5844   0.4937   0.4459
adjCV        3.627    1.434    1.014   0.5128   0.5726   0.4865   0.4341

TRAINING: % variance explained
          1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
X           63.88    93.35    99.99   100.00   100.00   100.00
Employed    87.91    93.70    98.51    98.65    99.16    99.55
```

```r
# Make predictions
plsPredictions<-predict(plsFit, longley, ncomp = 6)

#Summarize acuracy
plsMSE<-mean((longley$Employed - plsPredictions)^2)
print(plsMSE)
```

```
[1] 0.0522765
```

