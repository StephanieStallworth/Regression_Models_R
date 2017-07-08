# Longley Non-Linear Regression
Stephanie Stallworth  
April 21, 2017  



###**Executive Summary**

For this analysis, I will perform non-linear regression analysis on the `longley` data provided in the datasets package in R. The techniques to be applied are:

1. Multivariate Adaptive Regression Splines 
2. Support Vector Machine 
3. k-Nearest Neighbor
4. Neural Network 

###**PreProcessing**


```r
# Load data
data(longley)

# Examine structure
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
# View data
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
# Summarize data
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


###**Multivariate Adaptive Regression Splines**
Multivariate Adadptive Regression Slines is a non-parametric regression method that models multiple nonlinearities in data using hinge functions (functions with a kink in them).


```r
# Load package
library(earth)

# Fit model
marsFit<-earth(Employed ~., longley)

# Summarize the fit
summary(marsFit)
```

```
Call: earth(formula=Employed~., data=longley)

                      coefficients
(Intercept)            -1682.60259
Year                       0.89475
h(293.6-Unemployed)        0.01226
h(Unemployed-293.6)       -0.01596
h(Armed.Forces-263.7)     -0.01470

Selected 5 of 8 terms, and 3 of 6 predictors
Termination condition: GRSq -Inf at 8 terms
Importance: Year, Unemployed, Armed.Forces, GNP.deflator-unused, ...
Number of terms at each degree of interaction: 1 4 (additive model)
GCV 0.2389853    RSS 0.7318924    GRSq 0.9818348    RSq 0.996044
```

```r
# Summarize the importance of input varaibles
evimp(marsFit)
```

```
             nsubsets   gcv    rss
Year                4 100.0  100.0
Unemployed          3  24.1   23.0
Armed.Forces        2  10.4   10.8
```

```r
# Make predictions
marsPredictions <- predict(marsFit, longley)

# Summarize accuracy
marsMSE<-mean((longley$Employed - marsPredictions)^2)
print(marsMSE)
```

```
[1] 0.04574327
```

###**Support Vector Machine**
Support Vector Machines (SVM) are a class of methods, developed originally for classification, that find support points that best separate classes.  SVM for regression is called Support Vector Regression.


```r
# Load package
library(kernlab)

# Fit model
svmFit<-ksvm(Employed ~ .,longley)

# Summarize the fit
summary(svmFit)
```

```
Length  Class   Mode 
     1   ksvm     S4 
```

```r
# Make predictions
svmPredictions<-predict(svmFit, longley)

# Summarize accuracy
svmMSE<-mean((longley$Employed - svmPredictions)^2)
print(svmMSE)
```

```
[1] 0.1805879
```

###**k-Nearest Neighbor**
The k-Nearest Neighbor (kNN) does not create a model, rather it creates predictions from close data on-demand when a prediction is required.  A similarity measure (such as Euclidean distance) is used to locate close data in order to make predictions.


```r
# Load package
library(caret)

# Fit model
knnFit<-knnreg(longley[,1:6], longley[,7], k = 3)

# Summarize Fit
summary(knnFit)
```

```
        Length Class  Mode   
learn   2      -none- list   
k       1      -none- numeric
theDots 0      -none- list   
```

```r
# Make predictions
knnPredictions<-predict(knnFit, longley[ ,1:6])

# Summarize accuracy
knnMSE<-mean((longley$Employed - knnPredictions)^2)
print(knnMSE)
```

```
[1] 0.9259962
```

```r
# Summarize accuracy
```

###**Neural Network**
A Neural Network (NN) is a graph of computational units that receive inputs and transfer the result into an output that is passed on.  The units are ordered into layers to connect the features of the output vector.  With trianing, such as Back-Propagation algorithm, neural networks can be designed and trained to model the underlying relationship in data.


```r
#Load package
library(nnet)

# Fit model
x<-longley[,1:6]
y<-longley[,7]

nnFit<-nnet(Employed ~ ., longley, size = 12, maxit = 500, linout = T, decay = 0.01)
```

```
# weights:  97
initial  value 70929.418908 
iter  10 value 190.352720
iter  20 value 189.399095
iter  30 value 96.279936
iter  40 value 32.966260
iter  50 value 11.990885
iter  60 value 9.604410
iter  70 value 9.101502
iter  80 value 8.513425
iter  90 value 6.423863
iter 100 value 5.446948
iter 110 value 5.141208
iter 120 value 5.013242
iter 130 value 4.925890
iter 140 value 4.898913
iter 150 value 4.868128
iter 160 value 4.848987
iter 170 value 4.812213
iter 180 value 4.759273
iter 190 value 4.756647
iter 200 value 4.754564
iter 210 value 4.740188
iter 220 value 4.680806
iter 230 value 4.550778
iter 240 value 4.404011
iter 250 value 4.307356
iter 260 value 4.245238
iter 270 value 4.190944
iter 280 value 4.150518
iter 290 value 4.126680
iter 300 value 4.091334
iter 310 value 4.065540
iter 320 value 4.046733
iter 330 value 4.030202
iter 340 value 4.021301
iter 350 value 4.013827
iter 360 value 4.010214
iter 370 value 4.006676
iter 380 value 4.003331
iter 390 value 4.001390
iter 400 value 4.000979
iter 410 value 3.997753
iter 420 value 3.975111
iter 430 value 3.944599
iter 440 value 3.930329
iter 450 value 3.922260
iter 460 value 3.917996
iter 470 value 3.915637
iter 480 value 3.913622
iter 490 value 3.911651
iter 500 value 3.909429
final  value 3.909429 
stopped after 500 iterations
```

```r
# Summarize fit
summary(nnFit)
```

```
a 6-12-1 network with 97 weights
options were - linear output units  decay=0.01
 b->h1 i1->h1 i2->h1 i3->h1 i4->h1 i5->h1 i6->h1 
  0.00   0.01   0.07  -0.08  -0.07   0.00   0.01 
 b->h2 i1->h2 i2->h2 i3->h2 i4->h2 i5->h2 i6->h2 
  0.00  -0.10   0.01  -0.06   0.14   0.04   0.00 
 b->h3 i1->h3 i2->h3 i3->h3 i4->h3 i5->h3 i6->h3 
  0.00   0.00   0.06  -0.03  -0.12   0.01   0.01 
 b->h4 i1->h4 i2->h4 i3->h4 i4->h4 i5->h4 i6->h4 
  0.00   0.00   0.00   0.00   0.00   0.00   0.01 
 b->h5 i1->h5 i2->h5 i3->h5 i4->h5 i5->h5 i6->h5 
  0.00   0.04   0.14   0.14  -0.09   0.01  -0.04 
 b->h6 i1->h6 i2->h6 i3->h6 i4->h6 i5->h6 i6->h6 
  0.00   0.00   0.01  -0.01   0.00   0.00   0.00 
 b->h7 i1->h7 i2->h7 i3->h7 i4->h7 i5->h7 i6->h7 
  0.00  -0.42   0.05   0.00   0.04   0.49  -0.02 
 b->h8 i1->h8 i2->h8 i3->h8 i4->h8 i5->h8 i6->h8 
  0.00   0.00   0.00   0.00   0.00   0.00   0.03 
 b->h9 i1->h9 i2->h9 i3->h9 i4->h9 i5->h9 i6->h9 
  0.00  -0.18   0.02  -0.01   0.00   0.07   0.00 
 b->h10 i1->h10 i2->h10 i3->h10 i4->h10 i5->h10 i6->h10 
   0.00   -0.01   -0.04    0.05    0.06    0.00    0.01 
 b->h11 i1->h11 i2->h11 i3->h11 i4->h11 i5->h11 i6->h11 
   0.00    0.00    0.02   -0.03    0.02    0.00    0.00 
 b->h12 i1->h12 i2->h12 i3->h12 i4->h12 i5->h12 i6->h12 
   0.00   -0.13    0.13    0.35    0.02    0.04   -0.06 
  b->o  h1->o  h2->o  h3->o  h4->o  h5->o  h6->o  h7->o  h8->o  h9->o 
  5.74   5.73   5.68   5.74   5.74   3.60   5.74   3.79   5.74   5.73 
h10->o h11->o h12->o 
  5.74   5.74   5.86 
```

```r
# Make predictions
nnPredictions<-predict(nnFit,x, type = "raw")

# Summarize accuracy
nnMSE<-mean((y - nnPredictions)^2)
print(nnMSE)
```

```
[1] 0.000154819
```


