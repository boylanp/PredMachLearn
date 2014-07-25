Predictive Machine Learning - Exercise Activity
========================================================
The goal of this assignment is to predict the type of exercize performed based on sensor data.  This analysis will use the default values for a random forest model due to its high accuracy and acceptable computation time for non-linear data.  

## Data Exploration
First the data is read in and the training group is broken apart further into training (60%) and testing (40%) groups.


```r
# To get reproducable results everytime
set.seed(85)
library(caret, quietly = TRUE)

# Have r run on multiple cores/threads
library(doParallel, quietly = TRUE)
cl = makeCluster(detectCores())
registerDoParallel(cl)

# Read in the data
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

# Separate test data into two groups
training <- training[, -c(1, 3:5)]  #First column is redundant, remove noisy time data
inTrain <- createDataPartition(y = training$classe, p = 0.02, list = FALSE)
train1 <- training[inTrain, ]
test1 <- training[-inTrain, ]
```


## Cleaning Data
The next step is to identify features that contain mostly NaNs or will not contribute to the model due to low or near-zero variance. 

```r
nna <- apply(train1, 2, function(x) {
    mean(is.na(x))
}) < 0.9
train1 <- train1[, nna]

nzv <- nearZeroVar(train1, saveMetrics = TRUE)
nzv1 <- rownames(subset(nzv, nzv == TRUE))
train1 <- train1[, -which(names(train1) %in% nzv1)]
```


## Creating the model
This table shows the results from using the train1 subset of the training data.  The error rate of the model on the training data is 0.9XXXXX

```r
modFit <- train(classe ~ ., data = train1, method = "rf", importance = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit$finalModel$confusion
```

```
##     A  B  C  D  E class.error
## A 104  3  3  2  0     0.07143
## B  10 57  7  1  1     0.25000
## C   2  2 60  4  1     0.13043
## D   3  1  6 53  2     0.18462
## E   0  7  5  3 58     0.20548
```


## Cross Validation
The validation portion of the training data is used to determine the accuracy of the model.  The <b>accuracy is 0.9XXX</b> and the <b>out of sample error is 0.XXXX</b>.  We expected the out of sample error to be low, largely due to the large number of observations and features for the model to use.    

```r
confusionMatrix(predict(modFit, newdata = test1), test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5084  355   89   83   43
##          B  200 2839  323   93  315
##          C   42  393 2865  469  178
##          D  129  123   64 2349  136
##          E   13   11   12  157 2862
## 
## Overall Statistics
##                                         
##                Accuracy : 0.832         
##                  95% CI : (0.827, 0.837)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.787         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.930    0.763    0.854    0.745    0.810
## Specificity             0.959    0.940    0.932    0.972    0.988
## Pos Pred Value          0.899    0.753    0.726    0.839    0.937
## Neg Pred Value          0.972    0.943    0.968    0.951    0.958
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.264    0.148    0.149    0.122    0.149
## Detection Prevalence    0.294    0.196    0.205    0.146    0.159
## Balanced Accuracy       0.944    0.851    0.893    0.859    0.899
```


## Prediction of the original test dataset
Here are the results for predicting the exercise for the 20 cases in the testing dataset.  After submission, the model predicted the 20 cases with 100% accuracy.  

```r
answers <- predict(modFit, newdata = testing)
answers
```

```
##  [1] A A B A A E D B A A C C B A E B A B A B
## Levels: A B C D E
```

```r
# pml_write_files = function(x) { n = length(x) for (i in 1:n) { filename =
# paste0('./problem_id_', i, '.txt') write.table(x[i], file = filename,
# quote = FALSE, row.names = FALSE, col.names = FALSE) } }
# pml_write_files(answers)
```




```r
rf = modFit$finalModel
varImpPlot(rf, n.var = nrow(rf$importance[1:25, ]), main = "Importance of the top 25 sensors")
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 


