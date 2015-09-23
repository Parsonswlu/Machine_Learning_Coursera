---
title: "Coursera Machine Learning Project"
author: "Andrew Parsons"
date: "Sept. 22, 2015"
output: pdf_document
---

###Abstract

Using accelerometer data taken from six participants doing a variety of exercises, classification tree and random forest models were trained to predict which activity was being done given only the accelerometer data. Ultimately the random forest model was chosen for forecasting, as it had the highest prediction accuracy on the cross validation set (99.5%).

###Background

Six participants were asked to wear devices on their belt, forearm, arm and dumbell while performing barbell lifts correctly and incorrectly in five different ways. The data provided are readings from the accelerometers. The goal of the exercise is to use this data to ascertain which of the five different ways a user is doing the exercise based on new accelerometer data.

###Initializing the data

The first step is to load the apprpriate libraries and download and read the data from the [provided source of accelerometer data](http://groupware.les.inf.puc-rio.br/har).


```r
library(caret); library(knitr); library(rpart); library(rpart.plot); library(randomForest)
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

buildData <- read.csv(url(url_train), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(url_test), na.strings=c("NA","#DIV/0!",""))
```

###Examining the Data

Once the data is loaded it is examined to get an idea of what is being worked with.


```r
dim(buildData) 
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

The data initially contains 160 variates (including the one being predicted on - 'classe'). To simplify things, some data cleaning can be done so the machine learning model can provide a better result.

The first 7 columns are descriptive - i.e. the row index, the name of the participant, the time the exercise occurred, etc. These can be removed as they should have no robust predictive value.

There are also a large number of columns (100) that consist entirely (or almost entirely) of NA's. These are also removed for purposes of prediction.


```r
# First 7 columns are descriptive and not useful for prediction
summary(buildData[,1:7])
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window    num_window   
##  28/11/2011 14:14: 1498   no :19216   Min.   :  1.0  
##  05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0  
##  30/11/2011 17:11: 1440               Median :424.0  
##  05/12/2011 11:25: 1425               Mean   :430.6  
##  02/12/2011 14:57: 1380               3rd Qu.:644.0  
##  02/12/2011 13:34: 1375               Max.   :864.0  
##  (Other)         :11007
```

```r
# 100 variables contain almost entirely NAs
hist(colSums(is.na(buildData)))
```

![plot of chunk examine_data2](figure/examine_data2-1.png) 

```r
# Remove first 7 columns
buildData <- buildData[,8:length(buildData)]
testing <- testing[,8:length(testing)]

# Remove Columns that contain almost entirely NA's
columns_with_NAs <- as.logical(!(colSums(is.na(buildData)) == 0))
buildData <- buildData[!columns_with_NAs]
testing <- testing[!columns_with_NAs]
```

Now the data set is down to a more reasonable size of 53 (including the predictor 'classe').

###Split the Data

To determine which model is expect to perform best on the test set, the model will be trained on a training set then checked against a validation set. The data is split 70% training / 30% validation for this purpose. Seed is set as 5555 for reproducibility purposes.


```r
set.seed(5555)
inTrain <- createDataPartition(buildData$classe, p = 0.7, list = FALSE)
training <- buildData[inTrain,]
validation <- buildData[-inTrain,]
```

###Model 1 - Classification Tree Model

For the purpose of classifying the 'classe' variable into one of five categories, the most logical first step would be to try a straightforward classification tree using the 52 predictors. The training, visualization of the classification tree and results from a confusion matrix are shown below.


```r
# Classification Tree Training
fit_rpart <- rpart(classe ~ ., data = training, method = "class")

# Classification Tree Visualization
rpart.plot(fit_rpart)
```

![plot of chunk classification_tree](figure/classification_tree-1.png) 

```r
# Classification Tree Prediction results and Confusion Matrix
predict_rpart <- predict(fit_rpart, newdata = validation, type = "class")
confusionMatrix(predict_rpart, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1483  248   30  109   55
##          B   45  650  128   46  197
##          C   49  123  803  133  103
##          D   68  104   65  627   63
##          E   29   14    0   49  664
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7183          
##                  95% CI : (0.7066, 0.7297)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6418          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8859   0.5707   0.7827   0.6504   0.6137
## Specificity            0.8950   0.9123   0.9160   0.9390   0.9808
## Pos Pred Value         0.7704   0.6098   0.6631   0.6764   0.8783
## Neg Pred Value         0.9518   0.8985   0.9523   0.9320   0.9185
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2520   0.1105   0.1364   0.1065   0.1128
## Detection Prevalence   0.3271   0.1811   0.2058   0.1575   0.1285
## Balanced Accuracy      0.8905   0.7415   0.8493   0.7947   0.7973
```

Prediction using this model on the validation set provides an overall accuracy of 71.8%, which is good but not great. Note however that this classification model took very little processing time to execute, and in a pinch could be used if results were needed very quickly.

###Model 2 - Random Forests

Random forests are more complex than classification trees and offer additional benefits, including robustness to outliers. This comes at a tradeoff of increased processing time and reduced interpretability.

To help minimize the processing time for the random forest model, it may help to tune the model by determining ahead of time an ideal number of variables and trees to use. The tuneRF function below shows that the optimal number of variables to use in this case was 7 and that 250 trees was plenty to minimize the out-of-bag error estimate for each category of exercise. 


```r
# Try tuning Random Forest - optimal OOB error uses 7 variables
tune <- tuneRF(training[,-53], training$classe)
```

```
## mtry = 7  OOB error = 0.78% 
## Searching left ...
## mtry = 4 	OOB error = 0.97% 
## -0.2429907 0.05 
## Searching right ...
## mtry = 14 	OOB error = 0.9% 
## -0.1495327 0.05
```

![plot of chunk random_forest](figure/random_forest-1.png) 

```r
# Random Forest Training with 7 variables and 250 trees
fit_rf <- randomForest(classe ~ ., data = training, ntree = 250, mtry = 7,
                       prox=TRUE, keep.forest = TRUE)

# OOB error estimate per number of trees
plot(fit_rf, log="y", type='l', main="Out-of-bag error estimate per number of trees")
legend("topright", legend = colnames(fit_rf$err.rate), col=1:6, fill=1:6)
```

![plot of chunk random_forest](figure/random_forest-2.png) 

```r
# Random Forest Prediction results and Confusion Matrix
predict_rf <- predict(fit_rf, newdata = validation)
confusionMatrix(predict_rf, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    6    0    0    0
##          B    3 1132    5    0    0
##          C    0    1 1021   13    0
##          D    0    0    0  951    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9939   0.9951   0.9865   0.9972
## Specificity            0.9986   0.9983   0.9971   0.9994   1.0000
## Pos Pred Value         0.9964   0.9930   0.9865   0.9969   1.0000
## Neg Pred Value         0.9993   0.9985   0.9990   0.9974   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1924   0.1735   0.1616   0.1833
## Detection Prevalence   0.2850   0.1937   0.1759   0.1621   0.1833
## Balanced Accuracy      0.9984   0.9961   0.9961   0.9930   0.9986
```

Prediction using the random forest model provides an overall accuracy of 99.5% on the validation set. Note the improved accuracy came at a significant increase in processing time and a decreased intepretability as to why the chosen parameters were chosen.

###Random Forest Prediction on Testing Data

Given the high accuracy of the random forest model on the validation set, this model is chosen to be used on the test set of 20 cases using the following lines of code:


```r
answers = predict(fit_rf, newdata = testing)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)
```

Assuming the same out-of-sample error rate as was seen on the validation set (~0.5%), it is expected that the program will accurately predict all 20 exercises.

###Summary

Given a set of (cleaned) accelerometer data, classification tree and random forest models were trained and validated against to try and predict out-of-sample accuracy. The classification tree resulted in an okay prediction result of 71.8%, however the random forest model had a much higher accuracy rate of 99.5%. Even with the increased processing time, the random forest model was greatly superior in predicting out-of-sample performance on the test set.

##Appendix

###Systems Used

Program created in RStudio using R version 3.2.2 (2015-08-14) -- "Fire Safety" 

Windows 8.1 

Dell Laptop, 64-bit operating system 

###Data source

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

