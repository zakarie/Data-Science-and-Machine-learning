#-----------------------------------------------------------------------------#
# program : rf.R
# Purpose : build random forest classifier.
# Author  : Zakarie Hashi
# Date    : 05/10/2020
# Require : R >= 3.53
#-----------------------------------------------------------------------------#

# load libraries
library(dplyr)
library(e1071)
library(caret)
library(pROC)


# set up working directory
pth1 <- "C:\\Users\\Hashizx\\Documents\\ML"
pth2 <- "C:/Users/Hashizx/Documents/ML"

setwd(pth1)


# read data
cancer <- data.frame(
  read.csv("breast-cancer-wisconsin.data.txt", header = F, stringsAsFactors = F))


# renaming for readability
names(cancer) <- c('ID', 'thickness','cell_size','cell_shape','adhesion','epithelial_size',
                   'bare_nuclie','bland_cromatin','normal_nucleoli','mitoses','class')

# data wrangling
summary(cancer)
apply(cancer, 2, function(x) any(is.na(x)))
cancer$bare_nuclie <- replace(cancer$bare_nuclie, cancer$bare_nuclie=='?', NA)
apply(cancer, 2, function(x) any(is.na(x)))
cancer2 <- na.omit(cancer)
unique(cancer2$class)
cancer2$class <- (cancer2$class/2) - 1
unique(cancer2$class)

# set data partion - 67% vs 33%
set.seed(20200520)
index <- 1:nrow(cancer2)

test_indx <- sample(index, trunc(length(index)/3))
test_set  <- cancer2[test_indx,]
train_set <- cancer2[-test_indx,]

x_train <- data.matrix(train_set[,2:10]) # excluded 1  -ID
y_train <- as.factor(train_set[,11])

x_test <- data.matrix(test_set[,2:10]) # excluded 1  -ID
y_test <- data.matrix(test_set[,11])


# fit ml models- training models


rf_model <- train(x = x_train, 
                     y = y_train,
                     trControl = trainControl(method = "cv", number = 10), # 10-cross validation
                     method = "rf", 
                     metric = "Accuracy")

# evaluate on test set
rf_pred <- predict.train(rf_model,
                  x_test,
                  type='raw')

# confusion matrix
confusionMatrix(as.factor(rf_pred), as.factor(y_test))

# roc curve
roc_rf<- roc(as.vector(y_test), as.numeric(rf_pred))

# plot roc curve
plot.roc(roc_rf,
         ylim = c(0,1),
         xlim = c(1,0),
         main = 'ROC Curve')
lines(roc_logit, col = 'blue')
legend('bottomright',
       'Random Forest Classifier',
       col = 'blue',
       lwd = 2)

auc(roc_rf)


# apply new data to the trained validated model - Production ML
new_data <- data.frame(
   thickness       = 7.5
  ,cell_size       = 7
  ,cell_shape      = 8
  ,adhesion        = 5
  ,epithelial_size = 5
  ,bare_nuclie     = 7
  ,bland_cromatin  = 9
  ,normal_nucleoli = 8
  ,mitoses         = 10)

pred_rf_new <- predict.train(rf_model,
                        data.matrix(new_data),
                        type = 'prob')


pred_rf_new




