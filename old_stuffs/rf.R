###############################################################
## Implementation of Random Forest
###############################################################

source("feature.R")

library("randomForest")

# deleting variables with missing values
all_dat$var3 <- NULL
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[, i] <- NULL
}
all_dat$num_var12_0 <- NULL
all_dat$var36 <- NULL

# splitting the data for model
training <- all_dat[all_dat$ID %in% dat_train$ID, ]
testing <- all_dat[all_dat$ID %in% dat_test$ID, ]

training$ID <- NULL


# stratified sampling for training data
subdata <- training[training$TARGET == 0, ]
index_fold1 <- sample(1:nrow(subdata), nrow(subdata)/2)
training_fold1 <- subdata[index_fold1, ]
training_fold2 <- subdata[-index_fold1, ]
subdata1 <- training[training$TARGET == 1, ]
index_fold2 <- sample(1:nrow(subdata1), nrow(subdata1)/2)
training_fold1 <- rbind(training_fold1, subdata1[index_fold2, ])
training_fold2 <- rbind(training_fold2, subdata1[-index_fold2,])

# #Tuning
# mtry.max = ncol(train) - 1
# err.rf <- rep(0, mtry.max)
# for(m in 1:mtry.max){
#   set.seed(1234)
#   
#   rfmodel <- randomForest(as.factor(TARGET) ~ ., data=train[, !names(train) == 'ID'], mtry = m, ntree= 501, na.action=na.omit)
#   
#   err.rf[m] <- rf$err.rate[501]
# }
# 
# plot(1:mtry.max, err.rf, type = "b", xlab = "mtry", ylab = "OOB error")
# 
# best.mtry <- which.min(err.rf)

# train both folds
rf_clf_fold1 <- randomForest(as.factor(TARGET) ~ ., data=training_fold1, ntree= 100, do.trace=TRUE)

# train fold2
rf_clf_fold2 <- randomForest(as.factor(TARGET) ~ ., data=training_fold2, ntree= 100, do.trace=TRUE)

# train whole data
rf_clf <- randomForest(as.factor(TARGET) ~ ., data=training, ntree= 100, do.trace=TRUE)

# analysis of the model
varImpPlot(rf_clf_fold1)
varImpPlot(rf_clf_fold2)
training_fold2$preds <- predict(rf_clf_fold1, newdata=training_fold2)
table(training_fold2$preds, training_fold2$TARGET)
training_fold1$preds <- predict(rf_clf_fold2, newdata=training_fold1)
table(training_fold1$preds, training_fold1$TARGET)

# generate features for xgboost
testing$preds <- predict(rf_clf, newdata=testing)