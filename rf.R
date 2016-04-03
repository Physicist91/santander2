library("randomForest")
library("ROSE")

source("feature.R")

# splitting the data for model
train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]

# generating synthetic data
train <- ROSE(TARGET ~ ., data=train[!names(train) == 'ID'], seed=8888)$data

#Tuning
mtry.max = ncol(train) - 1
err.rf <- rep(0, mtry.max)
for(m in 1:mtry.max){
  set.seed(1234)
  
  rfmodel <- randomForest(as.factor(TARGET) ~ ., data=train[, !names(train) == 'ID'], mtry = m, ntree= 501, na.action=na.omit)
  
  err.rf[m] <- rf$err.rate[501]
}

plot(1:mtry.max, err.rf, type = "b", xlab = "mtry", ylab = "OOB error")

best.mtry <- which.min(err.rf)

rfmodel <- randomForest(as.factor(TARGET) ~ ., data=train[, !names(train) == 'ID'], ntree= 501, na.action=na.omit)

preds <- predict(rfmodel, newdata=test[, !names(test) %in% c("ID", "TARGET")], type='vote')
preds
