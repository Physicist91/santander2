library("randomForest")

#Tuning
mtry.max = ncol(train) - 1
err.rf <- rep(0, mtry.max)
for(m in 1:mtry.max){
  set.seed(1234)
  
  rf <- randomForest(as.factor(TARGET) ~ ., data=train, subset=train.index, ntree=501, mtry = m)
  
  err.rf[m] <- rf$err.rate[501]
}

plot(1:mtry.max, err.rf, type = "b", xlab = "mtry", ylab = "OOB error")

best.mtry <- which.min(err.rf)

rfmodel <- randomForest(TARGET ~ ., data=train[, !names(train) == 'ID'], mtry = best.mtry)