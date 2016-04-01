install.packages('randomForest', repos='http://cran.us.r-project.org', lib= "~/")

library("randomForest", lib.loc = "~/")

source("feature.R")

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

#rfmodel <- randomForest(as.factor(TARGET) ~ ., data=train[, !names(train) == 'ID'], mtry = best.mtry, ntree= 501, na.action=na.omit)
