install.packages("tuning.R", repos="http://cran.r-project.org", lib= "~/")
library(xgboost, lib.loc = "~/")


#Building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
			  eval_metric = "auc",
              nthread=2,
			  eta=0.02,
			  max_depth=5)

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                label=train$TARGET,
                nrounds=1000,
                nfold=7,
                params = param,
                verbose = 2,
                maximize=F,
                missing = NA,
                colsample_bytree=0.7,
                subsample=0.7,
                stratified=TRUE)
                
# xgbmodel1 <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
#                      label=train$TARGET,
#                      params=param,
#                      nrounds=480,
#                      verbose=2,
#                      maximize = T,
#                      missing=NA,
#                      colsample_bytree=0.7,
#                      subsample=0.7)
# 
# xgbmodel2 <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
#                      label=train$TARGET,
#                      params=param,
#                      nrounds=480,
#                      verbose=2,
#                      maximize = T,
#                      missing=NA,
#                      colsample_bytree=0.85,
#                      subsample=0.95)

preds <- rep(0,nrow(test))
for (z in 1:5) {
    set.seed(z+12345)
    
    clf <- xgboost(   params              = param, 
                        data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                        label=train$TARGET, 
                        nrounds             = 373, 
                        verbose             = 1,
                        maximize            = FALSE,
                        missing=NA,
                        colsample_bytree=0.85,
                        subsample=0.95
    )
    
    
    pred <- predict(clf, newdata= data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]), missing=NA)
    preds <- preds + pred
}
preds <- preds / 5.0

#Prediction
# preds1 <- predict(xgbmodel1, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]), missing=NA)
# preds2 <- predict(xgbmodel2, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]), missing=NA)
# preds.ensemble <- 0.5 * preds1 + 0.5 * preds2

submission <- data.frame(ID = test$ID, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
