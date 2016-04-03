library(xgboost)


#Building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
			  eval_metric = "auc",
              nthread=2,
			  eta=0.02,
			  max_depth=5)

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = dtrain,
                nrounds=1000,
                nfold=7,
                params = param,
                verbose = 2,
                maximize=T,
                colsample_bytree=0.6815,
                subsample=0.7,
                stratified=TRUE)
                
preds <- rep(0,nrow(test))
for (z in 1:5) {
    set.seed(z + 582365)
    clf <- xgb.train(   params              = param, 
                        data = dtrain,
                        nrounds             = 430, 
                        verbose             = 1,
                        maximize            = TRUE,
                        colsample_bytree=0.6815,
                        subsample=0.7
    )
    
    
    pred <- predict(clf, newdata= test, missing=-9999)
    preds <- preds + pred
}
preds <- preds / 5.0

submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
