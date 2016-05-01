###############################################################
## Implemented a single xgboost model
###############################################################


source("feature.R")

library(xgboost)
library(ggplot2)

train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train)

ID.test <- test$ID
test$ID <- NULL
test <- sparse.model.matrix(TARGET ~. -1, data=test)


# building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
              eval_metric = "auc",
              nthread=2,
              eta=0.010,
              max_depth=4,
              colsample_bytree=0.4,
              min_child_weight=20,
              max_delta_step=5,
              subsample=1)

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = dtrain,
                nrounds=1500,
                nfold=10,
                params = param,
                verbose = 2,
                stratified=TRUE,
                early.stop.round = 500,
                maximize=TRUE)

best.round <- which.max(xgbcv$test.auc.mean - xgbcv$test.auc.std)


ggplot(xgbcv, aes(x=1:nrow(xgbcv), y=train.auc.mean)) +
  geom_ribbon(aes(ymin=train.auc.mean - train.auc.std, ymax=train.auc.mean + train.auc.std), fill='grey') +
  geom_line() +
  geom_ribbon(aes(ymin=test.auc.mean - test.auc.std, ymax=test.auc.mean + test.auc.std),fill='grey70') +
  geom_line(aes(y=test.auc.mean), col='red') +
  ylab('auc')

preds_df <- data.frame(preds=rep(0, length(ID.test)))

for(z in 1:20){
  set.seed(z + 12)
  clf <- xgb.train(       params              = param,
                          data = dtrain,
                          nrounds             = best.round,
                          verbose             = 2
  )
  
  
  preds_df <- cbind(preds_df, predict(clf, newdata= test, missing=-9999))
}


preds <- rowMeans(preds_df[, -1])

submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
