##############################################################
# simple ensembling first
##############################################################

preds <- 0.4 * preds_rf + 0.6 * preds_xgb

submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)

##################################################################
## Ensembling, 2nd layer xgboost
###################################################################

source("rf.R")

y.train <- training$TARGET
train <- sparse.model.matrix(TARGET ~ .-1, data=training)
dtrain <- xgb.DMatrix(data=train, label=y.train, missing=-9999)

ID.test <- testing$ID
test$ID <- NULL
test <- sparse.model.matrix(TARGET ~. -1, data=testing)

# building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
              eval_metric = "auc",
              nthread=2,
              eta=0.015,
              max_depth=4,
              colsample_bytree=0.5,
              min_child_weight=10,
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

clf <- xgb.train(       params              = param,
                        data = dtrain,
                        nrounds             = best.round,
                        verbose             = 2
)


preds <- predict(clf, newdata= test, missing=-9999)

submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
