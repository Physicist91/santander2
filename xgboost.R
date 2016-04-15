library(xgboost)
library(caret)
library(ggplot2)

source("feature.R")

# one-hot encoding
dummies <- dummyVars(~var36, data=all_dat)
ohe <- as.data.frame(predict(dummies, newdata=all_dat))
all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('var36')], ohe)

# standardize NA to -9999 (required for dmatrix)
all_dat[is.na(all_dat$var3), "var3"] <- -9999
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}
var36s <- names(all_dat)[grep('var36', names(all_dat))]
for(i in var36s){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}
all_dat[is.na(all_dat$num_var12_0), "num_var12_0"] <- -9999
all_dat[is.na(all_dat$spain30), "spain30"] <- -9999

train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train, missing=-9999)

ID.test <- test$ID
test$ID <- NULL
test <- sparse.model.matrix(TARGET ~. -1, data=test)


# building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
			        eval_metric = "auc",
              nthread=2,
			        eta=0.01,
			        max_depth=4,
			        colsample_bytree=0.5,
			        min_child_weight=10,
			        max_delta_step=5,
			        subsample=0.75)

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

for(z in 1:50){
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
