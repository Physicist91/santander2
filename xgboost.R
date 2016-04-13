library(xgboost)
library(caret)

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
			        eta=0.02,
			        max_depth=5,
			        colsample_bytree=0.7
			        )

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = dtrain,
                nrounds=1000,
                nfold=7,
                params = param,
                verbose = 2,
                maximize=TRUE,
                stratified=TRUE)

best.score <- max(xgbcv$test.auc.mean/xgbcv$test.auc.std)
best.round <- which(xgbcv$test.auc.mean/xgbcv$test.auc.std == best.score)

set.seed(1234)
clf <- xgb.train(       params              = param,
                        data = dtrain,
                        nrounds             = best.round,
                        verbose             = 2,
                        maximize            = TRUE
       )


preds <- predict(clf, newdata= test, missing=-9999)

submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
