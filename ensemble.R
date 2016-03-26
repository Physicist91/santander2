xgbmodel1 <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                     label=train$TARGET,
                     nrounds = 310,
                     verbose=2,
                     maximize = T,
                     missing=NA,
                     objective = "binary:logistic",
                     booster = "gbtree",
                     eval_metric = "auc",
                     nthread=2,
                     eta=0.02,
                     max_depth=5,
                     colsample_bytree=0.7,
                     subsample=0.7)

xgbmodel2 <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                     label=train$TARGET,
                     nrounds = 310,
                     verbose=2,
                     maximize = T,
                     missing=NA,
                     objective = "binary:logistic",
                     booster = "gbtree",
                     eval_metric = "auc",
                     nthread=2,
                     eta=0.02,
                     max_depth=5,
                     colsample_bytree=0.85,
                     subsample=0.95)

preds1 <- predict(xgbmodel1, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]))
preds2 <- predict(xgbmodel2, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]))

preds.ensemble <- 0.5 * preds1 + 0.5 * preds2
