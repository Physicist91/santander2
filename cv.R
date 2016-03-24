set.seed(88)
xgbcv <- xgb.cv(data = as.matrix(train[,! names(train) %in% c("ID", "TARGET")]), params = param,
                    nrounds = 1000, max.depth = 5, eta = 0.03,
                    label = as.numeric(train$TARGET), maximize = FALSE, nfold=5, early.stop.round = 310)
plot(1:nrow(xgbcv), xgbcv$test.auc.mean)
xgbcv$test.auc.mean[230] #0.839

set.seed(88)
xgbmodel <- xgboost(data = as.matrix(train[,! names(train) %in% c("ID", "TARGET")]), params = param,
                    nrounds = 230, max.depth = 5, eta = 0.03,
                    label = as.numeric(train$TARGET), maximize = T)
importance_matrix <- xgb.importance(as.character(train$ID), model = xgbmodel)
