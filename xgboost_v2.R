library(xgboost)
library(ggplot2)
library(Matrix)

source("feature_engineering.R")

train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train)

AGE <- test$var15
var36.0 <- test$var36.0
num_var13_largo_0 <- test$num_var13_largo_0
saldo_var33 <- test$saldo_var33
ID.test <- test$ID
test$ID <- NULL
tc <- test
test <- sparse.model.matrix(TARGET ~. -1, data=test)


# building the model
param <- list(objective = "binary:logistic",
                    booster = "gbtree",
			        eval_metric = "auc",
                    nthread=2,
			        eta=0.017,
			        max_depth=5,
			        colsample_bytree=0.7,
			        min_child_weight=1,
			        subsample=0.7)

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = dtrain,
                nrounds=1500,
                nfold=5,
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
importance_matrix <- xgb.importance(train@Dimnames[[2]],model= clf)
xgb.plot.importance(importance_matrix[1:20,])
importance_matrix[1:30,]

preds <- predict(clf, newdata=test)
#preds_df <- data.frame(preds=rep(0, length(ID.test)))
# for(z in 1:10){
#   set.seed(z + 12)
#   clf <- xgb.train(       params              = param,
#                           data = dtrain,
#                           nrounds             = best.round,
#                           verbose             = 0
#   )
#   print(z)
#   
#   preds_df <- cbind(preds_df, predict(clf, newdata= test))
# }
#preds <- rowMeans(preds_df[, -1])

preds[AGE < 23] <- 0
preds[var36.0 == 1] <- 0
preds[num_var13_largo_0 > 3] <- 0
preds[saldo_var33 > 0] <- 0

nv = tc['num_var33']+tc['saldo_medio_var33_ult3']+tc['saldo_medio_var44_hace2']+tc['saldo_medio_var44_hace3']+
    tc['saldo_medio_var33_ult1']+tc['saldo_medio_var44_ult1']

preds[nv > 0] = 0
preds[tc['var15'] < 23] = 0
preds[tc['saldo_medio_var5_hace2'] > 160000] = 0
preds[tc['saldo_var33'] > 0] = 0
preds[tc['var38'] > 3988596] = 0
preds[tc['var21'] > 7500] = 0
preds[tc['num_var30'] > 9] = 0
preds[tc['num_var13_0'] > 6] = 0
preds[tc['num_var33_0'] > 0] = 0
preds[tc['imp_ent_var16_ult1'] > 51003] = 0
preds[tc['imp_op_var39_comer_ult3'] > 13184] = 0
preds[tc['saldo_medio_var5_ult3'] > 108251] = 0
preds[tc['num_var37_0'] > 45] = 0
preds[tc['saldo_var5'] > 137615] = 0
preds[tc['saldo_var8'] > 60099] = 0
preds[(tc['var15']+tc['num_var45_hace3']+tc['num_var45_ult3']+tc['var36']) <= 24] = 0
preds[tc['saldo_var14'] > 19053.78] = 0
preds[tc['saldo_var17'] > 288188.97] = 0
preds[tc['saldo_var26'] > 10381.29] = 0
preds[tc['num_var13_largo_0'] > 3] = 0
preds[tc['imp_op_var40_comer_ult1'] > 3639.87] = 0
preds[tc['num_var5_0'] > 6] = 0


submission <- data.frame(ID = ID.test, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)