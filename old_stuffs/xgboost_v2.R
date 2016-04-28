############################################################
# this version treats missing values separately
############################################################

source("feature.R")

library(xgboost)
library(ggplot2)

training <- all_dat[all_dat$ID %in% dat_train$ID, ]
testing <- all_dat[all_dat$ID %in% dat_test$ID, ]

testing$TARGET <- -1
all_data <- rbind(training, testing)

### Impute var3
var3_data <- all_data[, !names(all_data) %in% c("ID", "num_var12_0", "var36") & !grepl('^delta', names(all_data))]
var3_data <- var3_data[!is.na(var3_data$var3), ]
var3 <- var3_data$var3
dcg_var3 <- sparse.model.matrix(var3 ~. -1, data = var3_data)

dtrain <- xgb.DMatrix(data=dcg_var3,
                      label=var3_data$var3)
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
var3 <- xgb.cv(data=dtrain,
               params = param,
               nfold=5,
               )