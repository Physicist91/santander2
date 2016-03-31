# library(xgboost)
# library(Matrix)
# 
# set.seed(1234)
# 
# train <- read.csv("train.csv", stringsAsFactors = FALSE)
# test <- read.csv("test.csv", stringsAsFactors = FALSE)
# 
# ##### Removing IDs
# train$ID <- NULL
# test.id <- test$ID
# test$ID <- NULL
# 
# ##### Extracting TARGET
# train.y <- train$TARGET
# train$TARGET <- NULL
# 
# ##### 0 count per line
# count0 <- function(x) {
#     return( sum(x == 0) )
# }
# train$n0 <- apply(train, 1, FUN=count0)
# test$n0 <- apply(test, 1, FUN=count0)
# 
# ##### Removing constant features
# cat("\n## Removing the constants features.\n")
# for (f in names(train)) {
#     if (length(unique(train[[f]])) == 1) {
#         cat(f, "is constant in train. We delete it.\n")
#         train[[f]] <- NULL
#         test[[f]] <- NULL
#     }
# }
# 
# ##### Removing identical features
# features_pair <- combn(names(train), 2, simplify = F)
# toRemove <- c()
# for(pair in features_pair) {
#     f1 <- pair[1]
#     f2 <- pair[2]
#     
#     if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
#         if (all(train[[f1]] == train[[f2]])) {
#             cat(f1, "and", f2, "are equals.\n")
#             toRemove <- c(toRemove, f2)
#         }
#     }
# }
# 
# feature.names <- setdiff(names(train), toRemove)
# 
# train <- train[, feature.names]
# test <- test[, feature.names]
# 
# train$TARGET <- train.y
# 
# 
# train <- sparse.model.matrix(TARGET ~ ., data = train)

library(xgboost)
library(Matrix)

set.seed(1234)

train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
    return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
    if (length(unique(train[[f]])) == 1) {
        cat(f, "is constant in train. We delete it.\n")
        train[[f]] <- NULL
        test[[f]] <- NULL
    }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
        if (all(train[[f1]] == train[[f2]])) {
            cat(f1, "and", f2, "are equals.\n")
            toRemove <- c(toRemove, f2)
        }
    }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]

train$TARGET <- train.y


train <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=dtrain)

test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.01,
                max_depth           = 6,
                subsample           = 0.9,
                colsample_bytree    = 0.85
)

preds <- rep(0,nrow(test))
for (z in 1:5) {
    set.seed(z+582365)
    
    clf <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = 800, 
                        verbose             = 1,
                        watchlist           = watchlist,
                        maximize            = FALSE
    )
    
    
    pred <- predict(clf, test)
    preds <- preds + pred
}
preds <- preds / 5.0

submission <- data.frame(ID=test.id, TARGET=preds)
cat("saving the submission file\n")
write.csv(submission, "submission.csv", row.names = F)