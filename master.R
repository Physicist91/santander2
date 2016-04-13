install.packages('caret', repos='http://cran.us.r-project.org', lib= "~/")
install.packages('dplyr', repos='http://cran.us.r-project.org', lib= "~/")
install.packages("xgboost", repos="https://cran.rstudio.com", lib= "~/")
library(caret, lib.loc = "~/")
library(dplyr, lib.loc = "~/")
library(xgboost, lib.loc= "~/")
library(Matrix)

# dat_train <- read.csv("../input/train.csv", stringsAsFactors = F)
# dat_test <- read.csv("../input/test.csv", stringsAsFactors = F)
train <- read.csv("train.csv", stringsAsFactors = FALSE)
#dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

#dat_test$TARGET <- -1

# merging the test and train data
#all_dat <- rbind(dat_train, dat_test)

# standardize missing values
train[train$var3 == -999999, "var3"] <- NA
delta_vars <- names(train)[grep('^delta', names(train))]
for(i in delta_vars){
  train[train[, i] == 9999999999, i] <- NA
}
train[train$var36 == 99, "var36"] <- NA
train[train$num_var12_0 == 111, "num_var12_0"] <- NA

# removing the constant variables
zeroVar <- nearZeroVar(train, saveMetrics = TRUE, freqCut = (nrow(train) - 10)/10,uniqueCut = 1000/nrow(train))
cat(names(train)[zeroVar[,"nzv"]], sep="\n")
train <- train[, !zeroVar[, "nzv"]]

#Removing duplicate columns
temp <- names(train)[duplicated(lapply(train, summary))]
cat(temp, sep="\n")
train <- train[, !names(train) %in% temp]

#Removing highly correlated variables
cor_v <- abs(cor(train))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat(names(train)[unique(cor_v$row)], sep="\n")
train <- train[,-unique(cor_v$row)]

# treat age variable
# all_dat$ageDiscrete <- NA
# all_dat[all_dat$age < 18, "ageDiscrete" ] <- "below.18"
# all_dat[all_dat$age >= 18 &  all_dat$age < 28, "ageDiscrete"] <- "18.to.25"
# all_dat[all_dat$age >= 28 & all_dat$age < 40, "ageDiscrete" ] <- "28.to.40"
# all_dat[all_dat$age >= 40 & all_dat$age < 70, "ageDiscrete"] <- "40.to.70"
# all_dat[all_dat$age >= 70, "ageDiscrete"] <- "70.above"
# all_dat$ageDiscrete <- as.factor(all_dat$ageDiscrete)

# convert to categorical
train$var36 <- as.factor(train$var36)

train$nonzero <- apply(train, 1, function(x) (sum(x != 0, na.rm=TRUE)))

train$ind_count <- apply(train[, grep('^ind', names(train))], 1, function(x)(sum(x == 0)))

# balance/saldo zero or less
train$saldo0 <- apply(train[, grep('^saldo', names(train))], 1, function(x)(sum(x < 0)))
train$spain30 <- (train$var15 > 30 & train$var3 == 2) * 1

train$var5_ratio <- (train$saldo_medio_var5_hace2 + 1)/(train$saldo_medio_var5_hace3 + 1)

dummies <- dummyVars(~var36, data=train)
ohe <- as.data.frame(predict(dummies, newdata=train))
train <- cbind(train[, ! names(train) %in% c('var36')], ohe)

# standardize NA to -9999 (required for dmatrix)
train[is.na(train$var3), "var3"] <- -9999
delta_vars <- names(train)[grep('^delta', names(train))]
for(i in delta_vars){
  train[is.na(train[, i]), i] <- -9999
}
var36s <- names(train)[grep('var36', names(train))]
for(i in var36s){
  train[is.na(train[, i]), i] <- -9999
}
train[is.na(train$num_var12_0), "num_var12_0"] <- -9999
train[is.na(train$spain30), "spain30"] <- -9999

xgb_grid <- expand.grid(nrounds=c(300, 500, 700, 1000, 1500),
                        max_depth=c(5, 6, 7, 8),
                        eta=c(0.001, 0.005, 0.01, 0.02, 0.03, 0.1),
                        gamma=c(0, 1),
                        colsample_bytree=c(0.3, 0.5, 0.7, 0.8),
                        min_child_weight=c(5, 10, 20, 40)
                        )

xgb_trcontrol <- trainControl(method='cv',
                              number=7,
                              verboseIter=TRUE,
                              returnData=TRUE,
                              returnResamp = "all",
                              classProbs=TRUE,
                              summaryFunction = twoClassSummary,
                              allowParallel = TRUE)

xgb_tune <- train(x=data.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                  y=ifelse(train$TARGET == 0, "S", "U"),
                  method='xgbTree',
                  trControl=xgb_trcontrol,
                  tuneGrid = xgb_grid,
                  verbose=TRUE,
                  eval_metric='auc',
                  objective='binary:logistic',
                  nthread=8,
                  missing=-9999)

write.csv(xgb_tune, "xgb_tune.csv")
