library(caret)
library(dplyr)
library(Matrix)
library(xgboost)
#library(ROSE)

# dat_train <- read.csv("../input/train.csv", stringsAsFactors = F)
# dat_test <- read.csv("../input/test.csv", stringsAsFactors = F)
dat_train <- read.csv("train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- -1

# merging the test and train data
all_dat <- rbind(dat_train, dat_test)

# standardize missing values
all_dat[all_dat$var3 == -999999, "var3"] <- NA
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[all_dat[, i] == 9999999999, i] <- NA
}
all_dat[all_dat$var36 == 99, "var36"] <- NA

# removing the constant variables
zeroVar <- nearZeroVar(all_dat, saveMetrics = TRUE)
names(all_dat)[zeroVar[,"zeroVar"]]
all_dat <- all_dat[, !zeroVar[, "zeroVar"]]

#Removing duplicate columns
temp <- names(all_dat)[duplicated(lapply(all_dat, summary))]
cat(temp, sep="\n")
all_dat <- all_dat[, !names(all_dat) %in% temp]

#Removing highly correlated variables
#This prevents overfitting
cor_v <- abs(cor(all_dat))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat(names(all_dat)[unique(cor_v$row)], sep="\n")
all_dat <- all_dat[,-unique(cor_v$row)]

# treat age variable
all_dat <- rename(all_dat, age=var15)
all_dat$ageDiscrete <- NA
all_dat[all_dat$age < 18, "ageDiscrete" ] <- "below.18"
all_dat[all_dat$age >= 18 &  all_dat$age < 28, "ageDiscrete"] <- "18.to.25"
all_dat[all_dat$age >= 28 & all_dat$age < 40, "ageDiscrete" ] <- "28.to.40"
all_dat[all_dat$age >= 40 & all_dat$age < 70, "ageDiscrete"] <- "40.to.70"
all_dat[all_dat$age >= 70, "ageDiscrete"] <- "70.above"

#Transforming categorical variable by one-hot-encoding
all_dat$var36 <- as.factor(all_dat$var36)

dummies <- dummyVars(~ ageDiscrete + var36, data=all_dat)
ohe <- as.data.frame(predict(dummies, newdata=all_dat))
all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('ageDiscrete', 'var36')], ohe)

# experimental: imputing zeros in saldo
# as there's another indicator in ind_var
# plus standardize NA to -9999
all_dat[all_dat$saldo_var30 == 0, "saldo_var30"] <- -9999
all_dat[all_dat$saldo_var5 == 0, "saldo_var5"] <- -9999
# standardize missing values
all_dat[is.na(all_dat$var3), "var3"] <- -9999
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}
var36s <- names(all_dat)[grep('var36', names(all_dat))]
for(i in var36s){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}

# Splitting the data for model
train <- all_dat[all_dat$ID %in% dat_train$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train, missing=-9999)

#Synthetic data generation
#train <- ROSE(TARGET ~ ., data=train[!names(train) == 'ID'], N=228060, seed=8888)$data

test <- all_dat[all_dat$ID %in% dat_test$ID, ]
y.test <- test$TARGET
ID.test <- test$ID
test$ID <- NULL
test <- sparse.model.matrix(TARGET ~. -1, data=test)
