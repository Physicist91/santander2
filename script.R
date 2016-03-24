#########################################################
############### Setup and Initialization ###############
########################################################

library(xgboost)
library(caret)
library(dplyr)

# dat_train <- read.csv("../input/train.csv", stringsAsFactors = F)
# dat_test <- read.csv("../input/test.csv", stringsAsFactors = F)
dat_train <- read.csv("C:/Users/jingpingwong/Downloads/santander/train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("C:/Users/jingpingwong/Downloads/santander/test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- NA


# Merging the test and train data
all_dat <- rbind(dat_train, dat_test)


#################################################################
####################   Feature engineering ######################
#################################################################

#Creating a sum-of-zero field
all_dat$n0 <- apply(all_dat, 1, function(x)(sum(x == 0, na.rm=T)))

# Removing the constant variables
for (i in names(all_dat)[-1])
{
    if (is.integer(all_dat[, i]) & length(unique(all_dat[, i])) == 1) 
    {
        all_dat[, i] <- NULL
        cat("Deleted constant variable: ", i, "\n")
        
    }
}

#Removing duplicate columns
all_dat <- all_dat[!duplicated(lapply(all_dat, summary))]

#Removing highly correlated variables
cor_v <- abs(cor(all_dat))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat("Removing ", names(all_dat)[-unique(cor_v$row)])
all_dat <- all_dat[,-unique(cor_v$row)]

#Handle missing values
all_dat$var3[all_dat$var3 == -999999] <- NA

#Renaming known variables
all_dat <- rename(all_dat, age=var15)

#Creating variable age of type categorical
# all_dat$age <- as.factor(round(all_dat$age/10, 0))

#One-hot encoding
# dummies <- dummyVars(~ age, data=all_dat)
# ohe <- as.data.frame(predict(dummies, newdata=all_dat))
# all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('age')], ohe)

###############################################################################
############################## Model Training #################################
###############################################################################

# Splitting the data for model
train <- all_dat[1:nrow(dat_train), ]
test <- all_dat[-(1:nrow(dat_train)), ]


#Building the model
param <- list("objective" = "binary:logistic",booster = "gbtree", gamma=1,
			  "eval_metric" = "auc", nthread=2, colsample_bytree=0.9, subsample=0.9, min_child_weight=1)

#Parameter values are obtained from cross-validation
set.seed(88)
xgbcv <- xgb.cv(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]), params = param,
                                label=train$TARGET, nrounds = 1500, max.depth = 7, eta = 0.03, maximize = F, missing=NA, nfold=5)
set.seed(88)
xgbmodel <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]), params = param,
                    label=train$TARGET, nrounds = 100, max.depth = 5, eta = 0.001, maximize = T, missing=NA)

#Prediction
preds <- predict(xgbmodel, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]), missing=NA)
submission <- data.frame(ID = test$ID, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
