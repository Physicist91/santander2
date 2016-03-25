#########################################################
############### Setup and Initialization ###############
########################################################

library(xgboost)
library(caret)
library(dplyr)

# dat_train <- read.csv("../input/train.csv", stringsAsFactors = F)
# dat_test <- read.csv("../input/test.csv", stringsAsFactors = F)
dat_train <- read.csv("~/santander2/train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("~/santander2/test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- NA


# Merging the test and train data
all_dat <- rbind(dat_train, dat_test)


#################################################################
####################   Feature engineering ######################
#################################################################

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
# cor_v <- abs(cor(all_dat))
# diag(cor_v) <- 0
# cor_v[upper.tri(cor_v)] <- 0
# cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
# cat("Removing ", names(all_dat)[-unique(cor_v$row)])
# all_dat <- all_dat[,-unique(cor_v$row)]

#Handle missing values
all_dat$var3NA <- ifelse(all_dat$var3 == -999999, 1, 0)

#Renaming known variables
all_dat <- rename(all_dat, age=var15)

#Creating variable of type categorical
all_dat$ageDiscrete <- as.factor(round(all_dat$age/10, 0))

#One-hot encoding
dummies <- dummyVars(~ ageDiscrete, data=all_dat)
ohe <- as.data.frame(predict(dummies, newdata=all_dat))
all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('ageDiscrete')], ohe)

###############################################################################
############################## Model Training #################################
###############################################################################

# Splitting the data for model
train <- all_dat[1:nrow(dat_train), ]
test <- all_dat[-(1:nrow(dat_train)), ]


#Building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
			        eval_metric = "auc",
              nthread=2,
			        eta=0.02,
			        max_depth=5,
			        colsample_bytree=0.7,
			        subsample=0.7)

#Parameter values are obtained from cross-validation
#445
xgbcv <- xgb.cv(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                nrounds = 550,
                label=train$TARGET,
                nfold=5,
                params = param,
                verbose = 2,
                maximize=FALSE)
                
xgbmodel <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                    label=train$TARGET,
                    nrounds = 550,
                    params = param,
                    verbose=2,
                    maximize = F
                    )

#Prediction
preds <- predict(xgbmodel, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]))
submission <- data.frame(ID = test$ID, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
