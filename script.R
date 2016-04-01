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

dat_test$TARGET <- -1


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
#This prevents overfitting
cor_v <- abs(cor(all_dat))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat("Removing ", names(all_dat)[unique(cor_v$row)])
all_dat <- all_dat[,-unique(cor_v$row)]

#Standardize missing values
all_dat[all_dat$var3 == -999999, "var3"] <- NA
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[all_dat[, i] == 9999999999, i] <- NA
}

#Renaming known variables
all_dat <- rename(all_dat, age=var15)

#Transforming age variable by one-hot-encoding
all_dat$ageDiscrete <- as.factor(round(all_dat$age/10, 0))
dummies <- dummyVars(~ ageDiscrete, data=all_dat)
ohe <- as.data.frame(predict(dummies, newdata=all_dat))
all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('ageDiscrete')], ohe)

#Transforming '^num' vars to average response
num_vars <- names(all_dat)[ grep('^num_var', names(all_dat))]
num_vars
for(i in num_vars){
  temp <- aggregate(formula(paste0("TARGET ~ ", i)), data=all_dat, FUN=mean)
  colnames(temp) <- c(i, paste0(i, "_kevin"))
  temp[temp[, 2] == -1, 2] <- NA 
  all_dat <- merge(all_dat, temp, by=i)
  all_dat[, i] <- NULL
}

###############################################################################
############################## Model Training #################################
###############################################################################

# Splitting the data for model
train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]


#Building the model
param <- list(objective = "binary:logistic",
              booster = "gbtree",
			        eval_metric = "auc",
              nthread=2,
			        eta=0.02,
			        max_depth=5,
			        colsample_bytree=0.85,
			        subsample=0.95)

#Parameter values are obtained from cross-validation
xgbcv <- xgb.cv(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
                nrounds = 550,
                label=train$TARGET,
                nfold=5,
                params = param,
                verbose = 2,
                maximize=T,
                missing = NA)
                
# xgbmodel <- xgboost(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]),
#                     label=train$TARGET,
#                     nrounds = 550,
#                     params = param,
#                     verbose=2,
#                     maximize = T,
#                     missing=NA)

#Prediction
preds <- predict(xgbmodel, newdata = data.matrix(test[, ! names(test) %in% c("ID", "TARGET")]))
submission <- data.frame(ID = test$ID, TARGET = preds)

write.csv(submission, "submission.csv", row.names = FALSE)
