install.packages('caret', repos='http://cran.us.r-project.org', lib= "~/")
install.packages('dplyr', repos='http://cran.us.r-project.org', lib= "~/")
install.packages("xgboost", repos="https://cran.rstudio.com", lib= "~/")
library(caret, lib.loc = "~/")
library(dplyr, lib.loc = "~/")
library(xgboost, lib.loc= "~/")
library(Matrix)

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
all_dat[all_dat$num_var12_0 == 111, "num_var12_0"] <- NA

# removing the constant variables
zeroVar <- nearZeroVar(all_dat, saveMetrics = TRUE, freqCut = (nrow(all_dat) - 10)/10,uniqueCut = 1000/nrow(all_dat))
cat(names(all_dat)[zeroVar[,"nzv"]], sep="\n")
all_dat <- all_dat[, !zeroVar[, "nzv"]]

#Removing duplicate columns
temp <- names(all_dat)[duplicated(lapply(all_dat, summary))]
cat(temp, sep="\n")
all_dat <- all_dat[, !names(all_dat) %in% temp]

#Removing highly correlated variables
cor_v <- abs(cor(all_dat))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat(names(all_dat)[unique(cor_v$row)], sep="\n")
all_dat <- all_dat[,-unique(cor_v$row)]

# treat age variable
# all_dat$ageDiscrete <- NA
# all_dat[all_dat$age < 18, "ageDiscrete" ] <- "below.18"
# all_dat[all_dat$age >= 18 &  all_dat$age < 28, "ageDiscrete"] <- "18.to.25"
# all_dat[all_dat$age >= 28 & all_dat$age < 40, "ageDiscrete" ] <- "28.to.40"
# all_dat[all_dat$age >= 40 & all_dat$age < 70, "ageDiscrete"] <- "40.to.70"
# all_dat[all_dat$age >= 70, "ageDiscrete"] <- "70.above"
# all_dat$ageDiscrete <- as.factor(all_dat$ageDiscrete)

# convert to categorical
all_dat$var36 <- as.factor(all_dat$var36)

all_dat$ind_count <- apply(all_dat[, grep('^ind', names(all_dat))], 1, function(x)(sum(x == 0)))

# balance/saldo zero or less
all_dat$saldo0 <- apply(all_dat[, grep('^saldo', names(all_dat))], 1, function(x)(sum(x < 0)))
all_dat$spain30 <- (all_dat$var15 > 30 & all_dat$var3 == 2) * 1

dummies <- dummyVars(~var36, data=all_dat)
ohe <- as.data.frame(predict(dummies, newdata=all_dat))
all_dat <- cbind(all_dat[, ! names(all_dat) %in% c('var36')], ohe)

# standardize NA to -9999 (required for dmatrix)
all_dat[is.na(all_dat$var3), "var3"] <- -9999
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}
var36s <- names(all_dat)[grep('var36', names(all_dat))]
for(i in var36s){
  all_dat[is.na(all_dat[, i]), i] <- -9999
}
all_dat[is.na(all_dat$num_var12_0), "num_var12_0"] <- -9999
all_dat[is.na(all_dat$spain30), "spain30"] <- -9999

train <- all_dat[all_dat$ID %in% dat_train$ID, ]
test <- all_dat[all_dat$ID %in% dat_test$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train, missing=-9999)

ID.test <- test$ID
test$ID <- NULL
test <- sparse.model.matrix(TARGET ~. -1, data=test)

source("nelder_mead.R")

vtcs <- NM_opt( vtcs_init = cbind(nround = log(c(200,500,700,1000,900,600,400,800)), 
                                  max_depth = log(c(14,12,6,10,4,9,5,10)), 
                                  eta = -log(1/c(0.2, 0.05, 0.09, 0.005,0.3,0.02,0.5,0.03) - 1 +1e-05),
                                  gamma = log(c(0.01,10,2,5,5,0.5,2,7)),
                                  colsample_bytree = -log(1/c(0.2,0.3,0.4,0.9,0.5,0.3,0.7,0.8) - 1 +1e-05),
                                  min_child_weight = log(c(4,3,4,9,2,10,3,10)),
                                  subsample = -log(1/c(0.5,0.3,0.9,0.4,0.3,0.5,0.8,0.1) - 1 +1e-05)),
                obj_fun = xgb_wrap_obj3,
                fxd_obj_param  = list(param = list("nthread" = 8,   # number of threads to be used 
                                                   "objective" = "binary:logistic",    # binary classification 
                                                   "eval_metric" ="auc"    # evaluation metric
                ),
                data=dtrain),
                bdry_fun = xgb_bdry3,
                fxd_bdry_param = list( ind_int = c("nround","max_depth"),
                                       ind_num = c("eta","gamma", "colsample_bytree", "min_child_weight","subsample"), 
                                       min_int = 1 ), 
                a_e = 2.7,
                max_0prgrss = 15)


#-----------------------------------------------

fxd_bdry_param = list( ind_int = c("nround","max_depth"),
                       ind_num = c("eta","gamma", "colsample_bytree", "min_child_weight","subsample"), 
                       min_int = 1 )

t(apply(vtcs[,2:8],1, function(x) do.call("xgb_bdry3", c(list(vtx = x), fxd_bdry_param))))

write.csv(vtcs, "vtcs.csv")
