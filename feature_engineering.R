library(caret)
library(class)

dat_train <- read.csv("train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- -1

# merging the test and train data
all_dat <- rbind(dat_train, dat_test)

# count no. of nonzero elements
all_dat$zero <- apply(all_dat[, !names(all_dat) %in% c('TARGET', 'ID')], 1, function(x) (sum(x == 0)))
all_dat$variance <- apply(all_dat[, !names(all_dat) %in% c('TARGET', 'ID')], 1, function(x)var(x))
all_dat$rowSums <- rowSums(all_dat[, !names(all_dat) %in% c('TARGET', 'ID')])
 
# one hot encoding on var36
all_dat$var36 <- as.factor(all_dat$var36)
dummies <- dummyVars(~var36, data=all_dat)
var36s <- predict(dummies, newdata=all_dat)
all_dat <- cbind(all_dat[, !names(all_dat) == 'var36'], var36s)

# break var38 into most common value and others
all_dat$var38mc <- (all_dat$var38 != (names(sort(table(all_dat$var38), decreasing=TRUE))[1])) * 1
all_dat$logvar38 <- log(all_dat$var38) * all_dat$var38mc
all_dat$var38 <- NULL

# removing the constant variables
zeroVar <- nearZeroVar(all_dat, saveMetrics = TRUE, freqCut = (nrow(all_dat) - 10)/10,uniqueCut = 1000/nrow(all_dat))
cat(names(all_dat)[zeroVar[,"zeroVar"]], sep="\n")
all_dat <- all_dat[, !zeroVar[, "zeroVar"]]

# delete all delta variables
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[, i] <- NULL
}

#Removing duplicate columns
temp <- names(all_dat)[duplicated(lapply(all_dat, summary))]
cat(temp, sep="\n")
all_dat <- all_dat[, !names(all_dat) %in% temp]

# ratios
all_dat$var5_diff <- all_dat$saldo_medio_var5_hace2 - all_dat$saldo_medio_var5_hace3
all_dat$var45_ratio <- log((all_dat$num_var45_hace2 + 1)/(all_dat$num_var45_hace3 + 1))
#all_dat$var22_ratio <- log((all_dat$num_var22_hace2 + 1)/(all_dat$num_var22_hace3 + 1))
#all_dat$var13_corto_ratio <- (all_dat$saldo_medio_var13_corto_hace2 + 1)/(all_dat$saldo_medio_var13_corto_hace3 + 1)
#all_dat$var13_largo_ratio <-  (all_dat$saldo_medio_var13_largo_hace2 + 1)/(all_dat$saldo_medio_var13_largo_hace3 + 1)
#all_dat$var8_ratio <- (all_dat$saldo_medio_var8_hace2 + 1)/(all_dat$saldo_medio_var8_hace3 + 1)

# adding mean of TARGET for num_var30, num_var5, num_var42, var15
num_var30 <- tapply(dat_train$TARGET, dat_train$num_var30, mean)
all_dat$num_var30 <- num_var30[as.character(all_dat$num_var30)]
all_dat[is.na(all_dat$num_var30), "num_var30"] <- 0
# 
# num_var5 <- tapply(dat_train$TARGET, dat_train$num_var5, mean)
# all_dat$num_var5 <- num_var5[as.character(all_dat$num_var5)]
# all_dat[is.na(all_dat$num_var5), "num_var5"] <- 0
# 
# num_var42 <- tapply(dat_train$TARGET, dat_train$num_var42, mean)
# all_dat$num_var42 <- num_var42[as.character(all_dat$num_var42)]
# all_dat[is.na(all_dat$num_var42), "num_var42"] <- 0

age_propensity <- tapply(dat_train$TARGET, dat_train$var15, mean)
all_dat$age_propensity <- age_propensity[as.character(all_dat$var15)]
all_dat[is.na(all_dat$age_propensity), "age_propensity"] <- 0

# add k-nearest neighbor
normalize <- function(x){
    num <- x - min(x)
    denom <- max(x) - min(x)
    return (num/denom)
}
feature_list <- c("var15", "saldo_var30", "zero", "variance", "logvar38", "var45_ratio",
                  "age_propensity", "var5_diff", "rowSums", "saldo_medio_var5_ult1",
                  "saldo_medio_var5_ult3")
all_dat_norm <- as.data.frame(lapply(all_dat[feature_list], normalize))
dat_train_norm <- all_dat_norm[1:nrow(dat_train),]
dat_test_norm <- all_dat_norm[-(1:nrow(dat_train)),]
inTrain <- createDataPartition(dat_train$TARGET, p=0.5, list=TRUE)
y.training_norm <- dat_train$TARGET[inTrain[[1]]]
y.testing_norm <- dat_train$TARGET[-inTrain[[1]]]
training_norm <- dat_train_norm[inTrain[[1]],]
testing_norm <- dat_train_norm[-inTrain[[1]],]
dat_train_norm[-inTrain[[1]], "knn"] <- knn(train=training_norm, test=testing_norm, cl=y.training_norm, k =5)
dat_train_norm[inTrain[[1]], "knn"] <- knn(train=testing_norm, test=training_norm, cl=y.testing_norm, k=5)
all_dat[all_dat$ID %in% dat_test$ID, "knn"] <- knn(train=all_dat_norm[1:nrow(dat_train),],
                                                   test=all_dat_norm[-(1:nrow(dat_train)),],
                                                   cl=dat_train$TARGET,
                                                   k=5)
all_dat[all_dat$ID %in% dat_train$ID, "knn"] <- dat_train_norm$knn