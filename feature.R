library(caret)
library(dplyr)
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
# all_dat$ageDiscrete <- NA
# all_dat[all_dat$age < 18, "ageDiscrete" ] <- "below.18"
# all_dat[all_dat$age >= 18 &  all_dat$age < 28, "ageDiscrete"] <- "18.to.25"
# all_dat[all_dat$age >= 28 & all_dat$age < 40, "ageDiscrete" ] <- "28.to.40"
# all_dat[all_dat$age >= 40 & all_dat$age < 70, "ageDiscrete"] <- "40.to.70"
# all_dat[all_dat$age >= 70, "ageDiscrete"] <- "70.above"
# all_dat$ageDiscrete <- as.factor(all_dat$ageDiscrete)

# convert to categorical
all_dat$var36 <- as.factor(all_dat$var36)

# count of ind
all_dat$ind_count <- rowSums(all_dat[, grep('ind', names(all_dat))])

# balance/saldo zero or less
all_dat$saldo0 <- apply(all_dat[, grep('saldo', names(all_dat))], 1, function(x)(sum(x < 10)))
all_dat$spain30 <- (all_dat$var15 > 30 & all_dat$var3 == 2) * 1
