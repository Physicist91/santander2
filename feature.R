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
zeroVar <- nearZeroVar(all_dat, saveMetrics = TRUE, freqCut = (nrow(all_dat) - 10)/10,uniqueCut = 1000/nrow(all_dat))
cat(names(all_dat)[zeroVar[,"nzv"]], sep="\n")
all_dat <- all_dat[, !zeroVar[, "nzv"]]

#Removing duplicate columns
temp <- names(all_dat)[duplicated(lapply(all_dat, summary))]
cat(temp, sep="\n")
all_dat <- all_dat[, !names(all_dat) %in% temp]

# convert to categorical
all_dat$var36 <- as.factor(all_dat$var36)

# count no. of nonzero elements
all_dat$nonzero <- apply(all_dat, 1, function(x) (sum(x != 0, na.rm=TRUE)))

# another simple count
all_dat$ind_count <- apply(all_dat[, grep('^ind', names(all_dat))], 1, function(x)(sum(x == 0)))

# balance/saldo zero or less
all_dat$saldo0 <- apply(all_dat[, grep('^saldo', names(all_dat))], 1, function(x)(sum(x < 0)))
all_dat$spain30 <- (all_dat$var15 > 30 & all_dat$var3 == 2) * 1

# ratios
all_dat$var5_ratio <- (all_dat$saldo_medio_var5_hace2 + 1)/(all_dat$saldo_medio_var5_hace3 + 1)
