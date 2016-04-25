library(Matrix)
library(caret)

dat_train <- read.csv("train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- -1

# merging the test and train data
all_dat <- rbind(dat_train, dat_test)

# count no. of nonzero elements
all_dat$nonzero <- apply(all_dat[, !names(all_dat) == 'TARGET'], 1, function(x) (sum(x != 0)))

# change var3 to either 2 (Spain) or not 2
all_dat$is_spain <- ifelse(all_dat$var3 == 2, 1, 0)

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
all_dat$var5_ratio <- (all_dat$saldo_medio_var5_hace2 + 1)/(all_dat$saldo_medio_var5_hace3 + 1)
