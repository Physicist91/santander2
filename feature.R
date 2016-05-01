library(Matrix)
library(caret)

dat_train <- read.csv("train.csv", stringsAsFactors = FALSE)
dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

dat_test$TARGET <- -1

# merging the test and train data
all_dat <- rbind(dat_train, dat_test)

# count no. of nonzero elements
all_dat$nonzero <- apply(all_dat[, !names(all_dat) %in% c('TARGET', 'ID')], 1, function(x) (sum(x != 0)))
all_dat$variance <- apply(all_dat[, !names(all_dat) %in% c('TARGET', 'ID')], 1, function(x) var(x))

# change some variables to flags
all_dat$is_spain <- ifelse(all_dat$var3 == 2, 1, 0)
all_dat$num_var12_0 <- (all_dat$num_var12_0 == 0) * 1

# one-hot-encoding on var 36
all_dat$var36 <- as.factor(all_dat$var36)
dummies <- dummyVars( ~ var36, data=all_dat)
var36s <- predict(dummies, newdata=all_dat)
all_dat <- cbind(all_dat[, !names(all_dat) == 'var36'], var36s)

# log transform
all_dat$var38 <- log(all_dat$var38)

# rescaling
# rescale <- function(x){
#   (x - min(x))/(max(x) - min(x))
# }
# all_dat$saldo_medio_var5 <- log(rescale(all_dat$saldo_medio_var5_hace2 - all_dat$saldo_medio_var5_hace3))
# all_dat$saldo_medio_var8 <- log(rescale(all_dat$saldo_medio_var8_hace2 - all_dat$saldo_medio_var8_hace3))
# all_dat$saldo_medio_var5_u <- log(rescale(all_dat$saldo_medio_var5_ult1 - all_dat$saldo_medio_var5_ult3))


# removing the constant variables
zeroVar <- nearZeroVar(all_dat, saveMetrics = TRUE, freqCut = (nrow(all_dat) - 10)/10,uniqueCut = 1000/nrow(all_dat))
cat(names(all_dat)[zeroVar[,"zeroVar"]], sep="\n")
all_dat <- all_dat[, !zeroVar[, "zeroVar"]]

# delete all delta variables
all_dat$var13_delta_num <- all_dat$delta_num_aport_var13_1y3
all_dat$var13_delta_imp <- all_dat$delta_imp_aport_var13_1y3
delta_vars <- names(all_dat)[grep('^delta', names(all_dat))]
for(i in delta_vars){
  all_dat[, i] <- NULL
}

#Removing duplicate columns
temp <- names(all_dat)[duplicated(lapply(all_dat, summary))]
cat(temp, sep="\n")
all_dat <- all_dat[, !names(all_dat) %in% temp]

# ratios
all_dat$var45_ratio <- log((all_dat$num_var45_hace2 + 1)/(all_dat$num_var45_hace3 + 1))
all_dat$var22_ratio <- log((all_dat$num_var22_hace2 + 1)/(all_dat$num_var22_hace3 + 1))
