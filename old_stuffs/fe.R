#################################################################
## Feature Engineering (although mainly data cleaning)
#################################################################

# Reading data

training <- read.csv("train.csv", stringsAsFactors = FALSE)
testing <- read.csv("test.csv", stringsAsFactors = FALSE)

testing$TARGET <- -1
all_data <- rbind(training, testing)

# divide var3 into two: Spain vs not Spain.
all_data$var3 <- (all_data$var3 == 2) * 1

# num_var12_0
all_data$num_var12_0 <- (all_data$num_var12_0 == 0) * 1

#Remove unimportant delta variables
all_data$var13_delta_num <- all_data$delta_num_aport_var13_1y3
all_data$var13_delta_imp <- all_data$delta_imp_aport_var13_1y3
all_data <- all_data[, -grep('^delta', names(all_data))]

#Remove duplicate columns
temp <- names(all_data)[duplicated(lapply(all_data, summary))]
cat(temp, sep="\n")
all_data <- all_data[, !names(all_data) %in% temp]

#Remove variables with zero variance
nzv <- nearZeroVar(all_data, saveMetrics = TRUE, freqCut = (nrow(all_data) - 10)/10,uniqueCut = 1000/nrow(all_data))
cat(names(all_data)[nzv[,"nzv"]], sep="\n")
all_data <- all_data[, !nzv[, "nzv"]]