library("randomForest")

source("feature.R")

# splitting the data for model
training <- all_dat[all_dat$ID %in% dat_train$ID, ]
testing <- all_dat[all_dat$ID %in% dat_test$ID, ]

training$ID <- NULL
training$TARGET <- as.factor(training$TARGET)

rfmodel <- randomForest(TARGET ~ ., data=training, ntree= 101, do.trace=TRUE, mtry=145)

preds <- predict(rfmodel, newdata=test[, !names(test) %in% c("ID", "TARGET")], type='vote')
preds
