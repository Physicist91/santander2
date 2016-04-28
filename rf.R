library("randomForest")
library('pROC')

source("feature.R")

# splitting the data for model
training <- all_dat[all_dat$ID %in% dat_train$ID, ]
testing <- all_dat[all_dat$ID %in% dat_test$ID, ]

training$ID <- NULL
training$TARGET <- as.factor(ifelse(training$TARGET == 0, 'S', 'U'))

fitControl <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary, verboseIter=TRUE)

rfmodel <- train(TARGET ~ ., data=training, method='rf', trControl=fitControl, ntree= 101, verbose=TRUE, metric='ROC')


preds_rf <- predict(rfmodel, newdata=test[, !names(test) %in% c("ID", "TARGET")], type='vote')

