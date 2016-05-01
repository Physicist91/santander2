#######################################################
## Only produce a root node.
#######################################################


source("feature.R")

library(rpart)

training <- all_dat[all_dat$ID %in% dat_train$ID, ]
testing <- all_dat[all_dat$ID %in% dat_test$ID, ]

training$ID <- NULL

tree_clf <- rpart(as.factor(TARGET) ~ var15 + saldo_var30 + var38, data=training, method='class', minsplit=10)

summary(tree_clf)
plot(tree_clf)
text(tree_clf)

preds_tree <- predict(tree_clf, training)

table(ifelse(preds_tree[, 2] > 0.5, 1, 0), training$TARGET)
