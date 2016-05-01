result <- data.frame(eta=double(), max.depth=integer(), nrounds=integer(), auc=double())

for(stepsize in c(0.01, 0.015, 0.02, 0.025, 0.03, 0.035)){
    
    for(depth in 4:8){
        print(paste0("Running xgb.cv with stepsize ", stepsize, " and maxdepth ", depth, "...\n"))
        set.seed(88)
        xgbcv <- xgb.cv(data = as.matrix(train[, !names(train) %in% c("ID", "TARGET")]), params = param,
                        label=train$TARGET, nrounds = 4000, max.depth = depth, eta = stepsize, maximize = F, missing=NA, nfold=5,
                        early.stop.round=1600)
        best <- max(xgbcv$test.auc.mean)[1]
        new <- setNames(as.list(c(stepsize, depth, which(xgbcv$test.auc.mean==best)[1], best)), names(result))
        result <- rbind(result, new)
        result
        plot(1:nrow(xgbcv), xgbcv$test.auc.mean, main=paste0("Stepsize = ", stepsize, " and maxdepth = ", depth))
    }
}
