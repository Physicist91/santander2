source("xgboost.R")
source("nelder_mead.R")

vtcs <- NM_opt( vtcs_init = cbind(nround = log(c(200,500,700,1000,900,600,400)), 
                                  max_depth = log(c(14,12,6,10,4,9,5,10)), 
                                  eta = -log(1/c(0.2, 0.05, 0.09, 0.005,0.3,0.02,0.5,0.03) - 1 +1e-05),
                                  gamma = log(c(0.01,10,2,5,5,0.5,2,7)),
                                  colsample_bytree = -log(1/c(0.2,0.3,0.4,0.9,0.5,0.3,0.7,0.8) - 1 +1e-05),
                                  min_child_weight = log(c(4,3,4,9,2,10,3,10)),
                                  subsample = -log(1/c(0.5,0.3,0.9,0.4,0.3,0.5,0.8,0.1) - 1 +1e-05)),
                obj_fun = xgb_wrap_obj3,
                fxd_obj_param  = list(param = list("nthread" = 2,   # number of threads to be used 
                                                   "objective" = "binary:logistic",    # binary classification 
                                                   "eval_metric" ="auc"    # evaluation metric
                                                    ),
                                  data=dtrain),
                bdry_fun = xgb_bdry3,
                fxd_bdry_param = list( ind_int = c("nround","max_depth"),
                                       ind_num = c("eta","gamma", "colsample_bytree", "min_child_weight","subsample"), 
                                       min_int = 1 ), 
                a_e = 2.7,
                max_0prgrss = 15)


#-----------------------------------------------

fxd_bdry_param = list( ind_int = c("nround","max_depth"),
                       ind_num = c("eta","gamma", "colsample_bytree", "min_child_weight","subsample"), 
                       min_int = 1 )

t(apply(vtcs[,2:8],1, function(x) do.call("xgb_bdry3", c(list(vtx = x), fxd_bdry_param))))
