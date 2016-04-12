install.packages('caret', repos='http://cran.us.r-project.org', lib= "~/")
install.packages('dplyr', repos='http://cran.us.r-project.org', lib= "~/")
install.packages("xgboost", repos="https://cran.rstudio.com", lib= "~/")
library(caret, lib.loc = "~/")
library(dplyr, lib.loc = "~/")
library(xgboost, lib.loc= "~/")
library(Matrix)

# dat_train <- read.csv("../input/train.csv", stringsAsFactors = F)
# dat_test <- read.csv("../input/test.csv", stringsAsFactors = F)
train <- read.csv("train.csv", stringsAsFactors = FALSE)
#dat_test <- read.csv("test.csv", stringsAsFactors = FALSE)

#dat_test$TARGET <- -1

# merging the test and train data
#all_dat <- rbind(dat_train, dat_test)

# standardize missing values
train[train$var3 == -999999, "var3"] <- NA
delta_vars <- names(train)[grep('^delta', names(train))]
for(i in delta_vars){
  train[train[, i] == 9999999999, i] <- NA
}
train[train$var36 == 99, "var36"] <- NA
train[train$num_var12_0 == 111, "num_var12_0"] <- NA

# removing the constant variables
zeroVar <- nearZeroVar(train, saveMetrics = TRUE, freqCut = (nrow(train) - 10)/10,uniqueCut = 1000/nrow(train))
cat(names(train)[zeroVar[,"nzv"]], sep="\n")
train <- train[, !zeroVar[, "nzv"]]

#Removing duplicate columns
temp <- names(train)[duplicated(lapply(train, summary))]
cat(temp, sep="\n")
train <- train[, !names(train) %in% temp]

#Removing highly correlated variables
cor_v <- abs(cor(train))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_v <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
cat(names(train)[unique(cor_v$row)], sep="\n")
train <- train[,-unique(cor_v$row)]

# treat age variable
# all_dat$ageDiscrete <- NA
# all_dat[all_dat$age < 18, "ageDiscrete" ] <- "below.18"
# all_dat[all_dat$age >= 18 &  all_dat$age < 28, "ageDiscrete"] <- "18.to.25"
# all_dat[all_dat$age >= 28 & all_dat$age < 40, "ageDiscrete" ] <- "28.to.40"
# all_dat[all_dat$age >= 40 & all_dat$age < 70, "ageDiscrete"] <- "40.to.70"
# all_dat[all_dat$age >= 70, "ageDiscrete"] <- "70.above"
# all_dat$ageDiscrete <- as.factor(all_dat$ageDiscrete)

# convert to categorical
train$var36 <- as.factor(train$var36)

train$ind_count <- apply(train[, grep('^ind', names(train))], 1, function(x)(sum(x == 0)))

# balance/saldo zero or less
train$saldo0 <- apply(train[, grep('^saldo', names(train))], 1, function(x)(sum(x < 0)))
train$spain30 <- (train$var15 > 30 & train$var3 == 2) * 1

dummies <- dummyVars(~var36, data=train)
ohe <- as.data.frame(predict(dummies, newdata=train))
train <- cbind(train[, ! names(train) %in% c('var36')], ohe)

# standardize NA to -9999 (required for dmatrix)
train[is.na(train$var3), "var3"] <- -9999
delta_vars <- names(train)[grep('^delta', names(train))]
for(i in delta_vars){
  train[is.na(train[, i]), i] <- -9999
}
var36s <- names(train)[grep('var36', names(train))]
for(i in var36s){
  train[is.na(train[, i]), i] <- -9999
}
train[is.na(train$num_var12_0), "num_var12_0"] <- -9999
train[is.na(train$spain30), "spain30"] <- -9999

#train <- train[train$ID %in% dat_train$ID, ]
#test <- train[train$ID %in% dat_test$ID, ]

y.train <- train$TARGET
train$ID <- NULL
train <- sparse.model.matrix(TARGET ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=train, label=y.train, missing=-9999)

#ID.test <- test$ID
#test$ID <- NULL
#test <- sparse.model.matrix(TARGET ~. -1, data=test)

#source("nelder_mead.R")
xgb_wrap_obj <- function(param_fixed, vtx, data=dtrain, nfolds = 5, pred = FALSE, verb = FALSE ){ 
  return(  -xgb.cv( param=param_fixed, nround = vtx["nround"],
                    max_depth = vtx["max_depth"], eta = vtx["eta"],
                    data= dtrain, nfold = nfolds, 
                    stratified = TRUE, prediction=pred, verbose=verb)$test.auc.mean[vtx["nround"]] )
}

xgb_bdry = function(vtx = vector(0), ind_int = character(0), ind_num = character(0), min_int = integer(0)) {
  vtx[ind_int[1]] <- exp(vtx[ind_int[1]])
  vtx[ind_int[2]] <- exp(vtx[ind_int[2]])
  #to ensure that integer outputs are in their appropriate integer format
  vtx[ind_num] <- exp(vtx[ind_num])/(1+exp(vtx[ind_num]))
  vtx[ind_int][ vtx[ind_int] < min_int ] <- min_int
  vtx[ind_int] <- round(vtx[ind_int])
  return(vtx)
}

xgb_wrap_obj2 <- function(param_fixed, vtx, nround_ipt, data=dtrain, nfolds = 5, pred = FALSE, verb = FALSE ){ 
  return(  -xgb.cv( param=param_fixed, nround = nround_ipt,
                    gamma = vtx["max_depth"], colsample_bytree = vtx["colsample_bytree"], min_child_weight = vtx["min_child_weight"],
                    data= dtrain, nfold = nfolds, 
                    stratified = TRUE, prediction=pred, verbose=verb)$test.auc.mean[nround_ipt] )
}

xgb_bdry2 = function(vtx = vector(0), ind_num = character(0)) {
  vtx[ind_num[1]] <- exp(vtx[ind_num[1]])
  vtx[ind_num[2]] <- exp(vtx[ind_num[2]])/(1+exp(vtx[ind_num[2]]))
  vtx[ind_num[3]] <- exp(vtx[ind_num[3]])
  return(vtx)
}

xgb_wrap_obj3 <- function(param_fixed, vtx, data=dtrain, nfolds = 5, pred = FALSE, verb = FALSE ){ 
  return(  -xgb.cv( param=param_fixed, nround = vtx["nround"],
                    max_depth = vtx["max_depth"], 
                    eta = vtx["eta"],
                    gamma = vtx["gamma"], 
                    colsample_bytree = vtx["colsample_bytree"], 
                    min_child_weight = vtx["min_child_weight"],
                    subsample = vtx["subsample"],
                    data= dtrain, nfold = nfolds, 
                    stratified = TRUE, prediction=pred, verbose=verb, missing = NA)$test.auc.mean[vtx["nround"]] )
}

xgb_bdry3 = function(vtx = vector(0), ind_int = character(0), ind_num = character(0), min_int = integer(0)) {
  #mapping integer outputs
  vtx[ind_int[1]] <- exp(vtx[ind_int[1]])
  vtx[ind_int[2]] <- exp(vtx[ind_int[2]])
  vtx[ind_int][ vtx[ind_int] < min_int ] <- min_int
  vtx[ind_int] <- round(vtx[ind_int])
  #mapping numeric outputs
  vtx[ind_num[1]] <- exp(vtx[ind_num[1]])/(1+exp(vtx[ind_num[1]]))
  vtx[ind_num[2]] <- exp(vtx[ind_num[2]])
  vtx[ind_num[3]] <- exp(vtx[ind_num[3]])/(1+exp(vtx[ind_num[3]]))
  vtx[ind_num[4]] <- exp(vtx[ind_num[4]])
  vtx[ind_num[5]] <- exp(vtx[ind_num[5]])/(1+exp(vtx[ind_num[5]]))
  return(vtx)
}




#----------------------user inputs----------------------------------------------------------------

NM_opt <- function( vtcs_init,        #initialiation of Nelder Mead vertices,
                    obj_fun,          #objective function
                    fxd_obj_param,    #objective function fixed parameter inputs
                    bdry_fun = NULL, fxd_bdry_param = NULL,
                    max_iter = 200, max_0prgrss = 10,
                    a_r = 1, a_e = 2, a_c = 0.5, a_s = 0.5) {
  
  #-------------------------------------------------------------------------------------
  #Description of Nelder Mead parameters
  #a_r -> reflect_cstnt
  #a_e -> expand_cstnt
  #a_c -> ctrt_cstnt
  #a_s -> shrink_cstnt
  
  #Nelder Mead specific function
  centroid <- function(all_vertices, worst_vertex) {
    output <- ( apply(all_vertices,2,sum) - worst_vertex ) / ( nrow(all_vertices) - 1 ) 
    output[1] <- round(output[1]) 
    output[2] <- round(output[2])
    return(output)
  }
  #-------------------------------------------------------------------------------------
  
  #checks if obj_fun is a function and passing 
  if (!is.function(obj_fun)) stop("input to obj_fun is not a function")
  
  #checks if unconstrained to constrained mapping is considered.
  if (!is.null(bdry_fun))  {
    boundary <- TRUE
    #checks if bdry_fun is a function
    if (!is.function(bdry_fun)) stop("input to bdry_fun is not a function")
    bdry_fun_args <- c(list(vtx = NULL),fxd_bdry_param)
  }
  
  #checks if obj_fun is a function and passing 
  if (!is.function(obj_fun)) stop("input to obj_fun is not a function")
  
  #checks if the vertices input are valid 
  if( !(is.matrix(vtcs_init) | is.data.frame(vtcs_init)) ) stop("vtcs_init format should be a matrix or data.frame")
  if( !length(dim(vtcs_init)) == 2 ) stop("dimensions of vtcs_init should be 4 vertices x 3 fields")
  if( !( (nrow(vtcs_init)- ncol(vtcs_init)) == 1) )  stop("[1] if optimization dimensions is N, number of vertices should be N + 1 \n
                                                          [2] please ensure that rows represent the vertices and columns the optization parameters")
  
  
  #passing in Nelder Mead vertices input and objtive function fixed parameters
  obj_fun_args <- c(list(vtx = NULL), fxd_obj_param)
  
  
  #passing in intialization parameters
  vtcs <- cbind( obj = rep(0,nrow(vtcs_init)), vtcs_init )
  
  #Miscellaneous
  N_vtcs <- nrow(vtcs)
  var_names <- colnames(vtcs)[!( colnames(vtcs) %in% c("obj") )]
  
  #initialization of vertices
  for ( i in 1:N_vtcs)  {
    if (boundary == TRUE) {
      bdry_fun_args[["vtx"]] <- vtcs[i, var_names]
      obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
    }
    vtcs[i,"obj"] <- do.call( obj_fun, obj_fun_args )
  }
  vtcs <- vtcs[order(vtcs[,"obj"]),] # sort from best to worst
  
  
  #NM iteration initialization
  obj_tm1 <- obj_t <- 0
  N_shrink <- 0
  iter = 1
  iter_eq0 = 1
  
  
  #NM begins
  while ( (iter <= max_iter) & (iter_eq0 <= max_0prgrss) )  {
    path <- "reflection"
    X_w <- vtcs[N_vtcs,var_names] 
    C <- centroid(vtcs[,var_names], X_w ) #centroid of best hyperplane which is opposite the worst vertex 
    
    #---- Reflection step ----------------------------------------------------------------------------
    if (path == "reflection") {
      X_r <- a_r*(C - X_w) 
      if (boundary == TRUE) {
        bdry_fun_args[["vtx"]] <- X_r
        obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
      }
      obj_r <- do.call( obj_fun, obj_fun_args )
      #Relect if ( (vtcs[,"obj"] >= obj_r) & (coj_r > vtcs[,"obj"]) ) { 
      if ( (vtcs[1,"obj"] <= obj_r) & (obj_r < vtcs[2,"obj"]) ) { 
        vtcs[N_vtcs,] <- c(cbj = obj_r, X_r); 
        # path <- "stopping" 
      }        else if ( obj_r < vtcs[1,"obj"] ) { #go to expansion
        path <- "expansion" 
      }        else {
        path <- "contraction"
      }
    } # ---------------------------------------------------------------------------------------------
    
    
    #---- Expansion Step ----------------------------------------------------------------------------
    if (path == "expansion") {
      X_e <- C + a_e*(X_r - C)
      if (boundary == TRUE) {
        bdry_fun_args[["vtx"]] <- X_e
        obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
      }
      obj_e <- do.call( obj_fun, obj_fun_args )
      if (obj_e < obj_r)       {
        vtcs[N_vtcs,] <- c(obj = obj_e, X_e) 
      } else { 
        vtcs[N_vtcs,] <- c(obj = obj_r, X_r)
      }
    } # ---------------------------------------------------------------------------------------------
    
    
    #---- Contraction Step ---------------------------------------------------------------------------- 
    if (path == "contraction") { 
      if ( (vtcs[2,"obj"] <= obj_r) & (obj_r < vtcs[N_vtcs,"obj"]) ) { # g(X_bad) <= g(X_r) < g(X_worst) 
        #outer contraction 
        X_o <- C + a_c*(X_r - C)
        if (boundary == TRUE) {
          bdry_fun_args[["vtx"]] <- X_o
          obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
        }
        obj_o <- do.call( obj_fun, obj_fun_args )
        if (obj_o <= obj_r) { #g(X_o) <= g(X_r) 
          vtcs[N_vtcs, ] <- c( obj = obj_o, X_o) 
          #path <- "stopping" 
        } else { 
          path <- "shrinking" 
        }
      } else if (vtcs[N_vtcs,"obj"] <= obj_r) {# g(X_worst) <= g(X_r) 
        #inner contraction 
        X_i <- C + a_c*(X_w - C)
        if (boundary == TRUE) {
          bdry_fun_args[["vtx"]] <- X_i
          obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
        }
        obj_i <- do.call( obj_fun, obj_fun_args )
        if (obj_i <= vtcs[N_vtcs,"obj"]) { #g(x_o) <= g(x_r) 
          vtcs[N_vtcs, ] <- c( obj = obj_i, X_i) 
          #path <- "stopping" 
        } else { 
          path <- "shrinking" 
        } 
      }
    } # ---------------------------------------------------------------------------------------------
    
    # Shrinking Step --------------------------------------------------------------------------------
    if (path == "shrinking") {
      idx_s <- which(!vtcs[,"obj"] %in% vtcs[1,"obj"])
      X_best <- t(matrix( rep(vtcs[1,var_names], length(idx_s)), nrow = N_vtcs - 1, ncol = length(idx_s) ))
      vtcs[idx_s, var_names] <- X_best + a_s*(vtcs[idx_s,var_names] - X_best)
      for (i in idx_s) {
        if (boundary == TRUE) {
          bdry_fun_args[["vtx"]] <- vtcs[i, var_names]
          obj_fun_args[["vtx"]] <- do.call( bdry_fun, bdry_fun_args )
        }
        vtcs[i,"obj"] <- do.call( obj_fun, obj_fun_args )
      }
    }
    # -----------------------------------------------------------------------------------------------
    
    
    #---- Stopping Step -----------------------------------------------------------------------------
    vtcs <- vtcs[ order(vtcs[,"obj"]),]
    obj_t <- vtcs[1,"obj"]
    flush.console()
    print( obj_t - obj_tm1 )
    print(path)
    print(vtcs)
    print(iter)
    if ( (obj_t - obj_tm1) != 0 )       {
      iter_eq0 = 0
    }          else {
      iter_eq0 = iter_eq0 + 1
    }
    obj_tm1 <- obj_t
    iter = iter + 1
    
  } # ---------------------------------------------------------------------------------------------- 
  return(vtcs)
}

vtcs <- NM_opt( vtcs_init = cbind(nround = log(c(200,500,700,1000,900,600,400,800)), 
                                  max_depth = log(c(14,12,6,10,4,9,5,10)), 
                                  eta = -log(1/c(0.2, 0.05, 0.09, 0.005,0.3,0.02,0.5,0.03) - 1 +1e-05),
                                  gamma = log(c(0.01,10,2,5,5,0.5,2,7)),
                                  colsample_bytree = -log(1/c(0.2,0.3,0.4,0.9,0.5,0.3,0.7,0.8) - 1 +1e-05),
                                  min_child_weight = log(c(4,3,4,9,2,10,3,10)),
                                  subsample = -log(1/c(0.5,0.3,0.9,0.4,0.3,0.5,0.8,0.1) - 1 +1e-05)),
                obj_fun = xgb_wrap_obj3,
                fxd_obj_param  = list(param = list("nthread" = 8,   # number of threads to be used 
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

write.csv(vtcs, "vtcs.csv")