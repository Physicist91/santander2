


#setwd("C:\\Users\\ekhongl\\Documents\\CODES - R\\Kaggle - Classification Framework")


#Creating the Various data table types
library("Matrix")
dat = train



dat_mod = model.matrix(TARGET~.-1, data = dat, sparse = TRUE) #
dat_sparse = sparse.model.matrix(TARGET~.-1, data = dat)
dat_y = dat$TARGET


#creating the indices for out-of-fold validation
library(caret)
K_Folds_K = 3
K_Folds = createFolds(dat[,"TARGET"], k = K_Folds_K, list = TRUE, returnTrain = FALSE)
table(dat[K_Folds[[1]],"TARGET"])
table(dat[K_Folds[[2]],"TARGET"])
table(dat[K_Folds[[3]],"TARGET"])


#--------------------------------------------------------------------------------------------------------
#	[5] Modelling - NelderMead 
#--------------------------------------------------------------------------------------------------------
library("xgboost")
library(dplyr)


k <- 1

a_r <- 1
a_e <- 2
a_c <- 0.5
a_s <- 0.5

min_max = 0 # if max: min_max = 1, if min: min_max = 0
boundary = TRUE



# xgboost parameters
param_fixed <- list("num_class" = 3,
					"nthread" = 2,   # number of threads to be used 
					"objective" = "binary:logistic",    # binary classification 
					"eval_metric" ="auc",    # evaluation metric
					"gamma" = 0,    # minimum loss reduction 
					"subsample" = 0.9,    # part of data instances to grow tree 
					"colsample_bytree" = 0.9,  # subsample ratio of columns when constructing each tree 
					"min_child_weight" = 1  # minimum sum of instance weight needed in a child
					)

#xbg function wrap that only returns the objective function output 
xgb_wrap_obj <- function(param_fixed, vtx, dat_x, dat_y, nfolds = 5, pred = FALSE, verb = FALSE ){ 
    return(	-xgb.cv( param=param_fixed, nround = vtx["nround"],
                    max_depth = vtx["max_depth"], eta = vtx["eta"],
                    data= dat_x, label = dat_y, nfold = nfolds, 
                    stratified = TRUE, prediction=pred, verbose=verb)$test.auc.mean[vtx["nround"]] )
}

# =================================== ============== 
#Nelder Mead functions # =================================== ========================================= 
centroid <- function(all_vertices, worst_vertex) {
    output <- ( apply(all_vertices,2,sum) - worst_vertex ) / ( nrow(all_vertices) - 1 ) 
    output[1] <- round(output[1]) 
    output[2] <- round(output[2])
    return(output)
}	


bdry_cond <- function(input_vec = vector(0), ind_int = character(0), ind_num = character(0), min_int = integer(0), min_num = numeric(0)) {
    if(length(ind_int) != 0) input_vec[ind_int][ input_vec[ind_int] < min_int ] <- min_int 
    if(length(ind_num) != 0) input_vec[ind_num][ input_vec[ind_num] < min_num ] <- min_num
    #to ensure that integer outputs are in their appropriate integer format
    input_vec[ind_int] <- round(input_vec[ind_int]) 
    return(input_vec)
}

obj_fun <- xgb_wrap_obj
obj_fun_args <- list(param_fixed = param_fixed, vtx = NULL, dat_x = dat_sparse[-K_Folds[[k]],], dat_y = dat_y[-K_Folds[[k]]])


vtcs <- cbind( obj = rep(0,4), nround = c(150, 310, 100, 400), max_depth = c(8,6,6,4), eta = c(0.3, 0.5, 0.09, 0.05) )

N_vtcs <- nrow(vtcs)

var_names <- colnames(vtcs)[!( colnames(vtcs) %in% c("obj") )]

int_names <- c("nround","max_depth")
num_names <- c("eta")
min_intgr <- 1
min_numrc <- 0.005

bdry_param <- list( ind_int = c("nround","max_depth"), ind_num = c("eta"), min_int = 1, min_num = 0.005)
#stopping criterion initialization


#nelder_mead initialization
#set.seed(7159)
for ( i in 1:N_vtcs)	{
	obj_fun_args[["vtx"]] <- vtcs[i, var_names]
	vtcs[i,"obj"] <- do.call( getFunction("obj_fun"), obj_fun_args )
}
vtcs <- vtcs[order(vtcs[,"obj"], decreasing = min_max),] # sort from best to worst





obj_tm1 <- obj_t <- 0
N_shrink <- 0
iter = 1
iter_eq0 = 1



#start
while ( (iter <= 150) + (iter_eq0 <= 10) >= 2 ) 	{
	path <- "reflection"
	X_w <- vtcs[N_vtcs,var_names] 
	C <- centroid(vtcs[,var_names], X_w ) #centroid of best hyperplane which is opposite the worst vertex 

	#---- Reflection step ----------------------------------------------------------------------------
	if (path == "reflection") {
		X_r <- a_r*(C - X_w) 
		if ( boundary == TRUE )	{
			X_r <- do.call( get("bdry_cond"), c( list(input_vec = X_r), bdry_param) )
		}
		obj_fun_args[["vtx"]] <- X_r
		obj_r <- do.call( getFunction("obj_fun"), obj_fun_args )
		
		#Relect if ( (vtcs[,"obj"] >= obj_r) & (coj_r > vtcs[,"obj"]) ) { 
		if ( (vtcs[1,"obj"] <= obj_r) & (obj_r < vtcs[2,"obj"]) ) { 
			vtcs[N_vtcs,] <- c(cbj = obj_r, X_r); 
			# path <- "stopping" 
		}	else if ( obj_r < vtcs[1,"obj"] ) { #go to expansion
			path <- "expansion" 
		}	else {
			path <- "contraction"
		}
	}


	#---- Expansion Step ----------------------------------------------------------------------------
	if (path == "expansion") {
		X_e <- C + a_e*(X_r - C) 
		if ( boundary == TRUE )	{
			X_e <- do.call( get("bdry_cond"), c( list(input_vec = X_r), bdry_param) )
		}
		obj_fun_args[["vtx"]] <- X_e
		obj_e <- do.call( getFunction("obj_fun"), obj_fun_args )
		if (obj_e < obj_r)	{
			vtcs[N_vtcs,] <- c(obj = obj_e, X_e) 
		} else { 
			vtcs[N_vtcs,] <- c(obj = obj_r, X_r)
		}
	} # ----------------------------------------------------------------------------- 


	#---- Contraction Step ---------------------------------------------------------------------------- 
	if (path == "contraction") { 
		if ( (vtcs[2,"obj"] <= obj_r) & (obj_r < vtcs[N_vtcs,"obj"]) ) { # g(X_bad) <= g(X_r) < g(X_worst) 
			#outer contraction 
			X_o <- C + a_c*(X_r - C) 
			if ( boundary == TRUE )	{
				X_o <- do.call( get("bdry_cond"), c( list(input_vec = X_r), bdry_param) )
			}
			obj_fun_args[["vtx"]] <- X_o
			obj_o <- do.call( getFunction("obj_fun"), obj_fun_args )
			if (obj_o <= obj_r) { #g(X_o) <= g(X_r) 
				vtcs[N_vtcs, ] <- c( obj = obj_o, X_o) 
		#path <- "stopping" 
			} else { 
				path <- "shrinking" 
			}
		} else if (vtcs[N_vtcs,"obj"] <= obj_r) {# g(X_worst) <= g(X_r) 
			#inner contraction 
			X_i <- C + a_c*(X_w - C) 
			if ( boundary == TRUE )	{
				X_i <- do.call( get("bdry_cond"), c( list(input_vec = X_r), bdry_param) )
			}
			obj_fun_args[["vtx"]] <- X_i
			obj_i <- do.call( getFunction("obj_fun"), obj_fun_args )
			if (obj_i <= vtcs[N_vtcs,"obj"]) { #g(x_o) <= g(x_r) 
				vtcs[N_vtcs, ] <- c( obj = obj_i, X_i) 
		#path <- "stopping" 
			} else { 
				path <- "shrinking" 
			} 
		}
	}

	# Shrinking Step 
	if (path == "shrinking") {
		idx_s <- which(!vtcs[,"obj"] %in% vtcs[1,"obj"])
		X_best <- t(matrix( rep(vtcs[1,var_names], length(idx_s)), nrow = N_vtcs - 1, ncol = length(idx_s) ))
		Xsj_m_Xbest <- vtcs[idx_s,var_names] - X_best
		if ( boundary == TRUE )	{
			Xsj_m_Xbest <- t( apply(Xsj_m_Xbest, 
									1, 
									function(x) do.call( get("bdry_cond"), c(list( input_vec = x), bdry_param ) ) 
									) )
		}
		vtcs[idx_s, var_names] <- t( apply(X_best + a_s*Xsj_m_Xbest, 
										   1, 
										   function(x) do.call( get("bdry_cond"), c(list( input_vec = x), bdry_param ) )
										   ) )
		for (i in idx_s) {
			vtcs[i,"obj"] <- do.call( getFunction("obj_fun"), obj_fun_args )
		}
	}
	# ----------------------------------------------------------------------------- 


	#---- Stopping Step 
	vtcs <- vtcs[ order(vtcs[,"obj"], decreasing = min_max),]
	obj_t <- vtcs[1,"obj"]
	flush.console()
	print( obj_t - obj_tm1 )
	print(path)
	print(vtcs)
	print(iter)
	if ( (obj_t - obj_tm1) != 0 )	{
		iter_eq0 = 0
	}	else {
		iter_eq0 = iter_eq0 + 1
	}
	obj_tm1 <- obj_t
	iter = iter + 1
}


#__________________________________________________________________________ 
# ------------------------------------------------------------------------




# bdry_cond_mat <- function( input_mat = matrix(0), ind_int = character(0), ind_num = character(0), min_int = integer(0), min_num = numeric(0))  {
#	to ensure that inputs are non-zeros for both integer and numeric entries 
	# ind <- input_mat[, ind_int] < min_int 
	# input_mat[, ind_int][ind] <- min_int 
	
	# which( input_mat[, ind_num] < min_num )
	
	# for ( i in 1:length(ind_num)) { 
		# idx <- which( input_mat[, ind_num[i]] < min_num )
		# num_min <- min(input_mat[-idx, ind_num])
		# if ( (length(idx) > 0) & (num_min >0) )	{
			# input_mat[idx, ind_num] <- min( num_min, min_deft )
		# } else if ( (length(idx) > 0) & (num_min <= 0) )	{
			# input_mat[idx, ind_num] <-  min_deft
		# }
	# }
	#to ensure that integer outputs are in their appropriate integer form 
	# output[,ind_int] <- apply(output[,ind_int],2, round) 
	# return(output) 
# }











# plot train error vs test error
library(tidyr)		
library(readr)
library(dplyr)	  
bst.cv$dt %>%
  select(-contains("std")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()
  
bst = list()
for (k in 1:K_Folds_K)	{
	bst[[k]] = xgboost(param=param, data=dat_sparse[-K_Folds[[k]],] , label = dat_y[-K_Folds[[k]]],
					 nround=nround.opt, prediction=TRUE, verbose=TRUE) #which(bst.cv$dt[,test.auc.mean] %in% max(bst.cv$dt[,test.auc.mean]))				  
}

pred = integer(length(dat_y))
for (k in 1:K_Folds_K)	{
	pred[K_Folds[[k]]] <- apply( matrix(predict(bst[[k]], dat_sparse[K_Folds[[k]],]), ncol=3, byrow = TRUE),1 , which.max) - 1
}	

sum( pred != dat_y ) / length(dat_y)

importance_matrix = xgb.importance(colnames(dat_sparse), model=bst[[1]])
# plot feature importance
gp = xgb.plot.importance(importance_matrix)
print(gp)	