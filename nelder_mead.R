install.packages('Matrix', repos='http://cran.us.r-project.org', lib= "~/")
install.packages('dplyr', repos='http://cran.us.r-project.org', lib= "~/")
install.packages("xgboost", repos="https://cran.rstudio.com", lib= "~/")

#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo


library("Matrix", lib.loc = "~/")

dat.set1 <- train[,-1]
#dat_sparse = model.matrix(TARGET~.-1, data = dat.set1, sparse = TRUE) #
#dat_sparse = sparse.model.matrix(Species~.-1, data = dat)
dat_sparse <- as.matrix(train[, !names(train) %in% c("ID", "TARGET")])
dat_y = as.numeric(dat.set1[,"TARGET"])


#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo


library("xgboost", lib.loc = "~/")
library(dplyr, lib.loc = "~/")

set.seed(7157)
k <- 1
a_r <- 1
a_e <- 2
a_c <- 0.5
a_s <- 0.5

min_max = 0 # if max: min_max = 1, if min: min_max = 0
boundary = TRUE

if (opt_max == 0)	{
	opt_pos = 1
} else {
	opt_pos = 0
}

#Nelder Mead functions # =================================== ========================================= 
centroid <- function(all_vertices, worst_vertex) {
    output <- ( apply(all_vertices,2,sum) - worst_vertex ) / ( nrow(all_vertices) - 1 ) 
    output[1] <- round(output[1]) 
    output[2] <- round(output[2])
    return(output)
}	

#*******************
bdry_cond <- function(input_vec = vector(0), ind_int = character(0), ind_num = character(0), min_int = integer(0)) {
    input_vec[ind_int[1]] <- exp(input_vec[ind_int[1]])
    input_vec[ind_int[2]] <- exp(input_vec[ind_int[2]])
    #to ensure that integer outputs are in their appropriate integer format
    input_vec[ind_num] <- exp(input_vec[ind_num])/(1+exp(input_vec[ind_num]))
    input_vec[ind_int][ input_vec[ind_int] < min_int ] <- min_int
    input_vec[ind_int] <- round(input_vec[ind_int])
    return(input_vec)
}
#xbg function wrap that only returns the objective function output 
xgb_wrap_obj <- function(param_fixed, vtx, dat_x, dat_y, nfolds = 5, pred = FALSE, verb = FALSE ){ 
    return(	-xgb.cv( param=param_fixed, nround = vtx["nround"],
                     max_depth = vtx["max_depth"], eta = vtx["eta"],
                     data= dat_x, label = dat_y, nfold = nfolds, 
                     stratified = TRUE, prediction=pred, verbose=verb, missing=NA)$test.auc.mean[vtx["nround"]] )
}

# =================================== ============== 


#***********
# xgboost parameters
param_fixed <- list("nthread" = 3,   # number of threads to be used 
					"objective" = "binary:logistic",    # binary classification 
					"eval_metric" ="auc",    # evaluation metric
					"gamma" = 0,    # minimum loss reduction 
					"subsample" = 0.9,    # part of data instances to grow tree 
					"colsample_bytree" = 0.9,  # subsample ratio of columns when constructing each tree 
					"min_child_weight" = 1)  # minimum sum of instance weight needed in a child

obj_fun <- xgb_wrap_obj
obj_fun_args <- list(param_fixed = param_fixed, vtx = NULL, dat_x = dat_sparse, dat_y = dat_y)

#******************************
vtcs <- cbind( obj = rep(0,4), nround = log(c(300,100,400,200)), max_depth = log(c(14,14,6,10)), eta = -log(1/c(0.2, 0.2, 0.09, 0.005) - 1 +1e-05) )
#vtcs <- cbind( obj = rep(0,4), subsample = -log(1/c(0.1,0.5,0.9,0.2) - 1 + 1e-5), colsample_bytree = -log(1/c(0.2,0.9,0.4,0.8) - 1 + 1e-5), min_child_weight = log(c(1, 4,9, 8)) )


N_vtcs <- nrow(vtcs)

var_names <- colnames(vtcs)[!( colnames(vtcs) %in% c("obj") )]

int_names <- c("nround","max_depth")
num_names <- c("eta")
min_intgr <- 1
min_numrc <- 0.005

bdry_param <- list( ind_int = c("nround","max_depth"), ind_num = c("eta"), min_int = 1)
#stopping criterion initialization


#nelder_mead initialization
#set.seed(7159)
for ( i in 1:N_vtcs)	{
	obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = vtcs[i, var_names]), bdry_param) )
	vtcs[i,"obj"] <- do.call( getFunction("obj_fun"), obj_fun_args )
}
vtcs <- vtcs[order(vtcs[,"obj"], decreasing = min_max),] # sort from best to worst



#NM iteration initialization
obj_tm1 <- obj_t <- 0
N_shrink <- 0
iter = 1
iter_eq0 = 1

#NM begins
while ( ((iter <= 200) + (iter_eq0 <= 10)) >= 2 )	{
	path <- "reflection"
	X_w <- vtcs[N_vtcs,var_names] 
	C <- centroid(vtcs[,var_names], X_w ) #centroid of best hyperplane which is opposite the worst vertex 

	#---- Reflection step ----------------------------------------------------------------------------
	if (path == "reflection") {
		X_r <- a_r*(C - X_w) 
		obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = X_r), bdry_param) )
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
		obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = X_e), bdry_param) )
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
			obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = X_o), bdry_param) )
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
			obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = X_i), bdry_param) )
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
		vtcs[idx_s, var_names] <- X_best + a_s*(vtcs[idx_s,var_names] - X_best)
		for (i in idx_s) {
			obj_fun_args[["vtx"]] <- do.call( get("bdry_cond"), c( list(input_vec = vtcs[i, var_names]), bdry_param) )
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


t(apply(vtcs[, var_names], 1, function(x) do.call( get("bdry_cond"), c( list(input_vec = x), bdry_param) ) ))

