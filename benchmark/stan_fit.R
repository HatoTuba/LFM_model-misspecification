# This R script fits Stan DDM to well-specified (data generated by the basic DDM) and 
# misspecified (data generated by the basic LFM)

## IMPORTANT: run the models and save the posteriors separately because variables 
# may overwrite 

library(dplyr)
library(rstan)
library(ggplot2)

########-----------WELL-SPECIFIED CASE-----------########
#read data
dat <- read.csv('all_ddm_data.csv')
PARAMS <- c("v1", "v2", "a", "bias", "ndt")
mc.cores = parallel::detectCores()
# prepare dfs and lists to save the output
posterior_all = data.frame() # gelecek posterior buyuklugunde data frame olustur
stanfit_list <- list()
stan_data_list <- list()
# unique_subids <- unique(df_n$subid)
unique_subids <- seq(100)
# unique_subids <- sample(1:1000, 10, replace=TRUE)

for (subid in unique_subids){
  sub_dat <- dat[dat$subid == subid, ]
  # NDT_BOUND <- min(sub_dat$rt) * 0.95
  min_rt <- min(sub_dat$rt)
  
  # create stan data for different subjects in each iteration
  stan_data = list(
    `T`  = nrow(sub_dat),
    cond = sub_dat$cond,
    resp = sub_dat$resp,
    rt   = sub_dat$rt,
    min_rt = min_rt
  )
  
  # save data
  stan_data_list[[as.character(subid)]] <- stan_data
  
  ##------------------------------------------------
  ## FITTING PREPARATION
  ##------------------------------------------------
  
  init = function(chains=4) {
    L = list()
    for (c in 1:chains) {
      L[[c]]=list()
      L[[c]]$v1   = rnorm(1, 2, 2)
      L[[c]]$v2   = rnorm(1, -2, 2)
      L[[c]]$a    = runif(1, 0.5, 4.5)
      L[[c]]$ndt  = runif(1, 0.01, min_rt*0.95)
      L[[c]]$bias = runif(1, 0.4, 0.6)
    }
    return (L)
  }
  
  ##------------------------------------------------
  ## FIT MODEL
  ##------------------------------------------------
  
  fit_m1 <- stan(
    "standard_ddm.stan",
    data=stan_data,
    chains=4,   
    iter = 4000,
    cores=parallel::detectCores(),
    control = list(adapt_delta=0.95)
  )
  
  stanfit_list[[subid]] <- fit_m1 # to save each stanfit object 
  # save the posteriors to a df 
  post <- data.frame(rstan::extract(fit_m1, pars=PARAMS))
  post$subid <- rep(subid, nrow(post))
  posterior_all <- rbind(posterior_all, post)
  
}

# save posteriors to a csv file for further analysis in python
write.csv(posterior_all, file = 'posterior_all.csv', row.names = F, col.names = F)


########-----------MISSPECIFIED CASE-----------########
#read data
dat <- read.csv('all_levy_data.csv')
PARAMS <- c("v1", "v2", "a", "bias", "ndt")

# prepare dfs and lists to save the output
posterior_all_lf = data.frame() # prepare a dataframe for posterior samples
stanfit_list_lf <- list()
stan_data_list_lf <- list()
# select a subset of 100 to fit the basic DDM
unique_subids <- seq(100)


# prepare stan_data for fitting
for (subid in unique_subids){
  sub_dat <- dat[dat$subid == subid, ]
  # NDT_BOUND <- min(sub_dat$rt) * 0.95
  min_rt <- min(sub_dat$rt)
  
  # create stan data for different subjects in each iteration
  stan_data = list(
    `T`  = nrow(sub_dat),
    cond = sub_dat$cond,
    resp = sub_dat$resp,
    rt   = sub_dat$rt,
    min_rt = min_rt
  )
  
  # save data
  stan_data_list_lf[[as.character(subid)]] <- stan_data
  
  ##------------------------------------------------
  ## FITTING PREPARATION
  ##------------------------------------------------
  
  init = function(chains=4) {
    L = list()
    for (c in 1:chains) {
      L[[c]]=list()
      L[[c]]$v1   = rnorm(1, 2, 2)
      L[[c]]$v2   = rnorm(1, -2, 2)
      L[[c]]$a    = runif(1, 0.5, 4.5)
      L[[c]]$bias = runif(1, 0.4, 0.6)
      L[[c]]$ndt  = runif(1, 0.01, min_rt*0.95)
    }
    return (L)
  }
  
  ##------------------------------------------------
  ## FIT MODEL
  ##------------------------------------------------
  
  fit_m1 <- stan(
    "standard_ddm.stan",
    data=stan_data,
    chains=4,   
    iter = 4000,
    cores=parallel::detectCores(),
    control = list(adapt_delta=0.95)
  )
  
  stanfit_list_lf[[subid]] <- fit_m1 # to save each stanfit object 
  # save the posteriors to a df 
  post <- data.frame(rstan::extract(fit_m1, pars=PARAMS))
  post$subid <- rep(subid, nrow(post))
  posterior_all_lf <- rbind(posterior_all_lf, post)
}

# save posteriors to csv file for further analysis in python
write.csv(posterior_all_lf, file = 'lf_posterior.csv', row.names = F, col.names = F)
