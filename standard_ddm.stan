data {
  int<lower=1>           T;         // number of trials
  int<lower=1, upper=2>  resp[T];   // choice data
  int<lower=1, upper=2>  cond[T];   // condition
  vector<lower=0>[T]     rt;        // response time data
  real<lower=0>          min_rt;    // minimum RT
}

parameters {
  real                        v1;      // drift rate 1
  real                        v2;      // drift rate 2
  real<lower=0>               a;       // threshold
  real<lower=0, upper=1>      bias;    // starting point
  real<lower=0, upper=min_rt> ndt;     // non-decision time

}

model {
    // Priors
    v1   ~ normal(2, 2);
    v2   ~ normal(-2, 2);
    a    ~ normal(3, 1);
    bias ~ beta(5, 5);
    ndt  ~ gamma(3, 12);

    // Likelihood
    for (i in 1:T) {
        if (cond[i] == 1) {
            if (resp[i] == 1) {
                rt[i] ~ wiener(a, ndt, bias, v1);
            }
            else {
                rt[i] ~ wiener(a, ndt, 1-bias, -v1);
            }
        }
        else {
            if (resp[i] == 1) {
                rt[i] ~ wiener(a, ndt, bias, v2);
            }
            else {
                rt[i] ~ wiener(a, ndt, 1-bias, -v2);
            }
        }
    }
}
