data {
  int<lower=1> N;          // number of data points
  array[N] real y;         // observations

  vector[N] AGE;           // predictor for component 1
  vector[N] DURATION;      // predictor for component 1
  vector[N] DISTANCE;      // predictor for theta
}

parameters {
  vector<lower=0>[2] sigma;   // scales of mixture components

  real alpha1;                // intercept for AGE
  real beta1;                 // slope for AGE

  real alpha2;                // intercept for DURATION
  real beta2;                 // slope for DURATION

  real alpha_t;               // intercept for theta
  real beta_l1;               // slope for first language
  real beta_l2;               // slope for second language
  real beta_d;                // slope for distance
}

model {
  //priors
  sigma ~ lognormal(0, 50);

  alpha1 ~ normal(0, 10);
  beta1 ~ normal(40, 10);
  alpha2 ~ normal(0, 10);
  beta2 ~ normal(40, 10);
  alpha_t ~ normal(0, 10);
  beta_l1 ~ normal(0, 10);
  beta_l2 ~ normal(0, 10);
  beta_d ~ normal(0, 10);

  //likelihood
  for (n in 1:N) {
    vector[2] lps; // log probabilities of each component
    real mu1 = alpha1 + beta1 * log(AGE[n]+machine_precision()); // learning curve for mu[1]
    real mu2 = alpha2 + beta2 * log(DURATION[n]+machine_precision()); // learning curve for mu[2]

    lps[1] = normal_lpdf(y[n] | mu1, sigma[1]); // first component
    lps[2] = normal_lpdf(y[n] | mu2, sigma[2]); // second component
    real theta = inv_logit(alpha_t + beta_l1 * lps[1] + beta_l2 * lps[2] + beta_d * DISTANCE[n]); // logistic regression for theta
    vector[2] log_theta; // log of mixing proportion and its complement
    log_theta[1] = log(theta);
    log_theta[2] = log(1 - theta);

    lps += log_theta;

    target += log_sum_exp(lps); // increment log probability
  }
}

