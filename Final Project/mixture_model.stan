data {
  int<lower=1> N;          // number of data points
  array[N] real y;         // observations (overall score)

  vector[N] AGE;           // predictor for component 1, as AGE increases, L1 score increases
  vector[N] DURATION;      // predictor for component 2, as DURATION increases, L2 score increases
  vector[N] DISTANCE;      // predictor for theta which is the proportion of L1 transfer
}

parameters {
  real<lower = 0> sigma;              // variance of score distribution
  real<lower = 0> alpha;              // innate knowledge about langauge
  real<lower = 0> beta;               // learning ability 
  real<lower = 0, upper = 1> exp;     // exposure discount for L2

  real beta_d;                        // slope for theta
}

model {
  //priors
  sigma ~ lognormal(0, 10);
  alpha ~ normal(100, 10);
  beta ~ normal(100, 10);
  exp ~ normal(0.5,0.1);

  beta_d ~ normal(0, 1);

  //likelihood
  for (n in 1:N) {
    vector[2] lps;                                                                              // log probabilities of each component
    real mu1 = alpha + (1-DISTANCE[n]) * beta * log(AGE[n]+1+machine_precision());                                // learning curve for L1, as mu[1]
    real mu2 = alpha + exp * beta * log(DURATION[n]+1+machine_precision());                     // learning curve for L2, as mu[2]

    lps[1] = normal_lpdf(y[n] | mu1, sigma);                                    // first component, L1
    lps[2] = normal_lpdf(y[n] | mu2, sigma);                                                    // second component, L2
    
    real theta = inv_logit(log(mu1/mu2) + beta_d * DISTANCE[n]);                                // logistic regression for theta
    vector[2] log_theta;                                                                        // log of mixing proportion and its complement
    log_theta[1] = log(theta);
    log_theta[2] = log(1-theta);

    lps += log_theta;

    target += log_sum_exp(lps);                                                                 // increment log probability
  }
}

generated quantities {
  real<lower=0, upper=1> theta_pred[N];            // predicted values of theta for each data point
  real y_pred[N];                      // predicted values of y for each data point

  for (n in 1:N) {
    real mu1_pred = alpha + (1 - DISTANCE[n]) * beta * log(AGE[n] + 1 + machine_precision());   // predicted value of mu1 for each data point
    real mu2_pred = alpha + exp * beta * log(DURATION[n] + 1 + machine_precision());            // predicted value of mu2 for each data point

    theta_pred[n] = inv_logit(log(mu1_pred / mu2_pred) + beta_d * DISTANCE[n]);                  // predicted value of theta for each data point

    // generate predicted values of y using the predicted values of mu1, mu2, and theta
    y_pred[n] = theta_pred[n] * normal_rng(mu1_pred, sigma) + (1 - theta_pred[n]) * normal_rng(mu2_pred, sigma);
  }
}







