data {
    int<lower=0> N;
    int<lower=1> Nattr; // number of covariates
    matrix[N,Nattr] X;  
    array[N] real y;

    real<lower=0> prior_width;
}

transformed data {
   /* ... declarations ... statements ... */
}

parameters {
    vector[Nattr] beta;  // attribute effects
    real <lower=0> sigma; // standard deviation of the noise
}

transformed parameters {
    // real <lower=0> sigmasquared = sigma*sigma; // variance of the noise
   /* ... declarations ... statements ... */
}

model {
    // Priors
    beta ~ normal(0, prior_width);
    sigma ~ normal(0, 10);

    // Likelihood
    y ~ normal(X*beta, sigma);
}