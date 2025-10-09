import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd

# --------------------------------------------------
# 1. Generate data
# --------------------------------------------------
n = 500
X = torch.randn(n, 2)
true_beta = torch.tensor([1.0, -1.0])
logits = X @ true_beta
p_treat = torch.sigmoid(logits)

# Treatment assignment
T = pyro.sample("T", dist.Bernoulli(p_treat)).detach()

# Potential outcomes
Y0 = 2 + X[:, 0] + 0.5 * torch.randn(n)
Y1 = 4 + 2 * X[:, 0] + 0.5 * torch.randn(n)

# Observed outcome
Y = torch.where(T.bool(), Y1, Y0)

# --------------------------------------------------
# 2. Define Bayesian propensity model
# --------------------------------------------------
def propensity_model(X, T=None):
    beta = pyro.sample("beta", dist.Normal(torch.zeros(X.shape[1]), 2.0))
    alpha = pyro.sample("alpha", dist.Normal(0., 5.0))
    logits = alpha + X @ beta
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=T)
    return logits

# --------------------------------------------------
# 3. Fit model via NUTS
# --------------------------------------------------
nuts_kernel = NUTS(propensity_model)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
mcmc.run(X, T)

posterior_samples = mcmc.get_samples()
alpha_post = posterior_samples["alpha"]
beta_post = posterior_samples["beta"]

# --------------------------------------------------
# 4. Compute posterior predictive propensity scores
# --------------------------------------------------
# Average over posterior samples
with torch.no_grad():
    logits_post = alpha_post.mean() + X @ beta_post.mean(0)
    e_hat = torch.sigmoid(logits_post)

# --------------------------------------------------
# 5. Compute inverse propensity weights and ATE
# --------------------------------------------------
weights = torch.where(T == 1, 1 / e_hat, 1 / (1 - e_hat))
ipw_ate = torch.mean(T * Y / e_hat - (1 - T) * Y / (1 - e_hat))

print(f"Estimated ATE (IPW): {ipw_ate.item():.3f}")
