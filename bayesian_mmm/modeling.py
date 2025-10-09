import numpy as np, pandas as pd, torch
import pyro, pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# --- Synthetic Data Generation ---
np.random.seed(42)
days = 365
dates = pd.date_range("2024-01-01", periods=days)
# Daily spends
paid_search = np.clip(np.random.normal(1000, 200, days), 0, None)
social_media = np.clip(np.random.normal(200, 50, days), 0, None)
email        = np.clip(np.random.normal(100, 20, days), 0, None)
promotions   = np.clip(np.random.normal(500, 150, days), 0, None)
paid_search *= np.linspace(1.0, 1.3, days)  # add trend

# Adstock transform
def apply_adstock(spend, retention):
    eff = np.zeros_like(spend)
    eff[0] = spend[0]
    for t in range(1, len(spend)):
        eff[t] = spend[t] + retention * eff[t-1]
    return eff

asp = apply_adstock(paid_search, 0.6)
asm = apply_adstock(social_media, 0.3)
ae  = apply_adstock(email,        0.1)
ap  = apply_adstock(promotions,   0.4)

# True effects and sales
beta_true = np.array([0.10, 0.30, 0.20, 0.05])
base = 50
X_np = np.column_stack([asp, asm, ae, ap])
sales = base + X_np.dot(beta_true) + np.random.normal(0, 10, size=days)

# --- Pyro Model ---
# Convert to torch tensors
X = torch.tensor(X_np, dtype=torch.float)
y = torch.tensor(sales, dtype=torch.float)

def model(X, y=None):
    alpha = pyro.sample("alpha", dist.Normal(0., 100.))
    beta  = pyro.sample("beta",  dist.Normal(torch.zeros(4), 1.0 * torch.ones(4)))
    sigma = pyro.sample("sigma", dist.HalfCauchy(5.0))
    mu = alpha + (X * beta).sum(axis=1)
    with pyro.plate("data", size=len(y)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

# Run NUTS sampling
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
mcmc.run(X, y)

# Extract and summarize
samples = mcmc.get_samples()
beta_samples = samples['beta'].detach().numpy()
print("Estimated ROI (posterior means):", beta_samples.mean(axis=0))
