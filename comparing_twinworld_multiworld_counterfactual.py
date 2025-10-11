"""
Demonstrating
- Pyro causal model with latent confounder U
- Simulate data
- SVI posterior inference with AutoNormal
- ChiRho MultiWorldCounterfactual and TwinWorldCounterfactual using posterior
- Sampling, ATE/ITE computation, and plotting
"""

import os
import sys
import math
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import matplotlib.pyplot as plt
from chirho.counterfactual import MultiWorldCounterfactual, TwinWorldCounterfactual

# reproducibility
pyro.set_rng_seed(0)
torch.manual_seed(0)

# ---- Model hyperparameters
N = 500                       # number of individuals
alpha = 1.0                   # logistic link for P(T=1 | U)
beta = 2.0                    # treatment effect
gamma = 1.0                   # effect of U on Y
sigma = 0.5                   # observation noise

# ---- Simulate observed data
u_true = torch.randn(N)                             # latent confounder (unobserved)
p_t = torch.sigmoid(alpha * u_true)                 # propensity per individual
t_true = torch.bernoulli(p_t)                       # observed treatment (0/1)
y_true = beta * t_true + gamma * u_true + sigma * torch.randn(N)  # observed outcome

# Put observed data in dictionary (torch tensors)
data = {"treatment": t_true, "outcome": y_true}

# ---- Pyro model (vectorized over data) 
def model(data=None):
    # data: dict with "treatment" and "outcome" tensors of shape [N]
    if data is None:
        raise ValueError("This model expects 'data' to be provided for inference.")
    treatments = data["treatment"]
    outcomes = data["outcome"]
    N_local = treatments.shape[0]

    with pyro.plate("data", N_local):
        # latent confounder per individual
        u = pyro.sample("U", dist.Normal(0., 1.))
        # treatment depends on U
        t = pyro.sample("treatment", dist.Bernoulli(logits=alpha * u), obs=treatments)
        # outcome depends on treatment and U
        y_mean = beta * t + gamma * u
        pyro.sample("outcome", dist.Normal(y_mean, sigma), obs=outcomes)
    return {}

# ---- Variational inference (AutoNormal guide + SVI)
pyro.clear_param_store()
guide = AutoNormal(model)
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_steps = 3000
print("Running SVI...")
for step in range(num_steps):
    loss = svi.step(data)
    if step % 500 == 0:
        print(f"SVI step {step:4d} loss = {loss:.2f}")
print("SVI complete.")

# Test: draw one posterior sample from the guide (vectorized)
# AutoNormal returns a dict mapping plate-sampled sites to tensors.
posterior_sample = guide(data)
# 'U' should be a tensor of shape [N]
U_post_example = posterior_sample["U"]
print("posterior sample 'U' shape:", U_post_example.shape)

# ---- Build ChiRho counterfactual objects using the posterior guide
# NOTE: API used here follows the earlier discussion: passing posterior_guide and data.
# If your ChiRho version uses different kwarg names, adapt accordingly.
print("Constructing counterfactual objects...")
multi_cf = MultiWorldCounterfactual(
    model,
    interventions={"treatment": [0.0, 1.0]},
    posterior_guide=guide,
    data=data
)

twin_cf = TwinWorldCounterfactual(
    model,
    interventions={"treatment": [0.0, 1.0]},
    posterior_guide=guide,
    data=data
)

# ---- 5) Sample counterfactuals ----
n_samples = 1000
print(f"Sampling {n_samples} counterfactual draws from each object...")
multi_samples = multi_cf.sample(n=n_samples)
twin_samples = twin_cf.sample(n=n_samples)
print("Sampling complete.")

# Expected structure (per our conversation):
# multi_samples: dict with keys like "world_0", "world_1", each mapping to dicts of tensors
# twin_samples: same, but paired by posterior latent 'U' per individual.

# ---- Compute ATE (MultiWorld)
# Handle shapes carefully: worlds expected to have 'outcome' tensors
w0_out = multi_samples["world_0"]["outcome"]   # shape maybe [n_samples, N] or [n_samples, ...]
w1_out = multi_samples["world_1"]["outcome"]

# We want the population mean across samples and individuals.
# Collapse sample dimension then individual dimension.
def collapse_mean(tensor):
    # Accept torch tensor of arbitrary dims; return scalar mean as python float
    if isinstance(tensor, torch.Tensor):
        return float(tensor.mean().detach().cpu().numpy())
    else:
        raise ValueError("Expected a torch.Tensor")

ate_multi = collapse_mean(w1_out) - collapse_mean(w0_out)
print(f"Posterior ATE (MultiWorld) = {ate_multi:.4f}")

# ---- Compute ITE distribution & ATE from TwinWorld
tw_w0_out = twin_samples["world_0"]["outcome"]   # expected shape [n_samples, N]
tw_w1_out = twin_samples["world_1"]["outcome"]

# Compute ITE per sample-per-individual, then collapse sample axis by e.g. mean over samples
# We'll compute the posterior mean ITE per individual and also the distribution of sample-level ITEs.
# First compute sample-level ITEs:
ite_samples = tw_w1_out - tw_w0_out         # shape: [n_samples, N]

# Posterior mean ITE per individual (averaging over SVI/CF samples)
ite_mean_per_individual = ite_samples.mean(dim=0)   # shape: [N]

# Posterior ATE from TwinWorld (mean of the individual means)
ate_twin = float(ite_mean_per_individual.mean().detach().cpu().numpy())
print(f"Posterior ATE (TwinWorld, mean over individuals) = {ate_twin:.4f}")

# Plot histogram of posterior ITE (across individuals)
# We'll use the per-individual posterior mean ITEs as the main visualization.
ite_for_plot = ite_mean_per_individual.detach().cpu().numpy()

plt.figure(figsize=(8,5))
plt.hist(ite_for_plot, bins=40, edgecolor='k', alpha=0.7)
plt.axvline(ite_for_plot.mean(), color='red', linestyle='--', label=f"TwinWorld mean ATE = {ite_for_plot.mean():.3f}")
plt.axvline(ate_multi, color='black', linestyle=':', label=f"MultiWorld ATE = {ate_multi:.3f}")
plt.title("Posterior Individual Treatment Effects (ITE) — TwinWorld")
plt.xlabel("ITE (Y_treated - Y_control)")
plt.ylabel("Number of individuals (posterior mean ITE)")
plt.legend()
plt.tight_layout()

# Save plot to file and also show
out_plot = "posterior_ite_hist.png"
plt.savefig(out_plot, dpi=150)
print(f"Saved ITE histogram to {out_plot}")
plt.show()

# ---- Example: inspect a few individuals (first 5)
print("\nExample: first 5 individuals' posterior mean ITEs:")
for i in range(min(5, N)):
    print(f"  i={i:3d} ITE_mean = {ite_mean_per_individual[i].item():.4f}")
