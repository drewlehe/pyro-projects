import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch

# Soft-observed data
y_obs = torch.tensor(3.0)
sigma = torch.tensor(0.5)   # how soft the conditioning is

def f(z):
    # some deterministic function
    return z**2

def model():
    # Prior for z
    z = pyro.sample("z", dist.Normal(0., 3.))

    # --- Soft conditioning ---
    # Add soft likelihood penalty (Gaussian)
    residual = f(z) - y_obs
    pyro.factor("soft_conditioning", -0.5 * (residual / sigma)**2)

    return z

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=500)
mcmc.run()
samples = mcmc.get_samples()

print(samples["z"].mean(), samples["z"].std())
