import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoGuideList, AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def model_wait_times(t):
    # Simple Bayesian experimment to model wait times.
    alpha0 = torch.tensor(1.0)   # shape
    beta0  = torch.tensor(1.0)   # rate

    # Conjugate Prior is a Gamma distribution
    lam = pyro.sample("lambda", dist.Gamma(alpha0, beta0))

    # Likelihood is an Exponential distribution, whose 'lambda' parameter is defined by the conjugate prior
    # The likelihood in a Bayesian experiment models your 'process' that generates the data. The conjugate prior is a guess at that process's parameter(s).
    # So we essentially have two guesses: one for the process, and one for the process's parameters. This 'double-layered guess' is common to all Bayesian experiments.
    with pyro.plate("data", len(t)):
        # The pyro plate essentially says: 
        # “Everything I sample inside here happens independently for each item in this batch, so please handle it in a 
        # vectorized and probability-correct way.”
        pyro.sample("t_obs", dist.Exponential(lam), obs=t)
        # Whereas a loop would create separate 'sites', the plate keeps all samples in one site called "data"

# Inference using SVI with autoguide
def fit_svi(t, steps=2000, lr=0.02):
    guide = AutoGuideList(model_wait_times)
    # Conjugate here is analytic, but we can still use a simple guide:
    guide.append(AutoDelta(model_wait_times))  # point-estimate (MAP); swap for AutoNormal for full approx

    svi = SVI(model_wait_times, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    t_tensor = torch.as_tensor(t, dtype=torch.float32)

    for _ in range(steps):
        svi.step(t_tensor)

    post = guide.median({"t": t_tensor})  # contains "lambda"
    return float(post["lambda"])

# Posterior predictive sampling for next wait time
def sample_posterior_predictive(t, num_samples=1000):
    # Conjugate posterior parameters
    t = torch.as_tensor(t, dtype=torch.float32)
    alpha_post = 1.0 + t.numel()
    beta_post  = 1.0 + t.sum()
    lam_samps = dist.Gamma(alpha_post, beta_post).sample((num_samples,))
    # Predictive for next waiting time by mixing Exponential over posterior λ
    t_new = dist.Exponential(lam_samps).sample()
    return t_new
