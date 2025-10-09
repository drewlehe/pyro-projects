import numpy as np
import pandas as pd

np.random.seed(42)
days = 365
dates = pd.date_range("2024-01-01", periods=days)

#Simulate daily spend for each channel (normal around a mean)
paid_search = np.random.normal(1000, 200, size=days)
social_media = np.random.normal(200, 50, size=days)
email       = np.random.normal(100, 20, size=days)
promotions  = np.random.normal(500, 150, size=days)

#No negative spends
paid_search = np.clip(paid_search, 0, None)
social_media = np.clip(social_media, 0, None)
email        = np.clip(email, 0, None)
promotions   = np.clip(promotions, 0, None)

#Optional trend: increase paid_search over time
trend = np.linspace(1.0, 1.3, days)
paid_search *= trend

#Adstock function: effect[t] = spend[t] + retention * effect[t-1]
def apply_adstock(spend, retention):
    effect = np.zeros_like(spend)
    effect[0] = spend[0]
    for t in range(1, len(spend)):
        effect[t] = spend[t] + retention * effect[t-1]
    return effect

#Apply adstock to each channel with specified retention rates
retentions = {'paid_search': 0.6, 'social_media': 0.3, 'email': 0.1, 'promotions': 0.4}
asp = apply_adstock(paid_search, retentions['paid_search'])
asm = apply_adstock(social_media, retentions['social_media'])
ae  = apply_adstock(email,       retentions['email'])
ap  = apply_adstock(promotions,  retentions['promotions'])

#True channel effect coefficients (sales per unit adstocked spend)
beta_paid   = 0.10
beta_social = 0.30
beta_email  = 0.20
beta_prom   = 0.05

#Simulate sales: baseline + sum(channel_effect * adstock) + noise
base_sales = 50
noise = np.random.normal(0, 10, size=days)
sales = (base_sales 
         + beta_paid   * asp 
         + beta_social * asm 
         + beta_email  * ae 
         + beta_prom   * ap 
         + noise)

#Combine into a DataFrame for clarity (not used directly in Pyro code)
data = pd.DataFrame({
    'PaidSearch': asp,
    'SocialMedia': asm,
    'Email':      ae,
    'Promotions': ap,
    'Sales':      sales
}, index=dates)
