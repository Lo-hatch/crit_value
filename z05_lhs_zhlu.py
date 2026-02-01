import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import truncnorm
nsample = 50  # number of samples

param_names=["g1tuzet", "psi_50_leaf", "kmax", "P50", "P88dP50","root_shoot"]
# Continuous distributions (6 parameters)

######################################################
a, b =  1.4,np.inf # truncate at 0 (max)
mu, sigma = 3, 1.5
trunc_dist_g1 = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

a, b = -np.inf, 0  # truncate at 0 (max)
mu, sigma = -2, 1.5
trunc_dist_P50leaf = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

mu, sigma = -3, 1.5
trunc_dist_P50 = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
mu, sigma = -0.5, 1.5
trunc_dist_P88dP50 = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

a, b =  1,5 # truncate at 0 (max)
mu, sigma = 3, 1.5
trunc_dist_rootleaf = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

dist_cont = [trunc_dist_g1, trunc_dist_P50leaf, stats.lognorm, trunc_dist_P50, trunc_dist_P88dP50,trunc_dist_rootleaf]
param_cont = [(),(),(0.5,0,np.exp(0.1)),(),(),()]  # just for reference if you want

################################# deprecated#########################
# Discrete distribution (1 parameter, values 1-6), representing soil types

param_disc = [()]  # params for randint


param = param_cont + param_disc
dist = dist_cont
param = param_cont
if not isinstance(dist, (list, tuple)):
    nodim = True
    dist  = [dist]
    param = [param]
else:
    nodim = False
    assert len(dist) == len(param)
ndist = len(dist)
# Generate LHS
ran = np.random.uniform(0, 1, (ndist, nsample))
lhsout = np.empty((ndist, nsample))

for j, d in enumerate(dist):
    # if not isinstance(d, (stats.rv_discrete, stats.rv_continuous)):
    #     raise TypeError('dist is not a scipy.stats distribution object.')
    
    pars = tuple([float(k) for k in param[j]])
    idx = np.array(np.random.permutation(nsample), dtype=float)
    # Parameters are already included in d for uniform/randint
    #idx = np.random.permutation(nsample)
    p = (idx + ran[j, :]) / nsample
    if len(param[j]) > 0:  # continuous
        lhsout[j, :] = d(*pars).ppf(p) 
    else:                   # discrete, already frozen
        lhsout[j, :] = d.ppf(p)
    

# Round discrete parameter to integers (if necessary)
#lhsout[-1, :] = np.round(lhsout[-1, :])

print(lhsout)
lhsout = lhsout.T  # shape (nsample, ndist)

###################### make sure psi_50_leaf > P50
psi_idx = param_names.index("psi_50_leaf")
p50_idx = param_names.index("P50")

for i in range(nsample):
    while lhsout[i, psi_idx] <= lhsout[i, p50_idx]:
        lhsout[i, psi_idx] = trunc_dist_P50leaf.rvs()
# -------------------------------------------------------


ids = np.arange(1, nsample+1).reshape(-1,1)
lhs_table = np.hstack([ids, lhsout])

# Column names: first column is ID
columns = ['ID'] + param_names
df = pd.DataFrame(lhs_table, columns=columns)
print(df)
#df.to_excel('lhs_samples.xlsx', index=False)
# df.to_csv("lhs_samples_50_param6.csv", index=False)
# print("LHS table exported to lhs_samples.xlsx")

import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice style
sns.set(style="whitegrid")
# Plot histograms (and KDE) for each parameter
fig, axes = plt.subplots(len(param_names), 1, figsize=(7, 12))
fig.suptitle("LHS Sample Distributions of Parameters", fontsize=14)

for i, name in enumerate(param_names):
    # if name=='kmax':
    #     sns.histplot(np.exp(df[name]), kde=True, ax=axes[i])
    # else:
    sns.histplot(df[name], kde=True, ax=axes[i])
    axes[i].set_xlabel(name)
    axes[i].set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()