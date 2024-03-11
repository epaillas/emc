from scipy.stats import qmc
import numpy as np
from pathlib import Path

prior = 'full'

if prior == 'full':
    header = "logM_cut, logM_1, sigma, alpha, kappa, alpha_c, alpha_s, s, A_cen, A_sat, B_cen, B_sat"
    priors = {
        'logM_cut': [12.0, 14.0],
        'logM_1': [13.0, 15.0],
        'sigma': [-3.5, 1.0],
        'alpha': [0.5, 1.5],
        'kappa': [0.0, 1.5],
        'alpha_c': [0.0, 1.0],
        'alpha_s': [0.0, 2.0],
        's': [-1.0, 1.0],
        'A_cen': [-1.0, 1.0],
        'A_sat': [-1.0, 1.0],
        'B_cen': [-1.0, 1.0],
        'B_sat': [-1.0, 1.0],
    }

sampler = qmc.LatinHypercube(d=len(priors), seed=42)
params = sampler.random(n=85000)
pmins = np.array([priors[key][0] for key in priors])
pmaxs = np.array([priors[key][1] for key in priors])
params = pmins + params * (pmaxs - pmins)

cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
split_params = np.array_split(params, len(cosmos))

for i, cosmo in enumerate(cosmos):
    output_dir = Path(f'./hod_params/{prior}')
    output_fn = output_dir / f'hod_params_{prior}_c{cosmo:03}.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(output_fn, split_params[i], header=header, delimiter=',', fmt="%.5f")
