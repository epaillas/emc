import numpy as np
from pathlib import Path
import pandas as pd
from pymocker.catalogues import read_utils
import argparse


def write_sunbird_params():
    cosmo_cols = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
    for cosmo in cosmos:
        cosmo_dict = read_utils.get_abacus_params(cosmo)
        cosmo_params = [cosmo_dict[column] for column in cosmo_cols]

        hod_dir = Path('./hod_params/full')
        hod_fn = hod_dir / f'hod_params_full_c{cosmo:03}.csv'
        hod_df = pd.read_csv(hod_fn, delimiter=',')
        hod_cols = hod_df.columns.str.strip()
        hod_cols = list(hod_cols.str.strip('# ').values)
        hod_params = hod_df.values

        columns_csv = cosmo_cols + hod_cols
        df = pd.DataFrame(columns=columns_csv)

        for i, hod in enumerate(hods):
            hod_params_i = list(hod_params[i])
            params = cosmo_params + hod_params_i
            df.loc[i] = params

        output_dir = './sunbird_params/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = Path(output_dir, f'AbacusSummit_c{cosmo:03}.csv')
        df.to_csv(output_fn, sep=',', index=False)

def write_cosmopower_params():
    param_dir = './sunbird_params/'
    combined_df = pd.DataFrame()
    for cosmo in cosmos:
        param_fn = Path(param_dir, f'AbacusSummit_c{cosmo:03}.csv')
        param_df = pd.read_csv(param_fn)
        combined_df = pd.concat([combined_df, param_df])
    output_dir = './cosmopower_params'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir) / 'AbacusSummit.npy'
    np.save(output_fn, combined_df.to_dict('list'))


cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
hods = list(range(0, 100))

write_sunbird_params()
write_cosmopower_params()
