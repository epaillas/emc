import yaml
import numpy as np
import argparse
from abacusnbody.hod.abacus_hod import AbacusHOD
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def read_positions(hod_dict):
    data = hod_dict['LRG']
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x_rsd = (x + vx / (hubble * az)) % boxsize
    y_rsd = (y + vy / (hubble * az)) % boxsize
    z_rsd = (z + vz / (hubble * az)) % boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd

def run_hod(p, param_mapping, param_tracer, data_params, Ball, nthreads):
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1
    ngal_dict = Ball.compute_ngal(Nthread=nthreads)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread=nthreads)
    return mock_dict

def setup_hod(config):
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    newBall = AbacusHOD(sim_params, HOD_params)
    newBall.params['Lbox'] = boxsize
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return newBall, param_mapping, param_tracer, data_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    setup_logging(level='WARNING')
    boxsize = 2000
    redshift = 0.5

    # HOD configuration
    hod_prior = 'yuan23'
    config_dir = './'
    config_fn = Path(config_dir, f'abacushod_config.yaml')
    config = yaml.safe_load(open(config_fn))

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        hods_dir = Path(f'/pscratch/sd/e/epaillas/emc/hod_params/{hod_prior}/')
        hods_fn = hods_dir / f'hod_params_{hod_prior}_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)

            fig, ax = plt.subplots()
            for hod in range(start_hod, start_hod + n_hod):
                print(f'c{cosmo:03} ph{phase:03} hod{hod}')

                hod_dict = run_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, nthreads=256)

                x, y, z, x_rsd, y_rsd, z_rsd = read_positions(hod_dict)

                data_positions = {
                    'x': x, 'y': y, 'z': z,
                    'x_rsd': x_rsd, 'y_rsd': y_rsd, 'z_rsd': z_rsd,
                }
                output_dir = f'/pscratch/sd/e/epaillas/emc/hods/z0.5/{hod_prior}_prior2/c{cosmo:03}_ph{phase:03}'
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_fn = Path(output_dir) / f'hod{hod:03}.npy'
                np.save(output_fn, data_positions)

