import time
import yaml
import numpy as np
import argparse
from abacusnbody.hod.abacus_hod import AbacusHOD
from astropy.table import Table, vstack
from cosmoprimo.utils import DistanceToRedshift
from pathlib import Path
from pyrecon import utils
import os
from scipy.interpolate import CubicSpline
from cosmoprimo.fiducial import AbacusSummit, BOSS
from scipy.interpolate import InterpolatedUnivariateSpline
# import healpy
import fitsio
import logging
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_rsd_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict["LRG"]
    # vx = data["vx"]
    # vy = data["vy"]
    # vz = data["vz"]
    x = data["x"] + 990  # boxsize / 2
    y = data["y"] + 990  # boxsize / 2
    z = data["z"] + 990  # boxsize / 2
    # x_rsd = x + vx / (hubble * az)
    # y_rsd = y + vy / (hubble * az)
    # z_rsd = z + vz / (hubble * az)
    # x_rsd = x_rsd % boxsize
    # y_rsd = y_rsd % boxsize
    # z_rsd = z_rsd % boxsize
    return x, y, z  # , x_rsd, y_rsd, z_rsd


def downsample_mocks(zmin, zmax, data_real, data_mocks):
    # Downsample HOD mocks
    bins = np.arange(zmin, zmax, 0.001)
    nz = []
    nz_HOD = []
    random_selected_galaxies = []
    for b in range(len(bins) - 1):
        gal_in_bin_data = np.where(
            (data_real["Z"] >= bins[b]) & (data_real["Z"] < bins[b + 1])
        )[0]
        gal_in_bin_mocks = np.where(
            (data_mocks["Z"] >= bins[b]) & (data_mocks["Z"] < bins[b + 1])
        )[0]

        if len(gal_in_bin_mocks) > 0:
            nz = np.average(
                data_real["NZ"][gal_in_bin_data],
            )
            nz_HOD = np.average(data_mocks["NZ"][gal_in_bin_mocks])
            random_selected_galaxies = random_selected_galaxies + list(
                np.random.choice(
                    gal_in_bin_mocks,
                    int(np.round(nz / nz_HOD * len(gal_in_bin_mocks), 0)),
                    replace=False,
                )
            )
    random_selected_galaxies = np.array(random_selected_galaxies)

    data_mocks_downsampled = {}
    for key in data_mocks.keys():
        data_mocks_downsampled[key] = data_mocks[key][random_selected_galaxies]

    return data_mocks_downsampled


def create_randoms(zmin, zmax, data_mocks, mask_res, N_random_increase):
    mask_dir = "/pscratch/sd/e/epaillas/ds_desi/"
    mask_fn = Path(mask_dir) / f"AbacusSummit_HOD_mask_nside{mask_res}.npy"
    mask = np.load(mask_fn)
    ipix = np.where(mask == 1)[0]

    N_randoms = int(len(data_mocks["Z"]) * N_random_increase)

    phi_random = np.random.uniform(0, 2 * np.pi, N_randoms)
    theta_random = np.arccos(np.random.uniform(0, 2, N_randoms) - 1)
    pix_randoms = np.array(
        hp.ang2pix(nside=mask_res, phi=phi_random, theta=theta_random)
    )
    valid_randoms = np.where(np.isin(pix_randoms, ipix))[0]
    ra_randoms = np.degrees(phi_random[valid_randoms])
    dec_randoms = abs(np.degrees(theta_random[valid_randoms]) - 90)

    bins = np.arange(zmin, zmax, 0.00001)
    pdf, bin_edges = np.histogram(data_mocks["Z"], bins=bins, density=True)
    cdf = np.cumsum(pdf * np.diff(bin_edges))

    cdf_increase = []
    bin_center = []
    cdf_increase.append(cdf[0])
    bin_center.append(bins[0] + np.diff(bin_edges)[0] / 2)
    for i in np.arange(1, len(cdf)):
        if cdf[i] != cdf[i - 1]:
            cdf_increase.append(cdf[i])
            bin_center.append(bins[i] + np.diff(bin_edges)[0] / 2)
    cs_cdf = CubicSpline(cdf_increase, bin_center)

    z_randoms = cs_cdf(np.random.uniform(0, 1, len(ra_randoms)))

    randoms = Table()
    randoms["RA"] = ra_randoms
    randoms["DEC"] = dec_randoms
    randoms["Z"] = z_randoms

    return randoms


def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    # read the parameters
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == "sigma" and tracer_type == "LRG":
            Ball.tracers[tracer_type][key] = 10 ** p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
        # Ball.tracers[tracer_type][key] = p[mapping_idx]
    # a lot of this is a placeholder for something more suited for multi-tracer
    Ball.tracers["LRG"]["ic"] = 1
    ngal_dict = Ball.compute_ngal(Nthread=nthread)[0]
    N_lrg = ngal_dict["LRG"]
    Ball.tracers["LRG"]["ic"] = min(
        1, data_params["tracer_density_mean"]["LRG"] * Ball.params["Lbox"] ** 3 / N_lrg
    )
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread=nthread, reseed=args.seed)
    return mock_dict


def setup_hod(config):
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config["sim_params"]
    HOD_params = config["HOD_params"]
    data_params = config["data_params"]
    fit_params = config["fit_params"]
    # create a new abacushod object and load the subsamples
    Balls = []
    for ez in zranges[args.survey][args.tracer]:
        sim_params["z_mock"] = ez
        Balls += [AbacusHOD(sim_params, HOD_params)]
    # parameters to fit
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return Balls, param_mapping, param_tracer, data_params


def spl_nofz(zarray, fsky, cosmo, Nzbins=50):
    zmin, zmax = zarray.min(), zarray.max()
    zbins = np.linspace(zmin, zmax, Nzbins + 1)
    Nz, zbins = np.histogram(zarray, zbins)
    zmid = zbins[0:-1] + (zmax - zmin) / Nzbins / 2.0
    zmid[0], zmid[-1] = zbins[0], zbins[-1]
    rmin = cosmo.comoving_radial_distance(zbins[0:-1])
    rmax = cosmo.comoving_radial_distance(zbins[1:])
    vol = fsky * 4.0 / 3 * np.pi * (rmax**3.0 - rmin**3.0)
    nz_array = Nz / vol
    spl_nz = InterpolatedUnivariateSpline(zmid, nz_array)
    return spl_nz


# def get_fsky(ra, dec):
#     nside = 256
#     npix = healpy.nside2npix(nside)
#     phi = np.radians(ra)
#     theta = np.radians(90.0 - dec)
#     pixel_indices = healpy.ang2pix(nside, theta, phi)
#     pixel_unique = np.unique(pixel_indices)
#     fsky = len(pixel_unique) / npix
#     return fsky


def downsample(data, target_nz):
    idx_accept = []
    ratio = data['NZ'].max() / (target_nz(data['Z']).max()*1.2)
    norm = np.max(target_nz(data['Z']))
    if all(np.greater(data['NZ'], target_nz(data['Z'])*1.2)):
        data['Z'] = np.random.choice(data['Z'], size=round(len(data['Z'])/ratio), replace=False)
    rnd_num = np.random.uniform(size=len(data['Z']))
    idx_accept = np.where(rnd_num <= target_nz(data['Z'])/norm)[0]
    return idx_accept



if __name__ == "__main__":
    logger = logging.getLogger("ds_abacus_lightcone")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--survey", type=str, default="CMASS")
    parser.add_argument("--tracer", type=str, default="LRG")
    parser.add_argument("--match_nz", action="store_true")
    parser.add_argument("--hod_prior", type=str, default="baseline")
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    config_dir = "./"
    config_fn = Path(config_dir, "abacushod_config.yaml")
    config = yaml.safe_load(open(config_fn))

    boxsize = 2000  # size of boxes  that make up the lightcone
    fsky = 1 / 8  # default sky fraction for the lightcones

    zranges = {
        # "DESI": {"LRG": [0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]},
        "DESI": {"LRG": [0.400, 0.450, 0.500, 0.575, 0.650]},
        "LOWZ": {"LRG": [0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500]},
        "CMASS": {"LRG": [0.400, 0.450, 0.500, 0.575, 0.650, 0.725]},
    }

    if args.survey == "CMASS":
        data_dir = "/pscratch/sd/e/epaillas/summit_lightcones/CMASS/LRG/"
        data_fn = Path(data_dir) / "AbacusSummit_lightcone_c000_ph000_hod000.npy"
        data = np.load(data_fn, allow_pickle=True).item()

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        # az = 1 / (1 + redshift)
        # hubble = 100 * mock_cosmo.efunc(redshift)

        hods_dir = f"hod_params/{args.hod_prior}"
        hods_fn = Path(hods_dir, f"hod_params_{args.hod_prior}_c{cosmo:03}.csv")
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=",")

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f"AbacusSummit_base_c{cosmo:03}_ph{phase:03}"
            config["sim_params"]["sim_name"] = sim_fn
            try:
                Balls, param_mapping, param_tracer, data_params = setup_hod(config)
            except:
                logger.info(f"Skipping {sim_fn} as files are not present")
                continue

            for hod in range(start_hod, start_hod + n_hod):
                start_time = time.time()

                data_positions_sky = []
                for i, newBall in enumerate(Balls):
                    hod_dict = get_hod(
                        hod_params[hod],
                        param_mapping,
                        param_tracer,
                        data_params,
                        newBall,
                        3,
                    )
                    x, y, z = get_rsd_positions(hod_dict)

                    dist, ra, dec = utils.cartesian_to_sky(np.c_[x, y, z])
                    d2z = DistanceToRedshift(mock_cosmo.comoving_radial_distance)
                    redshift = d2z(dist)
                    mask = (redshift >= args.zmin) & (redshift <= args.zmax)

                    data_positions_sky.append(
                        np.c_[
                            ra[mask],
                            dec[mask],
                            redshift[mask],
                        ]
                    )
                data_positions_sky = np.concatenate(data_positions_sky, axis=0)

                logger.info(f"Calculating nz for c{cosmo:03} ph{phase:03} hod{hod:03}")
                spl_nz = spl_nofz(
                    data_positions_sky[:, 2],
                    fsky,
                    mock_cosmo,
                )
                nz = spl_nz(data_positions_sky[:, 2])

                output_dict = {
                    "RA": data_positions_sky[:, 0],
                    "DEC": data_positions_sky[:, 1],
                    "Z": data_positions_sky[:, 2],
                    "NZ": nz,
                }
                output_dir = f"/pscratch/sd/e/epaillas/summit_lightcones/HOD/{args.survey}/{args.tracer}/{args.hod_prior}/c{cosmo:03}_ph{phase:03}/complete/"
                if args.seed is not None: output_dir = output_dir + f"/seed{args.seed}/"
                Path.mkdir(Path(output_dir), parents=True, exist_ok=True)
                output_fn = (
                    Path(output_dir)
                    / f"AbacusSummit_lightcone_c{cosmo:03}_ph{phase:03}_hod{hod:03}.npy"
                )
                np.save(output_fn, output_dict)

                # if args.match_nz:
                #     data_mocks = np.load(output_fn, allow_pickle=True).item()
                #     if args.survey == "CMASS":
                #         data_dir = "/pscratch/sd/e/epaillas/ds_boss/CMASS"
                #         data_fn = Path(data_dir) / "galaxy_DR12v5_CMASS_North.fits"
                #         data_real = fitsio.read(data_fn)
                #         zmin, zmax = 0.4, 0.7
                #         data_mocks_downsampled = downsample_mocks(
                #             zmin=zmin,
                #             zmax=zmax,
                #             data_real=data_real,
                #             data_mocks=data_mocks,
                #         )
                #         spl_nz = spl_nofz(
                #             data_mocks_downsampled["Z"],
                #             fsky,
                #             mock_cosmo,
                #         )
                #         nz = spl_nz(data_mocks_downsampled["Z"])
                #         data_mocks_downsampled["NZ"] = nz
                #         output_dir = f"/pscratch/sd/e/epaillas/summit_lightcones/HOD/{args.survey}/{args.tracer}/{args.hod_prior}/c{cosmo:03}_ph{phase:03}"
                #         Path.mkdir(Path(output_dir), parents=True, exist_ok=True)
                #         output_fn = (
                #             Path(output_dir)
                #             / f"AbacusSummit_lightcone_c{cosmo:03}_ph{phase:03}_hod{hod:03}.npy"
                #         )
                #         np.save(output_fn, data_mocks_downsampled)
                #     elif args.survey == "LOWZ":
                #         data_mocks = np.load(output_fn, allow_pickle=True).item()
                #         data_fn = "/pscratch/sd/e/epaillas/dsl-lowz/nz_lowz.txt"
                #         data = np.genfromtxt(data_fn, skip_header=1)
                #         z = data[:, 0]
                #         nz = data[:, 1]
                #         nz_spline = InterpolatedUnivariateSpline(z, nz, k=1)
                #         downsampled_idx = downsample(data_mocks, nz_spline)
                #         downsampled_mocks = data_mocks.copy()
                #         downsampled_mocks['Z'] = data_mocks['Z'][downsampled_idx]
                #         spl_nz = spl_nofz(downsampled_mocks['Z'], fsky, mock_cosmo, Nzbins=100)
                #         downsampled_mocks['NZ'] = spl_nz(downsampled_mocks['Z'])
                #         output_dir = f"/pscratch/sd/e/epaillas/summit_lightcones/HOD/{args.survey}/{args.tracer}/{args.hod_prior}/c{cosmo:03}_ph{phase:03}"
                #         Path.mkdir(Path(output_dir), parents=True, exist_ok=True)
                #         output_fn = (
                #             Path(output_dir)
                #             / f"AbacusSummit_lightcone_c{cosmo:03}_ph{phase:03}_hod{hod:03}.npy"
                #         )
                #         np.save(output_fn, downsampled_mocks)

                #     else:
                #         raise NotImplementedError

