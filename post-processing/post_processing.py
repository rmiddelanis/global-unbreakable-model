import os

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from misc.helpers import xr_average_over_rp


def compute_poverty_increase(cat_info_data_, results_data_, hazard_protection_):
    groupby = ['iso3', 'hazard', 'rp']
    if 'hs' in cat_info_data_.dims:
        groupby.append('hs')
    if 'vs' in cat_info_data_.dims:
        groupby.append('vs')
    num_people = cat_info_data_.n * results_data_.pop

    poverty_increases_pl215 = num_people.where(
        (cat_info_data_.min_c <= cat_info_data_.pl215) & (cat_info_data_.c > cat_info_data_.pl215)
    ).stack(new=groupby).groupby('new').sum(...).unstack('new')
    poverty_increases_ = xr_average_over_rp(poverty_increases_pl215, hazard_protection_).expand_dims(pov_line=['pl215'])
    poverty_increases_ = poverty_increases_.round(0).reindex(pov_line=['pl215', 'all'])
    poverty_increases_.loc[dict(pov_line='all')] = poverty_increases_.loc[dict(pov_line='pl215')]

    if 'spl' in results_data_:
        pov_increase_spl = num_people.where(
            (cat_info_data_.min_c <= cat_info_data_.spl) & (cat_info_data_.min_c > cat_info_data_.pl215) &
            (cat_info_data_.spl > cat_info_data_.pl215) & (cat_info_data_.c > cat_info_data_.spl)
        ).stack(new=groupby).groupby('new').sum(...).unstack('new')
        pov_increase_spl = xr_average_over_rp(pov_increase_spl, hazard_protection_).round(0)

        poverty_increases_ = poverty_increases_.reindex(pov_line=['pl215', 'spl', 'all'])
        poverty_increases_.loc[dict(pov_line='spl')] = pov_increase_spl
        poverty_increases_.loc[dict(pov_line='all')] = poverty_increases_.loc[dict(pov_line='pl215')] + pov_increase_spl
    return poverty_increases_


def preprocess_simulation_data(simulation_outputs_dir_, outpath=None, exclude_scenarios=None):
    simulation_paths = {}
    for dir in os.listdir(simulation_outputs_dir_):
        if not os.path.isdir(os.path.join(simulation_outputs_dir_, dir)):
            continue
        try:
            scenario = dir
            try:
                int(scenario.split('_')[0])
                scenario = '_'.join(scenario.split('_')[1:])
            except ValueError:
                scenario = dir
            scenario_path = os.path.join(simulation_outputs_dir_, dir)
            if 'baseline' in scenario:
                simulation_paths[(scenario, 0, 0)] = scenario_path
            elif exclude_scenarios is not None and scenario in exclude_scenarios:
                continue
            else:
                horizontal_scopes = sorted(os.listdir(scenario_path))
                for horizontal_scope in horizontal_scopes:
                    try:
                        hs = int(float(horizontal_scope.split('-')[-1]) * 100)
                    except ValueError:
                        continue
                    horizontal_scope_path = os.path.join(scenario_path, horizontal_scope)
                    vertical_scopes = sorted(os.listdir(horizontal_scope_path))
                    for vertical_scope in vertical_scopes:
                        try:
                            vs = float(vertical_scope)
                            if scenario in ['post_disaster_support', 'insurance']:
                                vs = int(vs * 100)
                            elif scenario in ['reduce_exposure', 'reduce_vulnerability', 'scale_non_diversified_income']:
                                vs = int(round((1 - vs) * 100, 0))
                        except ValueError:
                            continue
                        simulation_paths[(scenario, hs, vs)] = os.path.join(horizontal_scope_path, vertical_scope)
        except ValueError:
            print(f"Skipping dir {dir} due to ValueError")
            continue

    cat_info_data_, macro_data_, results_data_, poverty_increases_, hazard_protections_ = None, None, None, None, None
    for (scenario, hs, vs), path in tqdm.tqdm(simulation_paths.items(), desc="Loading simulation data"):
        sim_cat_info_data = pd.read_csv(os.path.join(path, "iah.csv"), index_col=[0, 1, 2, 3, 4, 5])
        sim_macro_data = pd.read_csv(os.path.join(path, "macro.csv"), index_col=[0, 1, 2])
        sim_results_data = pd.read_csv(os.path.join(path, "results.csv"), index_col=0)
        sim_hazard_protection = pd.read_csv(os.path.join(path, "scenario__hazard_protection.csv"), index_col=[0, 1])

        if 'spl' in sim_results_data.columns:
            sim_results_data['spl'] *= 365
            sim_cat_info_data = pd.merge(sim_cat_info_data, sim_results_data.spl, left_index=True, right_index=True, how='left')

        sim_cat_info_data['min_c'] = sim_cat_info_data.c - sim_cat_info_data.dC_max
        sim_cat_info_data['pl215'] = 2.15 * 365

        sim_cat_info_data = xr.Dataset.from_dataframe(sim_cat_info_data)
        sim_macro_data = xr.Dataset.from_dataframe(sim_macro_data)
        sim_results_data = xr.Dataset.from_dataframe(sim_results_data)
        sim_hazard_protection = xr.Dataset.from_dataframe(sim_hazard_protection)

        sim_poverty_increase = compute_poverty_increase(sim_cat_info_data, sim_results_data, sim_hazard_protection)

        sim_cat_info_data = sim_cat_info_data.assign_coords(vs=[vs], hs=[hs], policy=[scenario])
        sim_cat_info_data = sim_cat_info_data.stack(concat_dim=['hs', 'vs', 'policy'])
        sim_macro_data = sim_macro_data.assign_coords(vs=[vs], hs=[hs], policy=[scenario])
        sim_macro_data = sim_macro_data.stack(concat_dim=['hs', 'vs', 'policy'])
        sim_results_data = sim_results_data.assign_coords(vs=[vs], hs=[hs], policy=[scenario])
        sim_results_data = sim_results_data.stack(concat_dim=['hs', 'vs', 'policy'])
        sim_poverty_increase = sim_poverty_increase.expand_dims(['hs', 'vs', 'policy']).assign_coords(vs=[vs], hs=[hs], policy=[scenario])
        sim_poverty_increase = sim_poverty_increase.stack(concat_dim=['hs', 'vs', 'policy'])
        sim_hazard_protection = sim_hazard_protection.assign_coords(vs=[vs], hs=[hs], policy=[scenario])
        sim_hazard_protection = sim_hazard_protection.stack(concat_dim=['hs', 'vs', 'policy'])
        if cat_info_data_ is None:
            cat_info_data_ = sim_cat_info_data
            macro_data_ = sim_macro_data
            results_data_ = sim_results_data
            poverty_increases_ = sim_poverty_increase
            hazard_protections_ = sim_hazard_protection
        else:
            cat_info_data_ = xr.concat([cat_info_data_, sim_cat_info_data], dim='concat_dim')
            macro_data_ = xr.concat([macro_data_, sim_macro_data], dim='concat_dim')
            results_data_ = xr.concat([results_data_, sim_results_data], dim='concat_dim')
            poverty_increases_ = xr.concat([poverty_increases_, sim_poverty_increase], dim='concat_dim')
            hazard_protections_ = xr.concat([hazard_protections_, sim_hazard_protection], dim='concat_dim')
    
    cat_info_data_ = cat_info_data_.unstack('concat_dim')
    macro_data_ = macro_data_.unstack('concat_dim')
    results_data_ = results_data_.unstack('concat_dim')
    poverty_increases_ = poverty_increases_.unstack('concat_dim')
    hazard_protections_ = hazard_protections_.unstack('concat_dim')

    cat_info_data_['t_reco_95'] = np.log(1 / .05) / cat_info_data_.lambda_h

    if outpath:
        cat_info_data_.to_netcdf(os.path.join(outpath, f'ensemble_cat_info.nc'))
        macro_data_.to_netcdf(os.path.join(outpath, f'ensemble_macro.nc'))
        results_data_.to_netcdf(os.path.join(outpath, f'ensemble_results.nc'))
        poverty_increases_.to_netcdf(os.path.join(outpath, f'ensemble_poverty_increases.nc'))
        hazard_protections_.to_netcdf(os.path.join(outpath, f'ensemble_hazard_protections.nc'))

    return cat_info_data_, macro_data_, results_data_, poverty_increases_, hazard_protections_