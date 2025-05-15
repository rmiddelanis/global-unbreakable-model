import os

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from misc.helpers import xr_average_over_rp, calculate_average_recovery_duration, average_over_rp


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


def preprocess_simulation_data(simulation_outputs_dir_, store_preprocessed=False, exclude_scenarios=None, concat_policy_parameters=False):
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
                simulation_paths[(scenario, 0, 0, 0)] = scenario_path
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
                                vs_sign = 1
                            else:
                                vs_sign = 1 if vs >= 1 else -1
                                vs = int(round(abs(1 - vs) * 100, 0))
                        except ValueError:
                            continue
                        simulation_paths[(scenario, hs, vs, vs_sign)] = os.path.join(horizontal_scope_path, vertical_scope)
        except ValueError:
            print(f"Skipping dir {dir} due to ValueError")
            continue

    cat_info_res_, event_res_, macro_res_, poverty_res_, hazard_prot_sc_, cat_info_sc_, macro_sc_, hazard_ratios_sc_ = [], [], [], [], [], [], [], []
    for (scenario, hs, vs, vs_sign), path in tqdm.tqdm(simulation_paths.items(), desc="Loading simulation data"):
        sim_res_cat_info_data = pd.read_csv(os.path.join(path, "iah.csv"), index_col=[0, 1, 2, 3, 4, 5])
        sim_res_macro_data = pd.read_csv(os.path.join(path, "macro.csv"), index_col=[0, 1, 2])
        sim_res_results_data = pd.read_csv(os.path.join(path, "results.csv"), index_col=0)
        sim_sc_hazard_protection = pd.read_csv(os.path.join(path, "scenario__hazard_protection.csv"), index_col=[0, 1])
        sim_sc_cat_info_data = pd.read_csv(os.path.join(path, "scenario__cat_info.csv"), index_col=[0, 1])
        sim_sc_macro_data = pd.read_csv(os.path.join(path, "scenario__macro.csv"), index_col=[0])
        sim_sc_hazard_ratios = pd.read_csv(os.path.join(path, "scenario__hazard_ratios.csv"), index_col=[0, 1, 2, 3])

        sim_res_cat_info_data['min_c'] = sim_res_cat_info_data.c - sim_res_cat_info_data.dC_max
        sim_res_cat_info_data['pl215'] = 2.15 * 365
        sim_res_cat_info_data['t_reco_95'] = np.log(1 / .05) / sim_res_cat_info_data.lambda_h

        if 'spl' in sim_res_results_data.columns:
            sim_res_results_data['spl'] *= 365
            sim_res_cat_info_data = pd.merge(sim_res_cat_info_data, sim_res_results_data.spl, left_index=True, right_index=True, how='left')

        if 'risk_to_consumption' not in sim_res_results_data.columns:
            if 'dc' not in sim_res_results_data.columns:
                groupby = ['iso3', 'hazard', 'rp']
                dc_event = (sim_res_cat_info_data.dc * sim_res_cat_info_data.n).groupby(groupby).sum()
                sim_res_results_data['dc'] = average_over_rp(dc_event, sim_sc_hazard_protection).groupby('iso3').sum().values
                sim_res_results_data['dc_tot'] = sim_res_results_data['dc'] * sim_res_results_data['pop']
            sim_res_results_data['risk_to_consumption'] = sim_res_results_data.dc / sim_res_results_data.gdp_pc_pp

        sim_res_results_data['t_reco_95'] = calculate_average_recovery_duration(sim_res_cat_info_data, 'iso3', sim_sc_hazard_protection, None)
        if 'risk' in sim_res_results_data.columns:
            sim_res_results_data.rename(columns={'risk': 'risk_to_wellbeing'}, inplace=True)

        sim_res_cat_info_data = xr.Dataset.from_dataframe(sim_res_cat_info_data)
        sim_res_macro_data = xr.Dataset.from_dataframe(sim_res_macro_data)
        sim_res_results_data = xr.Dataset.from_dataframe(sim_res_results_data)
        sim_sc_hazard_protection = xr.Dataset.from_dataframe(sim_sc_hazard_protection)
        sim_sc_cat_info_data = xr.Dataset.from_dataframe(sim_sc_cat_info_data)
        sim_sc_macro_data = xr.Dataset.from_dataframe(sim_sc_macro_data)
        sim_sc_hazard_ratios = xr.Dataset.from_dataframe(sim_sc_hazard_ratios)

        sim_poverty_increase = compute_poverty_increase(sim_res_cat_info_data, sim_res_results_data, sim_sc_hazard_protection)

        sim_poverty_increase_agg = (sim_poverty_increase.sum('hazard') / sim_res_results_data['pop']).to_dataset(dim='pov_line')
        rename_dict = {'all': 'risk_to_all_poverty', 'pl215': 'risk_to_extreme_poverty'}
        if 'spl' in sim_poverty_increase.pov_line.values:
            rename_dict['spl'] = 'risk_to_societal_poverty'
        sim_poverty_increase_agg = sim_poverty_increase_agg.rename(rename_dict)
        sim_res_results_data = xr.merge([sim_res_results_data, sim_poverty_increase_agg])

        if concat_policy_parameters:
            coords_dict = {'policy': [f"{scenario}/{hs}/{'-' if vs_sign == -1 else '+'}{vs}"]}
        else:
            coords_dict = {'hs': [hs], 'vs': [vs], 'vs_sign': [vs_sign], 'policy': [scenario]}
        sim_res_cat_info_data = sim_res_cat_info_data.assign_coords(coords_dict)
        sim_res_cat_info_data = sim_res_cat_info_data.stack(concat_dim=list(coords_dict.keys()))
        sim_res_macro_data = sim_res_macro_data.assign_coords(coords_dict)
        sim_res_macro_data = sim_res_macro_data.stack(concat_dim=list(coords_dict.keys()))
        sim_res_results_data = sim_res_results_data.assign_coords(coords_dict)
        sim_res_results_data = sim_res_results_data.stack(concat_dim=list(coords_dict.keys()))
        sim_poverty_increase = sim_poverty_increase.expand_dims(list(coords_dict.keys())).assign_coords(coords_dict)
        sim_poverty_increase = sim_poverty_increase.stack(concat_dim=list(coords_dict.keys()))
        sim_sc_hazard_protection = sim_sc_hazard_protection.assign_coords(coords_dict)
        sim_sc_hazard_protection = sim_sc_hazard_protection.stack(concat_dim=list(coords_dict.keys()))
        sim_sc_cat_info_data = sim_sc_cat_info_data.assign_coords(coords_dict)
        sim_sc_cat_info_data = sim_sc_cat_info_data.stack(concat_dim=list(coords_dict.keys()))
        sim_sc_macro_data = sim_sc_macro_data.assign_coords(coords_dict)
        sim_sc_macro_data = sim_sc_macro_data.stack(concat_dim=list(coords_dict.keys()))
        sim_sc_hazard_ratios = sim_sc_hazard_ratios.assign_coords(coords_dict)
        sim_sc_hazard_ratios = sim_sc_hazard_ratios.stack(concat_dim=list(coords_dict.keys()))

        cat_info_res_.append(sim_res_cat_info_data)
        event_res_.append(sim_res_macro_data)
        macro_res_.append(sim_res_results_data)
        poverty_res_.append(sim_poverty_increase)
        hazard_prot_sc_.append(sim_sc_hazard_protection)
        cat_info_sc_.append(sim_sc_cat_info_data)
        macro_sc_.append(sim_sc_macro_data)
        hazard_ratios_sc_.append(sim_sc_hazard_ratios)

    cat_info_res_ = xr.concat(cat_info_res_, dim='concat_dim').unstack('concat_dim')
    event_res_ = xr.concat(event_res_, dim='concat_dim').unstack('concat_dim')
    macro_res_ = xr.concat(macro_res_, dim='concat_dim').unstack('concat_dim')
    poverty_res_ = xr.concat(poverty_res_, dim='concat_dim').unstack('concat_dim')
    hazard_prot_sc_ = xr.concat(hazard_prot_sc_, dim='concat_dim').unstack('concat_dim')
    cat_info_sc_ = xr.concat(cat_info_sc_, dim='concat_dim').unstack('concat_dim')
    macro_sc_ = xr.concat(macro_sc_, dim='concat_dim').unstack('concat_dim')
    hazard_ratios_sc_ = xr.concat(hazard_ratios_sc_, dim='concat_dim').unstack('concat_dim')

    if store_preprocessed:
        outpath = os.path.join(simulation_outputs_dir_, '_preprocessed_simulation_output')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        cat_info_res_.to_netcdf(os.path.join(outpath, f'res_cat_info.nc'))
        event_res_.to_netcdf(os.path.join(outpath, f'res_event.nc'))
        macro_res_.to_netcdf(os.path.join(outpath, f'res_macro.nc'))
        poverty_res_.to_netcdf(os.path.join(outpath, f'res_poverty.nc'))
        hazard_prot_sc_.to_netcdf(os.path.join(outpath, f'sc_hazard_prot.nc'))
        cat_info_sc_.to_netcdf(os.path.join(outpath, f'sc_cat_info.nc'))
        macro_sc_.to_netcdf(os.path.join(outpath, f'sc_macro.nc'))
        hazard_ratios_sc_.to_netcdf(os.path.join(outpath, f'sc_hazard_ratios.nc'))

    return cat_info_res_, event_res_, macro_res_, poverty_res_, hazard_prot_sc_, cat_info_sc_, macro_sc_, hazard_ratios_sc_