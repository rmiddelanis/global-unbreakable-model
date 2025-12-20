import os

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unbreakable.misc.helpers import average_over_rp, calculate_average_recovery_duration


def compute_poverty_increase(cat_info_res_, macro_res_, hazard_prot_sc_):
    num_people = cat_info_res_.n * macro_res_.pop
    res = xr.Dataset()
    if isinstance(hazard_prot_sc_, xr.Dataset):
        hazard_prot_sc_ = hazard_prot_sc_.protection.to_series()
    for adjusted_poverty_lines in [True, False]:
        adj_string = "_adj" if adjusted_poverty_lines else ""

        if "extr_pov_line" + adj_string in macro_res_:
            extr_pov_incr = num_people.where(
                cat_info_res_['time_below_extr_pov_line' + adj_string].where(cat_info_res_['time_below_extr_pov_line' + adj_string] != np.inf, 0) > 0
            ).sum(['income_cat', 'affected_cat', 'helped_cat'])
            res["extr_pov_incr" + adj_string] = xr.DataArray.from_series(
                average_over_rp(extr_pov_incr.to_series(), hazard_prot_sc_).round(0)
            )
            extr_pov_time_incr = cat_info_res_[[f'time_below_extr_pov_line'+adj_string, 'n']].to_dataframe().fillna(0).replace(np.inf, 0)
            extr_pov_time_incr = extr_pov_time_incr.prod(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() / extr_pov_time_incr.n.groupby(['iso3', 'hazard', 'rp']).sum()
            res["extr_pov_time_incr" + adj_string] = xr.DataArray.from_series(
                average_over_rp(extr_pov_time_incr, hazard_prot_sc_) * 365
            )
            if "soc_pov_line" + adj_string in macro_res_:
                soc_pov_incr = num_people.where(
                    (cat_info_res_['time_below_soc_pov_line' + adj_string].where(cat_info_res_['time_below_soc_pov_line' + adj_string] != np.inf, 0) > 0) &
                    (cat_info_res_['time_below_extr_pov_line' + adj_string].where(cat_info_res_['time_below_extr_pov_line' + adj_string] != np.inf, 0) == 0)
                ).sum(['income_cat', 'affected_cat', 'helped_cat'])
                res["soc_pov_incr" + adj_string] = xr.DataArray.from_series(
                    average_over_rp(soc_pov_incr.to_series(), hazard_prot_sc_).round(0)
                )
                soc_pov_time_incr = cat_info_res_[[f'time_below_soc_pov_line' + adj_string, f'time_below_extr_pov_line' + adj_string, 'n']].to_dataframe()
                soc_pov_time_incr[f'time_below_soc_pov_line' + adj_string] -= soc_pov_time_incr[f'time_below_extr_pov_line' + adj_string]
                soc_pov_time_incr = soc_pov_time_incr.fillna(0).replace(np.inf, 0)[[f'time_below_soc_pov_line' + adj_string, 'n']]
                soc_pov_time_incr = soc_pov_time_incr.prod(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() / soc_pov_time_incr.n.groupby(['iso3', 'hazard', 'rp']).sum()
                res["soc_pov_time_incr" + adj_string] = xr.DataArray.from_series(
                    average_over_rp(soc_pov_time_incr, hazard_prot_sc_) * 365
                )
                res["total_pov_incr" + adj_string] = res["extr_pov_incr" + adj_string] + res["soc_pov_incr" + adj_string]
                res["total_pov_time_incr" + adj_string] = res["extr_pov_time_incr" + adj_string] + res["soc_pov_time_incr" + adj_string]
    if len(res.data_vars) > 0:
        return res
    print(f"Warning: Could not compute poverty increase as extreme poverty lines are missing in macro results.")


def process_simulation_ensemble(simulation_outputs_dir_, store_preprocessed=False, exclude_scenarios=None,
                                concat_policy_parameters=False, drop_vars_=None):
    preprocessed_outpath = os.path.join(simulation_outputs_dir_, '_preprocessed_simulation_output')
    if os.path.exists(preprocessed_outpath):
        results = {}
        for ds_name in ['res_cat_info', 'res_event', 'res_macro', 'res_poverty', 'sc_hazard_prot', 'sc_cat_info', 'sc_macro', 'sc_hazard_ratios']:
            ds_path = os.path.join(preprocessed_outpath, f"{ds_name}.nc")
            if os.path.exists(ds_path):
                ds = xr.load_dataset(ds_path)
                if len(ds.data_vars) == 1:
                    ds = ds[list(ds.data_vars)[0]]
                results[ds_name] = ds
        if len(results) > 0:
            return results
    simulation_paths = {}
    for dir in os.listdir(simulation_outputs_dir_):
        dir_path = os.path.join(simulation_outputs_dir_, dir)
        if not os.path.isdir(dir_path):
            continue
        scenario = dir
        try:
            int(scenario.split('_')[0])
            scenario = '_'.join(scenario.split('_')[1:])
        except ValueError:
            pass
        if exclude_scenarios and scenario in exclude_scenarios:
            continue
        if os.path.exists(os.path.join(dir_path, "simulation_outputs")):
            simulation_paths[(scenario, 0, 0, 0)] = dir_path
        else:
            for hs_dir in sorted(os.listdir(dir_path)):
                try:
                    hs = int(float(hs_dir.split('-')[-1]) * 100)
                except ValueError:
                    continue
                hs_path = os.path.join(dir_path, hs_dir)
                for vs_dir in sorted(os.listdir(hs_path)):
                    try:
                        vs = float(vs_dir)
                        if scenario in ['post_disaster_support', 'insurance']:
                            vs_sign = 1
                            vs = int(vs * 100)
                        else:
                            vs_sign = 1 if vs >= 1 else -1
                            vs = int(round(abs(1 - vs) * 100, 0))
                    except ValueError:
                        continue
                    simulation_paths[(scenario, hs, vs, vs_sign)] = os.path.join(hs_path, vs_dir)

    dataset_specs = {
        "res_cat_info": ("simulation_outputs/iah.csv", [0, 1, 2, 3, 4, 5]),
        "res_event": ("simulation_outputs/macro.csv", [0, 1, 2]),
        "res_macro": ("simulation_outputs/results.csv", 0),
        "sc_hazard_prot": ("model_inputs/scenario__hazard_protection.csv", [0, 1]),
        "sc_cat_info": ("model_inputs/scenario__cat_info.csv", [0, 1]),
        "sc_macro": ("model_inputs/scenario__macro.csv", [0]),
        "sc_hazard_ratios": ("model_inputs/scenario__hazard_ratios.csv", [0, 1, 2, 3])
    }

    results = {key: [] for key in dataset_specs.keys()}

    if drop_vars_ is None:
        drop_vars_ = {}

    for (scenario, hs, vs, vs_sign), path in tqdm.tqdm(simulation_paths.items(), desc="Loading simulation data"):
        loaded_datasets = {}
        for ds_name, (fname, idx) in dataset_specs.items():
            loaded_datasets[ds_name] = pd.read_csv(os.path.join(path, fname), index_col=idx).drop(columns=drop_vars_.get(ds_name,[]))

        # Compute additional variables
        loaded_datasets["res_cat_info"]["t_reco_95"] = np.log(1 / .05) / loaded_datasets["res_cat_info"].lambda_h
        loaded_datasets["res_macro"]["t_reco_95"] = calculate_average_recovery_duration(
            loaded_datasets["res_cat_info"], 'iso3', loaded_datasets["sc_hazard_prot"], None
        )

        for ds_name in loaded_datasets.keys():
            loaded_datasets[ds_name] = xr.Dataset.from_dataframe(loaded_datasets[ds_name])

        # Compute poverty increase
        sim_poverty_increase = compute_poverty_increase(
            loaded_datasets["res_cat_info"], loaded_datasets["res_macro"], loaded_datasets["sc_hazard_prot"]
        )
        if sim_poverty_increase:
            sim_poverty_increase_agg = (sim_poverty_increase.sum('hazard') / loaded_datasets["res_macro"]['pop']).rename({v: v.replace('_incr', '_risk') for v in list(sim_poverty_increase.data_vars)})
            loaded_datasets["res_macro"] = xr.merge([loaded_datasets["res_macro"], sim_poverty_increase_agg])
            loaded_datasets["res_poverty"] = sim_poverty_increase
            if 'res_poverty' not in results:
                results["res_poverty"] = []

        if concat_policy_parameters:
            coords_dict = {'policy': [f"{scenario}/{hs}/{'-' if vs_sign == -1 else '+'}{vs}"]}
        else:
            coords_dict = {'hs': [hs], 'vs': [vs], 'policy': [scenario]}

        for ds_name, ds in loaded_datasets.items():
            if isinstance(ds, xr.DataArray):
                ds = ds.expand_dims(list(coords_dict.keys())).assign_coords(coords_dict)
            ds = ds.assign_coords(coords_dict)
            ds = ds.stack(concat_dim=list(coords_dict.keys()))
            ds = ds.assign_coords(vs_sign=vs_sign)
            results[ds_name].append(ds)

    for ds_name in results:
        results[ds_name] = xr.concat(results[ds_name], dim='concat_dim').unstack('concat_dim')

    drop_vars = []
    if (np.unique(results["res_cat_info"].hs) == 0).all():
        drop_vars.append('hs')
    if (np.unique(results["sc_cat_info"].vs) == 0).all():
        drop_vars.extend(['vs', 'vs_sign'])

    for ds_name in results:
        results[ds_name] = results[ds_name].drop_vars(drop_vars)
        results[ds_name] = results[ds_name].squeeze(drop=True)

    if store_preprocessed:
        os.makedirs(preprocessed_outpath, exist_ok=True)
        for ds_name, ds in results.items():
            ds.to_netcdf(os.path.join(preprocessed_outpath, f"{ds_name}.nc"))

    return results
