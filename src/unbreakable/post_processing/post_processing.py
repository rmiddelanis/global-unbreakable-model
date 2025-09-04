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
    for adjusted_poverty_lines in [True, False]:
        adj_string = "_adj" if adjusted_poverty_lines else ""

        if "extr_pov_line" + adj_string in macro_res_:
            extr_pov_incr = num_people.where(
                (cat_info_res_.c - cat_info_res_.dC_max - macro_res_["extr_pov_line" + adj_string] * 365 < 0) &                             # fell below extreme pov line
                (cat_info_res_.c - macro_res_["extr_pov_line" + adj_string] * 365 >= 0)                                                     # was above extreme pov line before disaster
            ).sum(['income_cat', 'affected_cat', 'helped_cat'])
            res["extr_pov_incr" + adj_string] = xr.DataArray.from_series(
                average_over_rp(extr_pov_incr.to_series(), hazard_prot_sc_.protection.to_series()).round(0)
            )
            if "soc_pov_line" + adj_string in macro_res_:
                soc_pov_incr = num_people.where(
                    (cat_info_res_.c - cat_info_res_.dC_max - macro_res_["soc_pov_line" + adj_string] * 365 < 0) &                          # fell below social pov line
                    (cat_info_res_.c - macro_res_["soc_pov_line" + adj_string] * 365 >= 0) &                                                # was above social pov line before disaster
                    (cat_info_res_.c - cat_info_res_.dC_max - macro_res_["extr_pov_line" + adj_string] * 365 > 0) &                         # did not fall below extreme pov line
                    ((macro_res_["extr_pov_line" + adj_string] < macro_res_["soc_pov_line" + adj_string]).broadcast_like(cat_info_res_))    # extreme pov line is below social pov line
                ).sum(['income_cat', 'affected_cat', 'helped_cat'])
                res["soc_pov_incr" + adj_string] = xr.DataArray.from_series(
                    average_over_rp(soc_pov_incr.to_series(), hazard_prot_sc_.protection.to_series()).round(0)
                )
                total_pov_incr = res["extr_pov_incr" + adj_string] + res["soc_pov_incr" + adj_string]
                res["total_pov_incr" + adj_string] = total_pov_incr
    if len(res.data_vars) > 0:
        return res
    print(f"Warning: Could not compute poverty increase as extreme poverty lines are missing in macro results.")


def process_simulation_ensemble(simulation_outputs_dir_, store_preprocessed=False, exclude_scenarios=None,
                                concat_policy_parameters=False):
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
        if os.path.exists(os.path.join(dir_path, "results.csv")):
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
        "res_cat_info": ("iah.csv", [0, 1, 2, 3, 4, 5]),
        "res_event": ("macro.csv", [0, 1, 2]),
        "res_macro": ("results.csv", 0),
        "sc_hazard_prot": ("scenario__hazard_protection.csv", [0, 1]),
        "sc_cat_info": ("scenario__cat_info.csv", [0, 1]),
        "sc_macro": ("scenario__macro.csv", [0]),
        "sc_hazard_ratios": ("scenario__hazard_ratios.csv", [0, 1, 2, 3])
    }

    results_lists = {key: [] for key in dataset_specs.keys()}
    results_lists["res_poverty"] = []

    for (scenario, hs, vs, vs_sign), path in tqdm.tqdm(simulation_paths.items(), desc="Loading simulation data"):
        loaded_datasets = {}
        for key, (fname, idx) in dataset_specs.items():
            loaded_datasets[key] = pd.read_csv(os.path.join(path, fname), index_col=idx)

        # Compute additional variables
        loaded_datasets["res_cat_info"]["t_reco_95"] = np.log(1 / .05) / loaded_datasets["res_cat_info"].lambda_h
        loaded_datasets["res_macro"]["t_reco_95"] = calculate_average_recovery_duration(
            loaded_datasets["res_cat_info"], 'iso3', loaded_datasets["sc_hazard_prot"], None
        )

        for key in loaded_datasets.keys():
            loaded_datasets[key] = xr.Dataset.from_dataframe(loaded_datasets[key])

        # Compute poverty increase
        sim_poverty_increase = compute_poverty_increase(
            loaded_datasets["res_cat_info"], loaded_datasets["res_macro"], loaded_datasets["sc_hazard_prot"]
        )
        sim_poverty_increase_agg = (sim_poverty_increase.sum('hazard') / loaded_datasets["res_macro"]['pop']).rename({v: v.replace('_incr', '_risk') for v in list(sim_poverty_increase.data_vars)})
        loaded_datasets["res_macro"] = xr.merge([loaded_datasets["res_macro"], sim_poverty_increase_agg])
        loaded_datasets["res_poverty"] = sim_poverty_increase

        if concat_policy_parameters:
            coords_dict = {'policy': [f"{scenario}/{hs}/{'-' if vs_sign == -1 else '+'}{vs}"]}
        else:
            coords_dict = {'hs': [hs], 'vs': [vs], 'policy': [scenario]}

        for key, ds in loaded_datasets.items():
            if isinstance(ds, xr.DataArray):
                ds = ds.expand_dims(list(coords_dict.keys())).assign_coords(coords_dict)
            ds = ds.assign_coords(coords_dict)
            ds = ds.stack(concat_dim=list(coords_dict.keys()))
            ds = ds.assign_coords(vs_sign=vs_sign)
            results_lists[key].append(ds)

    for key in results_lists:
        results_lists[key] = xr.concat(results_lists[key], dim='concat_dim').unstack('concat_dim')

    drop_vars = []
    if (np.unique(results_lists["res_cat_info"].hs) == 0).all():
        drop_vars.append('hs')
    if (np.unique(results_lists["sc_cat_info"].vs) == 0).all():
        drop_vars.extend(['vs', 'vs_sign'])

    for key in results_lists:
        results_lists[key] = results_lists[key].drop_vars(drop_vars)
        results_lists[key] = results_lists[key].squeeze(drop=True)

    if store_preprocessed:
        outpath = os.path.join(simulation_outputs_dir_, '_preprocessed_simulation_output')
        os.makedirs(outpath, exist_ok=True)
        for key, ds in results_lists.items():
            ds.to_netcdf(os.path.join(outpath, f"{key}.nc"))

    return tuple(results_lists[key] for key in ['res_cat_info', 'res_event', 'res_macro', 'res_poverty',
                                                'sc_hazard_prot', 'sc_cat_info', 'sc_macro', 'sc_hazard_ratios'])
