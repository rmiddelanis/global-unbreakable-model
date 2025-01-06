import argparse

from lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd
import time
import yaml
from prepare_scenario import gather_data

warnings.filterwarnings("always", category=UserWarning)


def run_model(run_params, scenario_params, pds_params):
    # Run parameters
    outpath_ = run_params['outpath']
    simulation_name_ = run_params['simulation_name']
    num_cores = run_params.get('num_cores', None)

    print("Running global Unbreakable model")

    # Options and parameters
    event_level = ["iso3", "hazard", "rp"]  # levels of index at which one event happens
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")
    poor_cat = 'q1'

    if scenario_params is None:
        scenario_params = {}
    macro, cat_info, hazard_ratios, hazard_protection = gather_data(
        intermediate_dir_=scenario_params.get('intermediate_dir', None),
        force_recompute_=scenario_params.get('force_recompute', False),
        hazard_protection_=scenario_params.get('hazard_protection', "FLOPROS"),
        reduction_vul_=scenario_params.get('reduction_vul', .2),
        income_elasticity_eta_=scenario_params.get('income_elasticity_eta', 1.5),
        discount_rate_rho_=scenario_params.get('discount_rate_rho', .06),
        max_increased_spending_=scenario_params.get('max_increased_spending', .05),
        fa_threshold_=scenario_params.get('fa_threshold', .9),
        axfin_impact_=scenario_params.get('axfin_impact', .1),
        no_exposure_bias_=scenario_params.get('no_exposure_bias', False),
        reconstruction_capital_=scenario_params.get('reconstruction_capital', 'prv'),
        ew_year_=scenario_params.get('ew_year', 2018),
        ew_decade_=scenario_params.get('ew_decade', None),
        scale_self_employment_=scenario_params.get('scale_self_employment', 1),
        scale_non_diversified_income_=scenario_params.get('scale_non_diversified_income', 1),
        min_diversified_share_=scenario_params.get('min_diversified_share', 0),
        scale_gdp_pc_pp_=scenario_params.get('scale_gdp_pc_pp', 1),
        pol_opt_=scenario_params.get('pol_opt', ''),
        verbose=run_params.get('verbose', False),
        countries=scenario_params.get('countries', None),
        scale_liquidity_=scenario_params.get('scale_liquidity', 1),
        hazards_=scenario_params.get('hazards', 'all'),
    )

    # compute
    # reshape macro and cat_info to event level, move hazard_ratios data to cat_info_event
    macro_event, cat_info_event = reshape_input(
        macro=macro,
        cat_info=cat_info,
        hazard_ratios=hazard_ratios,
        event_level=event_level,
    )

    # calculate the potential damage to capital, and consumption
    # adds 'dk', 'dc', 'dc_npv_pre' to cat_info_event_ia, also adds aggregated 'dk' to macro_event
    macro_event, cat_info_event_ia = compute_dK(
        macro_event=macro_event,
        cat_info_event=cat_info_event,
        event_level=event_level,
        affected_cats=affected_cats,
    )

    # calculate the post-disaster response
    # adds 'error_incl', 'error_excl', 'max_aid', 'need', 'aid', 'unif_aid' to macro_event
    # adds 'help_received', 'help_fee', 'help_needed' to cat_info_event_ia(h)
    macro_event, cat_info_event_iah = calculate_response(
        macro_event=macro_event,
        cat_info_event_ia=cat_info_event_ia,
        event_level=event_level,
        poor_cat=poor_cat,
        helped_cats=helped_cats,
        pds_targeting=pds_params['pds_targeting'],
        pds_variant=pds_params['pds_variant'],
        pds_borrowing_ability=pds_params['pds_borrowing_ability'],
        loss_measure="dk_reco",
        pds_shareable=pds_params['pds_shareable'],
    )

    cat_info_event_iah, macro_event = compute_dw(
        cat_info_event_iah=cat_info_event_iah,
        macro_event=macro_event,
        event_level_=event_level,
        capital_t=20,
        delta_c_h_max=np.nan,
        num_cores=num_cores,
    )

    # aggregate to event-level (ie no more income_cat, helped_cat, affected_cat, n)
    # Computes results
    # results consists of all variables from macro, as well as 'aid' and 'dk' from macro_event
    results = prepare_output(
        macro=macro,
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        event_level=event_level,
        hazard_protection_=hazard_protection,
        is_local_welfare=True,
        return_stats=True,
    )

    date_time_string = ''
    if run_params.get('append_date', True):
        date_time_string = time.strftime("%Y-%m-%d_%H-%M_")
    folder_name = os.path.join(outpath_, date_time_string + simulation_name_)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save macro_event
    macro_event.to_csv(folder_name + '/macro.csv', encoding="utf-8",
                       header=True)

    # save cat_info_event_iah
    cat_info_event_iah.to_csv(folder_name + '/iah.csv', encoding="utf-8",
                              header=True)

    # Save results
    results.to_csv(folder_name + '/results.csv', encoding="utf-8",
                   header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--settings', type=str, help='Provide settings file to run the model.')

    args = parser.parse_args()

    with open(args.settings, "r") as file:
        params = yaml.safe_load(file)

    run_model(params['run_params'], params['scenario_params'], params['pds_params'])