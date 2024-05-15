import argparse

from lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd
import time

warnings.filterwarnings("always", category=UserWarning)


def run_model(climate_scenario_, scenario_, option_fee_, option_pds_, simulation_name_, exclude_hazards_, countries_):
    print(scenario_)
    print(f'optionFee ={option_fee_}, optionPDS ={option_pds_}, optionB ={option_b}, optionT ={option_t}')

    # Options and parameters
    econ_scope = "iso3"  # province, deparmtent
    event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens
    default_rp = "default_rp"  # return period to use when no rp is provided (mind that this works with protection)
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")
    poor_cat = 'q1'

    # read data
    # macro-economic country economic data
    macro = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', climate_scenario_, scenario_, "scenario__macro.csv"),
                        index_col=econ_scope)

    # consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
    cat_info = pd.read_csv(
        os.path.join(intermediate_dir, 'scenarios', climate_scenario_, scenario_, "scenario__cat_info.csv"),
        index_col=[econ_scope, "income_cat"])

    # exposure, vulnerability, and access to early warning by country, hazard, return period, income category
    hazard_ratios = pd.read_csv(
        os.path.join(intermediate_dir, 'scenarios', climate_scenario_, scenario_, "scenario__hazard_ratios.csv"),
        index_col=event_level + ["income_cat"])

    hazard_protection = pd.read_csv(
        os.path.join(intermediate_dir, 'scenarios', climate_scenario_, scenario_, "scenario__hazard_protection.csv"),
        index_col=[econ_scope, "hazard"])

    if len(exclude_hazards_) > 0:
        hazard_ratios = hazard_ratios[~hazard_ratios.index.get_level_values('hazard').isin(exclude_hazards_.split('+'))]
        hazard_protection = hazard_protection[
            ~hazard_protection.index.get_level_values('hazard').isin(exclude_hazards_.split('+'))]

    if len(countries_) > 0:
        countries = countries_.split('+')
        macro = macro.loc[countries]
        cat_info = cat_info.loc[countries]
        hazard_ratios = hazard_ratios.loc[countries]
        hazard_protection = hazard_protection.loc[countries]

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
        option_fee=option_fee_,
        option_t=option_t,
        option_pds=option_pds_,
        option_b=option_b,
        loss_measure="dk_reco",
        fraction_inside=1,
        share_insured=.25,
    )

    cat_info_event_iah, macro_event = compute_dw_new(
        cat_info_event_iah=cat_info_event_iah,
        macro_event=macro_event,
        event_level_=event_level,
        capital_t=20,
        delta_c_h_max=np.nan,
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
        econ_scope=econ_scope,
        default_rp=default_rp,
        is_local_welfare=True,
        return_stats=True,
        # long_term_horizon_=long_term_horizon,
    )

    date_time_string = time.strftime("%Y-%m-%d_%H-%M_")
    folder_name = f'output/scenarios/{climate_scenario_}/{date_time_string + scenario_}'
    if simulation_name_ != '':
        folder_name = folder_name + f'_{simulation_name_}'
    if len(exclude_hazards_) > 0:
        folder_name = folder_name + f'_ex_{exclude_hazards_}'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save macro_event
    macro_event.to_csv(folder_name + '/macro_' + option_fee_ + '_' + option_pds_ + '.csv', encoding="utf-8",
                       header=True)

    # save cat_info_event_iah
    cat_info_event_iah.to_csv(folder_name + '/iah_' + option_fee_ + '_' + option_pds_ + '.csv', encoding="utf-8",
                              header=True)

    # Save results
    results.to_csv(folder_name + '/results_' + option_fee_ + '_' + option_pds_ + '.csv', encoding="utf-8",
                   header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--climate_scenario', type=str, help='Climate scenario from CDRI GIRI report.')
    parser.add_argument('--scenarios', type=str, default='baselineEW-2018', help='Scenarios')
    parser.add_argument('--option_fee', type=str, default='tax', help='Fee option to fund PDS.')
    parser.add_argument('--countries', type=str, default='', help='Select countries for the analysis. Use + to separate countries. If empty, all countries are selected.')
    parser.add_argument('--option_pds', type=str, default='unif_poor', help='PDS option.')
    parser.add_argument('--simulation_name', type=str, default='', help='Name of the simluation.')
    parser.add_argument('--exclude_hazard', type=str, default='', help='Exclude hazards from analysis.')
    parser.add_argument('--do_not_execute', action='store_true', help='Do not execute the model.')

    args = parser.parse_args()

    climate_scenario = args.climate_scenario
    if climate_scenario not in ['Existing_climate', 'Upper_bound', 'Lower_bound']:
        raise ValueError(f'Invalid climate scenario: {climate_scenario}')

    exclude_hazards = args.exclude_hazard

    # define directory
    model = os.getcwd()  # get current directory
    input_dir = model + '/inputs/'  # get inputs data directory
    intermediate_dir = model + '/intermediate/'  # get outputs data directory

    scenarios = args.scenarios
    if scenarios == 'all':
        scenarios = os.listdir(os.path.join(intermediate_dir, 'scenarios', climate_scenario))
        scenarios = [s for s in scenarios if os.path.isdir(os.path.join(intermediate_dir, 'scenarios', climate_scenario, s))]
    else:
        scenarios = scenarios.split('+')

    option_fee = args.option_fee
    option_pds = args.option_pds
    if option_fee == "insurance_premium":
        option_b = 'unlimited'
        option_t = 'perfect'
    else:
        option_b = 'data'
        option_t = 'data'

    if not args.do_not_execute:
        for scenario in scenarios:
            run_model(
                climate_scenario_=climate_scenario,
                scenario_=scenario,
                option_fee_=option_fee,
                option_pds_=option_pds,
                simulation_name_=args.simulation_name,
                exclude_hazards_=exclude_hazards,
                countries_=args.countries,
            )
