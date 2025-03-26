import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from scenario.prepare_scenario import prepare_scenario
from model.lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd
import time
import yaml

warnings.filterwarnings("always", category=UserWarning)


def run_model(settings: dict):
    print("Running global Unbreakable model\n")

    model_params, scenario_params = settings['model_params'], settings['scenario_params']

    # Run parameters
    run_params = model_params['run_params']
    pds_params = model_params['pds_params']

    outpath_ = run_params['outpath']
    num_cores = run_params.get('num_cores', None)

    # Options and parameters
    event_level = ["iso3", "hazard", "rp"]  # levels of index at which one event happens
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    poor_cat = 'q1'

    # Generate scenario data
    macro, cat_info, hazard_ratios, hazard_protection = prepare_scenario(scenario_params=scenario_params)

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
    macro_event, cat_info_event_ia = compute_dk(
        macro_event=macro_event,
        cat_info_event=cat_info_event,
        event_level=event_level,
        affected_cats=affected_cats,
    )

    # calculate the post-disaster response
    macro_event, cat_info_event_iah = compute_response(
        macro_event=macro_event,
        cat_info_event_ia=cat_info_event_ia,
        event_level=event_level,
        scope=pds_params['pds_scope'],
        targeting=pds_params['pds_targeting'],
        lending_rate=pds_params['pds_lending_rate'],
        variant=pds_params['pds_variant'],
        borrowing_ability=pds_params['pds_borrowing_ability'],
        loss_measure="dk_reco",
        covered_loss_share=pds_params['covered_loss_share'],
    )

    cat_info_event_iah, macro_event = compute_dw(
        cat_info_event_iah=cat_info_event_iah,
        macro_event=macro_event,
        event_level_=event_level,
        capital_t=50,
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
    )

    date_time_string = ''
    if run_params.get('append_date', True):
        date_time_string = time.strftime("%Y-%m-%d_%H-%M_")
    folder_name = os.path.join(outpath_, date_time_string)

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
    parser.add_argument('settings', type=str, help='Provide settings file to run the model.')

    args = parser.parse_args()

    with open(args.settings, "r") as file:
        settings_dict = yaml.safe_load(file)

    run_model(settings_dict)