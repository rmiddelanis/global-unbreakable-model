"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model. It is based on
  previous work by Adrien Vogt-Schilb, Jinqiang Chen, Brian Walsh,
  and Jun Rentschler. See https://github.com/walshb1/gRIMM.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
from unbreakable.misc.helpers import get_population_scope_indices
from unbreakable.scenario.prepare_scenario import prepare_scenario
from unbreakable.model.lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd
import time
import yaml

warnings.filterwarnings("always", category=UserWarning)


def run_model(settings: dict):
    """
    Executes the global Unbreakable model using the provided settings.

    Args:
        settings (dict): Dictionary containing model and scenario parameters.

    Returns:
        None

    Notes:
        - The function generates scenario data, reshapes inputs, computes disaster impacts,
          and calculates post-disaster responses.
        - Results are saved to the specified output directory.
    """


    print("Running global Unbreakable model\n")

    model_params, scenario_params = settings['model_params'], settings['scenario_params']

    outpath_ = model_params['outpath']
    if not os.path.exists(outpath_):
        os.makedirs(outpath_)

    # Run parameters
    run_params = model_params['run_params']
    pds_params = model_params['pds_params']

    num_cores = run_params.get('num_cores', None)

    # Options and parameters
    event_level = ["iso3", "hazard", "rp"]  # levels of index at which one event happens
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    poor_cat = 'q1'

    # Generate scenario data
    macro, cat_info, hazard_ratios, hazard_protection = prepare_scenario(scenario_params=scenario_params, outpath=outpath_)

    # reshape macro and cat_info to event level, move hazard_ratios data to cat_info_event
    macro_event, cat_info_event = reshape_input(
        macro=macro,
        cat_info=cat_info,
        hazard_ratios=hazard_ratios,
        event_level=event_level,
    )

    # calculate the potential damage to capital, and consumption
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
        scope=get_population_scope_indices(pds_params['pds_scope'], cat_info_event_ia),
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

    # aggregate to event-level, computes results
    results = prepare_output(
        macro=macro,
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        event_level=event_level,
        hazard_protection_=hazard_protection,
        is_local_welfare=True,
    )

    sim_outpath = os.path.join(outpath_, 'simulation_outputs')

    if not os.path.exists(sim_outpath):
        os.makedirs(sim_outpath)

    # save macro_event
    macro_event.to_csv(sim_outpath + '/macro.csv', encoding="utf-8",
                       header=True)

    # save cat_info_event_iah
    cat_info_event_iah.to_csv(sim_outpath + '/iah.csv', encoding="utf-8",
                              header=True)

    # Save results
    results.to_csv(sim_outpath + '/results.csv', encoding="utf-8",
                   header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('settings', type=str, help='Provide settings file to run the model.')

    args = parser.parse_args()

    with open(args.settings, "r") as file:
        settings_dict = yaml.safe_load(file)

    run_model(settings_dict)