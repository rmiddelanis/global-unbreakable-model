"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model.

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


import os
import numpy as np
import pandas as pd

from unbreakable.misc.helpers import average_over_rp


def load_giri_hazard_loss_rel(gir_filepath_, climate_scenario='Existing climate'):
    """
    Loads and processes GIRI (Global Infrastructure Risk Model and Resilience Index) hazard loss data.

    This function processes hazard loss data to compute the fraction of value destroyed for each country, hazard,
    and return period. It supports extrapolation of return periods and filtering by climate scenarios.

    Args:
        gir_filepath_ (str): Path to the GIR hazard loss data file (compressed CSV).
        climate_scenario (str): Climate scenario to filter the data. Defaults to 'Existing climate'.

    Returns:
        pd.Series: A pandas Series with a MultiIndex of ['iso3', 'hazard', 'rp'] and values representing the
        fraction of value destroyed for each country, hazard, and return period.

    Raises:
        ValueError: If invalid return periods are provided during extrapolation.

    Notes:
        - The function handles disputed territories and merges data for specific regions into their respective countries.
        - It supports adding new return periods for frequent and infrequent events.
        - Zero values are dropped from the final result.
    """
    gir_data = pd.read_csv(gir_filepath_, compression='zip')
    gir_data.rename({'value_axis_1': 'loss', 'value_axis_2': 'rp', 'iso3cd': 'iso3',
                     'country_name': 'country'}, axis=1,
                    inplace=True)
    gir_data.drop(['unit_axis_1', 'name_axis_1', 'name_axis_2'], axis=1, inplace=True)

    # drop disputed territories without distinct iso3 src
    gir_data = gir_data[~gir_data.iso3.isin(['xUK', 'xAB', 'xAC', 'xAP', 'xJK', 'xRI', 'xPI', 'xJL', 'xxx'])]

    gir_data.country = gir_data.country.replace({'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'})

    # These are part of France / the UK / the Netherlands / New Zealand
    gir_data[['country', 'iso3']] = gir_data[['country', 'iso3']].replace(
        {
            'Réunion': 'France', 'REU': 'FRA',
            'Martinique': 'France', 'MTQ': 'FRA',
            'Mayotte': 'France', 'MYT': 'FRA',
            'Wallis and Futuna Islands': 'France', 'WLF': 'FRA',
            'Saint Barthélemy': 'France', 'BLM': 'FRA',
            'French Polynesia': 'France', 'PYF': 'FRA',
            'French Guiana': 'France', 'GUF': 'FRA',
            'Guadeloupe': 'France', 'GLP': 'FRA',
            'Saint Pierre and Miquelon': 'France', 'SPM': 'FRA',
            'Saint Helena': 'United Kingdom', 'SHN': 'GBR',
            'Montserrat': 'United Kingdom', 'MSR': 'GBR',
            'Jersey': 'United Kingdom', 'JEY': 'GBR',  # part of the Crown estate, not the UK
            'Guernsey': 'United Kingdom', 'GGY': 'GBR',  # part of the Crown estate, not the UK
            'Gibraltar': 'United Kingdom', 'GIB': 'GBR',
            'Pitcairn': 'United Kingdom', 'PCN': 'GBR',
            'Falkland Islands (Malvinas)': 'United Kingdom', 'FLK': 'GBR',
            'Bonaire, Sint Eustatius and Saba': 'Netherlands', 'BES': 'NLD',
            'Tokelau': 'New Zealand', 'TKL': 'NZL',
        }
    )

    # drop SSP scenarios
    gir_data = gir_data[gir_data.climate_scenario == climate_scenario].drop('climate_scenario', axis=1)

    # drop Loss exceedance curve
    gir_data = gir_data[gir_data.risk_metric_abbr.isin(['PML', 'AAL'])].drop('risk_metric', axis=1)

    # drop Landslides for now --> no overlap of subhazards between hazards
    gir_data = gir_data[gir_data.hazard != 'Landslide'].drop('hazard', axis=1).rename({'subhazard': 'hazard'}, axis=1)

    # set empty return (=AAL) periods to 0
    gir_data.rp = gir_data.rp.astype(object)
    gir_data.rp.fillna('AAL', inplace=True)

    gir_data[['cap_stock', 'gdp']] = gir_data[['cap_stock_capita', 'gdp_capita']].mul(gir_data['pop'], axis=0)
    gir_macro = gir_data[['iso3', 'country', 'pop', 'cap_stock', 'gdp']].drop_duplicates().groupby(['iso3', 'country']).sum().reset_index('country')
    gir_loss = gir_data.groupby(['iso3', 'hazard', 'rp']).loss.sum().to_frame()

    # compute the share of capital that is destroyed
    gir_loss['frac_value_destroyed'] = gir_loss.loss / gir_macro.cap_stock

    frac_value_destroyed_pml = gir_loss.frac_value_destroyed.drop('AAL', level='rp')
    frac_value_destroyed_aal = gir_loss.frac_value_destroyed.xs('AAL', level='rp')

    frac_value_destroyed_result = frac_value_destroyed_pml

    # drop zero values
    zero_values = frac_value_destroyed_result.groupby(['iso3', 'hazard']).apply(lambda x: (x == 0).all())
    frac_value_destroyed_result.drop(zero_values[zero_values].index, inplace=True)

    return frac_value_destroyed_result


if __name__ == '__main__':
    root_dir = os.getcwd()
    gir_filepath = os.path.join(root_dir, "data/raw/GIR_hazard_loss_data/export_all_metrics.csv.zip")
    default_rp = "default_rp"
    load_giri_hazard_loss_rel(
        gir_filepath_=gir_filepath,
    )
