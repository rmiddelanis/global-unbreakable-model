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

from src.misc.helpers import average_over_rp


def load_giri_hazard_loss_rel(gir_filepath_, extrapolate_rp_=True, climate_scenario='Existing climate', verbose=True):
    """
    Loads and processes GIRI (Global Infrastructure Risk Model and Resilience Index) hazard loss data.

    This function processes hazard loss data to compute the fraction of value destroyed for each country, hazard,
    and return period. It supports extrapolation of return periods and filtering by climate scenarios.

    Args:
        gir_filepath_ (str): Path to the GIR hazard loss data file (compressed CSV).
        extrapolate_rp_ (bool): Whether to extrapolate return periods. Defaults to True.
        climate_scenario (str): Climate scenario to filter the data. Defaults to 'Existing climate'.
        verbose (bool): Whether to print warnings and additional information. Defaults to True.

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

    # check for incoherences
    loss_incoherences = frac_value_destroyed_aal - average_over_rp(frac_value_destroyed_pml) < 0
    if loss_incoherences.any() and verbose:
        print(f"Warning: AAL is smaller than the average loss over all return periods for the following "
              f"(iso3, hazard) tuples:\n\n{loss_incoherences[loss_incoherences].index.values}.\n\nThis will result "
              f"in negative losses for additional return periods.")

    if not extrapolate_rp_:
        frac_value_destroyed_result = frac_value_destroyed_pml
    else:
        def add_rp(aal_data_, pml_data_, new_rp_):
            smallest_rp = min(pml_data_.index.get_level_values('rp'))
            largest_rp = max(pml_data_.index.get_level_values('rp'))
            if new_rp_ < smallest_rp:
                new_probability = 1 / new_rp_ - 1 / smallest_rp
            elif new_rp_ > largest_rp:
                new_probability = 1 / new_rp_
            else:
                raise ValueError("new_rp should be smaller than the smallest or larger than the largest return period in "
                                 "pml_data")
            # average_over_rp() averages return period losses, weighted with the probability of each return period, i.e.
            # the inverse of the return period
            new_data = (aal_data_ - average_over_rp(pml_data_).squeeze()) / new_probability
            negative_results = new_data < 0
            if np.any(negative_results) and verbose:
                print(f"Setting negative losses for return period {new_rp_} to 0 for {len(new_data[negative_results])} "
                      f"countries.")
                new_data[new_data < 0] = 0
            new_data = new_data.reset_index()
            new_data['rp'] = new_rp_
            new_data = new_data.set_index(['iso3', 'hazard', 'rp']).frac_value_destroyed
            res = pd.concat([pml_data_, new_data]).sort_index()
            # check that the new data is consistent with the overall AAL. Values should be 0 (tolerance 1e-10)
            max_deviation = (average_over_rp(res).squeeze() - aal_data_).abs().max()
            if max_deviation > 1e-10 and verbose:
                print(f"Warning: new data for return period {new_rp_} is not consistent with the overall AAL. The "
                      f"difference of the AAL to the return period average is up to {max_deviation}.")
            return res

        # add frequent events
        min_pml_rp = min(frac_value_destroyed_pml.index.get_level_values('rp'))
        new_min_rp = 1.0
        frac_value_destroyed_completed = add_rp(frac_value_destroyed_aal, frac_value_destroyed_pml, new_min_rp)

        # find countries where the loss for the new return period is higher than the loss for the previously smallest rp
        overflow_factor = 0.8
        overflow_countries = (frac_value_destroyed_completed.loc[:, :, new_min_rp]
                              >= overflow_factor * frac_value_destroyed_completed.loc[:, :, min_pml_rp])
        overflow_countries = overflow_countries[overflow_countries].index.values

        # add infrequent events for overflow countries
        if len(overflow_countries) > 0 and verbose:
            print("overflow in {n} (iso3, event) tuples.".format(n=len(overflow_countries)))
            # clip the new return period loss s.th. it is not higher than the loss for the previously smallest rp times
            # the overflow_factor
            frac_value_destroyed_completed.loc[:, :, new_min_rp] = frac_value_destroyed_completed.loc[:, :, new_min_rp].clip(
                upper=frac_value_destroyed_completed.loc[:, :, min_pml_rp] * overflow_factor
            )
            new_max_rp = 7500
            # add the remaining loss to the new maximum return period
            frac_value_destroyed_completed = add_rp(frac_value_destroyed_aal, frac_value_destroyed_completed, new_max_rp)
        frac_value_destroyed_result = frac_value_destroyed_completed

    # drop zero values
    zero_values = frac_value_destroyed_result.groupby(['iso3', 'hazard']).apply(lambda x: (x == 0).all())
    frac_value_destroyed_result.drop(zero_values[zero_values].index, inplace=True)

    return frac_value_destroyed_result


if __name__ == '__main__':
    root_dir = os.getcwd()
    gir_filepath = os.path.join(root_dir, "data/raw/GIR_hazard_loss_data/export_all_metrics.csv.zip")
    default_rp = "default_rp"
    extrapolate_rp = False
    load_giri_hazard_loss_rel(
        gir_filepath_=gir_filepath,
        extrapolate_rp_=extrapolate_rp,
    )
