import numpy as np
import pandas as pd

from lib_gather_data import average_over_rp
from pandas_helper import load_input_data


def load_gir_hazard_losses(root_dir, gir_filepath, default_rp):
    """
    Load GIR hazard loss data, process the data, and return the fraction of value destroyed for each
    country, hazard, and return period. GIR data contains data for hazards Tropical cyclone, Tsunami, Flood (riverine),
    Landslide, Earthquake. In addition, subhazards are provided for all hazards; Tropical cyclone: (Wind, Storm surge),
    Tsunami: (Tsunami), Flood: (Flood), Landslide: (Rain, Earthquake), Earthquake: (Earthquake).

    Parameters:
    root_dir (str): The root directory of the project repository.
    gir_filepath (str): The relative path (from input_dir) to the data file.
    default_rp (str): The default return period to use when no return period is provided in the data.

    Returns:
    pandas.Series: A pandas Series with a MultiIndex of ['iso3', 'hazard', 'rp'] and values representing the
    fraction of value destroyed for each country, hazard, and return period.
    """
    gir_data = load_input_data(root_dir, gir_filepath, version='new')
    gir_data.rename({'value_axis_2': 'rp', 'value_axis_1': 'loss', 'iso3cd': 'iso3',
                     'country_name': 'country'}, axis=1,
                    inplace=True)
    gir_data.drop('unit_axis_1', axis=1, inplace=True)

    # drop disputed territories without distinct iso3 code
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
    # TODO: West Sahara (ESH)

    # drop SSP scenarios
    gir_data = gir_data[gir_data.climate_scenario == 'Existing climate']

    # drop Loss exceedance curve
    gir_data = gir_data[gir_data.risk_metric_abbr.isin(['PML', 'AAL'])]

    # TODO: decide whether to use subhazard or hazard
    # TODO: Landslides
    # drop Landslides for now --> no overlap of subhazards between hazards
    gir_data = gir_data[gir_data.hazard != 'Landslide']

    # set empty return (=AAL) periods to 0
    gir_data.rp.fillna(0, inplace=True)
    gir_data = gir_data[['iso3', 'country', 'subhazard', 'rp', 'loss', 'pop', 'cap_stock_capita', 'gdp_capita']].groupby(
        ['iso3', 'country', 'subhazard', 'rp']#, 'sector_type', ]
    ).agg(
        {'loss': 'sum', 'pop': 'mean', 'cap_stock_capita': 'mean', 'gdp_capita': 'mean'}
    ).reset_index().rename({'subhazard': 'hazard'}, axis=1)

    # TODO: check that (gir_data.cap_stock_capita * gir_data.pop) = exposed value is valid
    gir_data['frac_value_destroyed'] = gir_data.loss / (gir_data.cap_stock_capita * gir_data['pop'])
    assert np.all(gir_data.frac_value_destroyed >= 0)
    frac_value_destroyed = gir_data.set_index(['iso3', 'hazard', 'rp']).frac_value_destroyed.reset_index()
    frac_value_destroyed_aal = frac_value_destroyed[frac_value_destroyed.rp == 0].drop('rp', axis=1)
    frac_value_destroyed_aal = frac_value_destroyed_aal.set_index(['iso3', 'hazard']).frac_value_destroyed
    frac_value_destroyed_pml = frac_value_destroyed[frac_value_destroyed.rp != 0]
    frac_value_destroyed_pml = frac_value_destroyed_pml.set_index(['iso3', 'hazard', 'rp']).frac_value_destroyed

    loss_incoherences = frac_value_destroyed_aal - average_over_rp(frac_value_destroyed_pml, default_rp) < 0
    if loss_incoherences.any():
        print(f"Warning: AAL is smaller than the average loss over all return periods for the following "
              f"(iso3, hazard) tuples:\n\n{loss_incoherences[loss_incoherences].index.values}.\n\nThis will result "
              f"in negative losses for additional return periods.")

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
        new_data = (aal_data_ - average_over_rp(pml_data_, default_rp).squeeze()) / new_probability
        negative_results = new_data < 0
        if np.any(negative_results):
            print(f"Setting negative losses for return period {new_rp_} to 0 for {len(new_data[negative_results])} "
                  f"countries.")
            new_data[new_data < 0] = 0
        new_data = new_data.reset_index()
        new_data['rp'] = new_rp_
        new_data = new_data.set_index(['iso3', 'hazard', 'rp']).frac_value_destroyed
        res = pd.concat([pml_data_, new_data]).sort_index()
        # check that the new data is consistent with the overall AAL. Values should be 0 (tolerance 1e-10)
        max_deviation = (average_over_rp(res, default_rp).squeeze() - aal_data_).abs().max()
        if max_deviation > 1e-10:
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
    if len(overflow_countries) > 0:
        print("overflow in {n} (iso3, event) tuples.".format(n=len(overflow_countries)))
        # clip the new return period loss s.th. it is not higher than the loss for the previously smallest rp times
        # the overflow_factor
        frac_value_destroyed_completed.loc[:, :, new_min_rp] = frac_value_destroyed_completed.loc[:, :, new_min_rp].clip(
            upper=frac_value_destroyed_completed.loc[:, :, min_pml_rp] * overflow_factor
        )
        new_max_rp = 7500
        # add the remaining loss to the new maximum return period
        # overflow_country_indexer = [i[0] + (i[1], ) for i in pd.MultiIndex.from_product(
        #     (overflow_countries, frac_value_destroyed_complete.index.get_level_values(2).unique())
        # ) if i[0] + (i[1], ) in frac_value_destroyed_complete.index]
        # data_update = add_rp(frac_value_destroyed_aal.loc[overflow_country_indexer],
        #                      frac_value_destroyed_complete[overflow_country_indexer], new_max_rp)  # TODO this is wrong, duplicate entries after indexing
        frac_value_destroyed_completed = add_rp(frac_value_destroyed_aal, frac_value_destroyed_completed, new_max_rp)
    return frac_value_destroyed_completed