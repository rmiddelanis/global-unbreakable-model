import os
import warnings
import pandas as pd
from pandas_helper import broadcast_simple
import numpy as np

from lib_compute_resilience_and_risk import average_over_rp


def replace_with_warning(series_in, dico, ignore_case=True, joiner=", "):
    # preprocessing
    series_to_use = series_in.copy()
    dico_to_use = dico.copy()

    if ignore_case:
        series_to_use = series_to_use.str.lower()
        dico_to_use.index = dico_to_use.index.str.lower()

    # processing
    out = series_to_use.replace(dico_to_use)

    # post processing
    are_missing = ~out.isin(dico_to_use)

    if are_missing.sum() > 0:
        warnings.warn(
            "These entries were not found in the dictionary:\n" + joiner.join(series_in[are_missing].unique()))

    return out


###########################
# Load fa for wind or surge
# def get_dk_over_k_from_file(path='inputs/GAR_data_surge.csv'):
def get_dk_over_k_from_file(path, iso3_to_wb):
    gar_data = pd.read_csv(path).dropna(axis=1, how='all')

    # these are part of France & the UK
    gar_data.Country.replace(['GUF', 'GLP', 'MTQ', 'MYT', 'REU'], 'FRA', inplace=True)
    gar_data.Country.replace(['FLK', 'GIB', 'MSR'], 'GBR', inplace=True)

    gar_data = gar_data.set_index(replace_with_warning(gar_data.Country, iso3_to_wb))[
        ['AAL', 'Exposed Value (from GED)']]

    return gar_data['AAL'].groupby(level=0).sum() / gar_data['Exposed Value (from GED)'].groupby(level=0).sum()


#######################
# RP (as str) to floats
def str_to_float(s):
    try:
        return (float(s))
    except ValueError:
        return s


def gar_preprocessing(input_dir, intermediate_dir, default_rp):
    # global iso3_to_wb

    iso3_to_wb = pd.read_csv(os.path.join(input_dir, 'iso3_to_wb_name.csv'), index_col='iso3').squeeze()

    # Names to WB names
    any_to_wb = pd.read_csv(os.path.join(input_dir, 'any_name_to_wb_name.csv'), index_col='any').squeeze()

    #######
    # AAL
    #
    # agg data
    gar_aal_data = pd.read_csv(os.path.join(input_dir, 'GAR15 results feb 2016_AALmundo.csv'))

    # These are part of France and the UKxs
    gar_aal_data.ISO.replace(['GUF', 'GLP', 'MTQ', 'MYT', 'REU'], 'FRA', inplace=True)
    gar_aal_data.ISO.replace(['FLK', 'GIB', 'MSR'], 'GBR', inplace=True)

    # WB spellings
    gar_aal_data = gar_aal_data.set_index(replace_with_warning(gar_aal_data.ISO.astype(str), iso3_to_wb)).drop(
        ['ISO', 'Country'], axis=1)
    gar_aal_data.index.name = 'country'

    # aggregates UK and France pieces to one country
    gar_aal_data = gar_aal_data.groupby(level='country').sum()

    # takes exposed value out
    gar_exposed_value = gar_aal_data.pop('EXPOSED VALUE')

    # rename hazards
    gar_aal_data = gar_aal_data.rename(columns=lambda s: (s.lower()))
    gar_aal_data = gar_aal_data.rename(
        columns={'tropical cyclones': 'cyclones', 'riverine floods': 'flood', 'storm surge': 'surge'})

    # gar_aal_data
    AAL = (gar_aal_data.T / gar_exposed_value).T

    # wind and surge
    aal_surge = get_dk_over_k_from_file(os.path.join(input_dir, 'GAR_data_surge.csv'), iso3_to_wb)
    aal_wind = get_dk_over_k_from_file(os.path.join(input_dir, 'GAR_data_wind.csv'), iso3_to_wb)
    aal_recomputed = aal_surge + aal_wind

    c = 'Costa Rica'
    f = (aal_surge / aal_recomputed).mean()
    aal_surge.loc[c] = f * AAL.cyclones[c]
    aal_wind.loc[c] = (1 - f) * AAL.cyclones[c]

    # AAL SPLIT
    AAL_splitted = AAL.drop('cyclones', axis=1).assign(wind=aal_wind, surge=aal_surge).rename(
        columns=dict(floods='flood')).stack()
    AAL_splitted.index.names = ['country', 'hazard']

    # PMLs and Exposed value
    gardata = pd.read_csv(os.path.join(input_dir, 'GAR15 results feb 2016_PML mundo.csv'), encoding='latin-1', header=[0, 1, 2],
                          index_col=0)
    gardata.index.name = 'country'

    gardata = gardata.reset_index()
    gardata.columns.names = ['hazard', 'rp', 'what']

    # These are part of france / the UK
    gardata.country = gardata.country.replace(["French Guiana", "Guadeloupe", "Martinique", "Mayotte", "Reunion"],
                                              "France")
    gardata.country = gardata.country.replace(["Falkland Islands (Malvinas)", "Gibraltar", "Montserrat"],
                                              "United Kingdom")
    gardata.country = replace_with_warning(gardata.country, any_to_wb.dropna())
    gardata = gardata.set_index("country")

    # Other format for zero
    gardata = gardata.replace("---", 0).astype(float)

    # aggregates france and uk
    gardata = gardata.groupby(level="country").sum()

    # Looks at the exposed value, for comparison
    exposed_value_GAR = gardata.pop("EXPOSED VALUE").squeeze()

    # rename hazards
    gardata = gardata.rename(columns=lambda s: (s.lower()))
    gardata = gardata.rename(columns={"wind": "wind", "riverine floods": "flood", "storm surge": "surge"})

    gardata = gardata.rename(columns=str_to_float)

    # Asset losses per event
    dK_gar = gardata.swaplevel("what", "hazard", axis=1)["million us$"].swaplevel("rp", "hazard", axis=1).stack(
        level=["hazard", "rp"], dropna=False).sort_index()

    # Exposed value per event
    ev_gar = broadcast_simple(exposed_value_GAR, dK_gar.index)

    # Fraction of value destroyed
    frac_value_destroyed_gar = dK_gar / ev_gar

    # save fraction of destroyed value to disk
    # contains fraction of value destroyed for each country, hazard, and return period
    frac_value_destroyed_gar.to_csv(os.path.join(intermediate_dir, "frac_value_destroyed_gar.csv"), encoding="utf-8",
                                    header=True)

    # add frequent events
    last_rp = 20
    new_rp = 1

    added_proba = 1 / new_rp - 1 / last_rp

    # average_over_rp() simply averages return period losses, weighted with the probability of each return period, i.e.
    # the inverse of the return period
    # TODO: why divide by added_proba?
    new_frac_destroyed = (AAL_splitted - average_over_rp(frac_value_destroyed_gar, default_rp).squeeze()) / added_proba

    hop = frac_value_destroyed_gar.unstack()
    hop[new_rp] = new_frac_destroyed
    hop = hop.sort_index(axis=1)

    frac_value_destroyed_gar_completed = hop.stack()
    # print(frac_value_destroyed_gar_completed.head(50))
    # ^ this shows earthquake was not updated

    # double check. expecting zeroes expect for quakes and tsunamis:
    (average_over_rp(frac_value_destroyed_gar_completed, default_rp).squeeze() - AAL_splitted).abs().sort_values(
        ascending=False).sample(10)

    # places where new values are higher than values for 20-yr RP
    test = frac_value_destroyed_gar_completed.unstack().replace(0, np.nan).dropna().assign(
        test=lambda x: x[new_rp] / x[20]).test

    max_relative_exp = .8

    overflow_frequent_countries = test[test > max_relative_exp].index
    print("overflow in {n} (country, event)".format(n=len(overflow_frequent_countries)))

    # for this places, add infrequent events
    hop = frac_value_destroyed_gar_completed.unstack()

    hop[1] = hop[1].clip(upper=max_relative_exp * hop[20])
    frac_value_destroyed_gar_completed = hop.stack()

    new_rp = 2000
    added_proba = 1 / 2000

    new_frac_destroyed = (AAL_splitted - average_over_rp(frac_value_destroyed_gar_completed, default_rp).squeeze()) / added_proba

    hop = frac_value_destroyed_gar_completed.unstack()
    hop[new_rp] = new_frac_destroyed.clip(upper=0.99)
    hop = hop.sort_index(axis=1)

    frac_value_destroyed_gar_completed = hop.stack()

    print('GAR preprocessing script: writing out intermediate/frac_value_destroyed_gar_completed.csv')
    frac_value_destroyed_gar_completed.to_csv(os.path.join(intermediate_dir, "frac_value_destroyed_gar_completed.csv"),
                                              encoding="utf-8", header=True)
