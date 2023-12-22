import os
import warnings

import numpy as np
import pandas as pd

from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories, load_input_data


def mystriper(string):
    """strip blanks and converts everythng to lower case"""
    if type(string) == str:
        return str.strip(string).lower()
    else:
        return string


# weighted average
def wavg(data, weights):
    df_matched = pd.DataFrame({"data": data, "weights": weights}).dropna()
    return (df_matched.data * df_matched.weights).sum() / df_matched.weights.sum()


# gets share per agg category from the data in one of the sheets in PAGER_XL
def get_share_from_sheet(pager_df, pager_code_to_aggcat, iso3_to_wb):
    data = pager_df.set_index("ISO-3digit")  # data as provided in PAGER
    # rename column to aggregate category
    data_agg = data[pager_code_to_aggcat.index].rename(
        columns=pager_code_to_aggcat)  # only pick up the columns that are the indice in paper_code_to_aggcat, and change each name to median, fragile etc. based on pager_code_to_aggcat

    # group by category and sum
    # data_agg = data_agg.groupby(level=0).sum(axis=1)  # sum each category up and shows only three columns with fragile, median and robust.
    data_agg = data_agg.T.groupby(level=0).sum().T

    data_agg = data_agg.set_index(data_agg.reset_index()["ISO-3digit"].replace(iso3_to_wb));

    data_agg.index.name = "country"
    return data_agg[data_agg.index.isin(iso3_to_wb)]  # keeps only countries


def social_to_tx_and_gsp(economy, cat_info):
    """(tx_tax, gamma_SP) from cat_info[["social","c","n"]] """

    # paper equation 4: \tau = (\Sigma_i t_i) / (\Sigma_i \mu k_i)
    # --> tax is the sum of all transfers paid over the sum of all income (excluding transfers ?!)
    # TODO: doesn't the calculation below include transfers in income?!
    tx_tax = cat_info[["social", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum() / \
             cat_info[["c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum()
    tx_tax.name = 'tau_tax'

    # paper equation 5: \gamma_i = t_i / (\Sigma_i \mu \tau k_i)
    gsp = cat_info[["social", "c"]].prod(axis=1, skipna=False) / \
          cat_info[["social", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum()
    gsp.name = 'gamma_SP'

    return tx_tax, gsp


###########################
# Load fa for wind or surge
# def get_dk_over_k_from_file(path='inputs/GAR_data_surge.csv'):
def get_dk_over_k_from_file(root_dir, input_data_file, iso3_to_wb):
    gar_data = load_input_data(root_dir, input_data_file).dropna(axis=1, how='all')

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


def gar_preprocessing(root_dir, intermediate_dir, default_rp):
    # global iso3_to_wb

    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    #######
    # AAL
    #
    # agg data
    gar_aal_data = load_input_data(root_dir, 'GAR15 results feb 2016_AALmundo.csv')


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
    aal_surge = get_dk_over_k_from_file(root_dir, 'GAR_data_surge.csv', iso3_to_wb)
    aal_wind = get_dk_over_k_from_file(root_dir, 'GAR_data_wind.csv', iso3_to_wb)
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
    gardata = load_input_data(root_dir, 'GAR15 results feb 2016_PML mundo.csv', encoding='latin-1',
                              header=[0, 1, 2], index_col=0)
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


def get_country_name_dicts(root_dir):
    # Country dictionaries
    any_to_wb = load_input_data(root_dir, "any_name_to_wb_name.csv", index_col="any")  # Names to WB names
    any_to_wb = any_to_wb[~any_to_wb.index.duplicated(keep='first')]  # drop duplicates

    # TODO: keep this? why?
    for _c in any_to_wb.index:
        __c = _c.replace(' ', '')
        if __c != _c:
            any_to_wb.loc[__c] = any_to_wb.loc[_c, 'wb_name']

    any_to_wb = any_to_wb.squeeze()

    # iso3 to wb country name table
    iso3_to_wb = load_input_data(root_dir, "iso3_to_wb_name.csv").set_index("iso3").squeeze()
    # iso2 to iso3 table
    iso2_iso3 = load_input_data(root_dir, "names_to_iso.csv",
                                usecols=["iso2", "iso3"]).drop_duplicates().set_index("iso2").squeeze()

    # TODO: do we want to aggregate these regions into FRA / GBR or keep them individually?
    # iso3_to_wb = pd.concat((
    #     iso3_to_wb,
    #     pd.Series(index=['GUF', 'GLP', 'MTQ', 'MYT', 'REU'], data=iso3_to_wb['FRA']),
    #     pd.Series(index=['FLK', 'GIB', 'MSR'], data=iso3_to_wb['GBR'])
    # ))

    # rename PWT countries to WB names:
    any_to_wb.loc["Côte d'Ivoire"] = "Cote d'Ivoire"
    any_to_wb.loc['D.R. of the Congo'] = 'Congo, Dem. Rep.'
    any_to_wb.loc['China, Hong Kong SAR'] = 'Hong Kong SAR, China'
    any_to_wb.loc["Lao People's DR"] = "Lao PDR"
    any_to_wb.loc['China, Macao SAR'] = 'Macao SAR, China'
    any_to_wb.loc["North Macedonia"] = "Macedonia, FYR"
    any_to_wb.loc["Eswatini"] = "Swaziland"
    any_to_wb.loc['U.R. of Tanzania: Mainland'] = 'Tanzania'
    return any_to_wb, iso3_to_wb, iso2_iso3


# TODO: fix protection. In it's current form, df and protection are required to have the same index. Protection needs
#  to be a Series object with boolean values indicating protection or not.
def average_over_rp(df, default_rp, protection=None):
    """Aggregation of the outputs over return periods"""

    # just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values("rp"):
        print("default_rp detected, dropping rp")
        return (df.T / protection).T.reset_index("rp", drop=True)

    # compute probability of each return period
    return_periods = df.index.get_level_values('rp').unique()
    rp_probabilities = pd.Series(1 / return_periods - np.append(1 / return_periods, 0)[1:], index=return_periods)

    # match return periods and their frequency
    probabilities = pd.Series(data=df.reset_index('rp').rp.replace(rp_probabilities).values, index=df.index,
                              name='probability')

    # removes events below the protection level
    if protection:
        raise NotImplementedError("Warning. Need to fix protection before using.")
        probabilities[protection] = 0

    # average weighted by probability
    res = df.mul(probabilities, axis=0).reset_index('rp', drop=True)
    res = res.groupby(level=list(range(res.index.nlevels))).sum()
    if type(df) is pd.Series:
        res.name = df.name
    return res


def gather_capital_data(root_dir):
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
    # pwt_data = load_input_data(root_dir, "pwt90.xlsx", sheet_name="Data")
    pwt_data = load_input_data(root_dir, "PWT_macro_economic_data/pwt1001.xlsx", sheet_name="Data")
    # retain only the most recent year
    pwt_data = pwt_data.groupby("country").apply(lambda x: x.loc[(x['year']) == np.max(x['year']), :])
    pwt_data = pwt_data.drop("country", axis=1).reset_index().drop("level_1", axis=1)

    # !! NOTE: PWT variable for capital stock has been renamed from 'ck' to 'cn' in the 10.0.0 version
    # pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'ck']]
    pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'cn']].rename({'cn': 'ck'}, axis=1)
    pwt_data["country"] = pwt_data.country.replace(any_to_wb)
    pwt_data = pwt_data.dropna()
    pwt_data.set_index("country", inplace=True)
    pwt_data.drop(["countrycode", 'year'], axis=1, inplace=True)

    # get capital data for SIDS from GAR
    sids_list = load_input_data(root_dir, "gar_name_sids.csv")
    sids_list['wbcountry'] = sids_list.reset_index().country.replace(any_to_wb)
    sids_list = sids_list[sids_list.isaSID == "SIDS"].dropna().reset_index().wbcountry
    sids_capital_gar = load_input_data(root_dir, "GAR_capital.csv")[['country', 'GDP', 'K']]
    sids_capital_gar["country"] = sids_capital_gar.country.replace(any_to_wb)
    sids_capital_gar.dropna(inplace=True)
    sids_capital_gar = sids_capital_gar.set_index("country")
    sids_capital_gar = sids_capital_gar.loc[np.intersect1d(sids_list.values, sids_capital_gar.index.values), :]
    sids_capital_gar = sids_capital_gar.replace(0, np.nan).dropna()
    sids_capital_gar.rename({'K': 'ck', 'GDP': 'cgdpo'}, axis=1, inplace=True)

    # merge capital data from PWT and GAR (SIDS)
    # compute average productivity of capital
    capital_data = pd.merge(pwt_data, sids_capital_gar, on='country', how='outer')
    capital_data['cgdpo'] = capital_data.cgdpo_x.fillna(capital_data.cgdpo_y)
    capital_data['ck'] = capital_data.ck_x.fillna(capital_data.ck_y)
    capital_data.drop(['cgdpo_x', 'cgdpo_y', 'ck_x', 'ck_y'], axis=1, inplace=True)
    capital_data["avg_prod_k"] = capital_data.cgdpo / capital_data.ck
    capital_data = capital_data.dropna()

    return capital_data
