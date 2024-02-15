import copy
import os
import warnings

import numpy as np
import pandas as pd
import pycountry as pc
from scipy.interpolate import interp1d

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
    # --> tax is the sum of all transfers paid over the sum of all income
    tx_tax = (cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum()
              / cat_info[["c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum())
    tx_tax.name = 'tau_tax'

    # income from social protection PER PERSON as fraction of PER CAPITA social protection
    # paper equation 5: \gamma_i = t_i / (\Sigma_i \mu \tau k_i)
    gsp = (cat_info[["diversified_share", "c"]].prod(axis=1, skipna=False)
           / cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum())
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
    any_to_wb = load_input_data(root_dir, "country_name_mappings/any_name_to_wb_name.csv", index_col="any")
    any_to_wb = any_to_wb[~any_to_wb.index.duplicated(keep='first')]  # drop duplicates

    any_to_wb = any_to_wb.squeeze()

    # iso3 to wb country name table
    iso3_to_wb = load_input_data(root_dir, "country_name_mappings/iso3_to_wb_name.csv").set_index("iso3").squeeze()
    # iso2 to iso3 table
    iso2_iso3 = load_input_data(root_dir, "country_name_mappings/names_to_iso.csv",
                                usecols=["iso2", "iso3"]).drop_duplicates().set_index("iso2").squeeze()

    # TODO: do we want to aggregate these regions into FRA / GBR or keep them individually?
    # iso3_to_wb = pd.concat((
    #     iso3_to_wb,
    #     pd.Series(index=['GUF', 'GLP', 'MTQ', 'MYT', 'REU'], data=iso3_to_wb['FRA']),
    #     pd.Series(index=['FLK', 'GIB', 'MSR'], data=iso3_to_wb['GBR'])
    # ))

    # rename PWT countries to WB names:
    any_to_wb.loc["Côte d'Ivoire"] = "Cote d'Ivoire"
    any_to_wb.loc["Côte d’Ivoire"] = "Cote d'Ivoire"
    any_to_wb.loc['D.R. of the Congo'] = 'Congo, Dem. Rep.'
    any_to_wb.loc['China, Hong Kong SAR'] = 'Hong Kong SAR, China'
    any_to_wb.loc["Lao People's DR"] = "Lao PDR"
    any_to_wb.loc['China, Macao SAR'] = 'Macao SAR, China'
    any_to_wb.loc["North Macedonia"] = "Macedonia, FYR"
    any_to_wb.loc["Eswatini"] = "Swaziland"
    any_to_wb.loc['U.R. of Tanzania: Mainland'] = 'Tanzania'
    any_to_wb.loc['Taiwan, China'] = 'Taiwan'
    any_to_wb.loc['Czechia'] = 'Czech Republic'
    any_to_wb.loc['Turkiye'] = 'Turkey'

    for _c in any_to_wb.index:
        _c_nospace = _c.replace(' ', '')
        _c_lower = str.lower(_c)
        _c_lower_nospace = str.lower(_c_nospace)
        for __c in [_c_nospace, _c_lower, _c_lower_nospace]:
            if __c != _c:
                any_to_wb.loc[__c] = any_to_wb.loc[_c]

    # add values to iso3_to_wb
    iso3_to_wb['KSV'] = 'Kosovo'
    iso3_to_wb['XKX'] = 'Kosovo'
    iso3_to_wb['ROM'] = 'Romania'
    iso3_to_wb['ZAR'] = 'Congo, Dem. Rep.'
    iso3_to_wb['WBG'] = 'West Bank and Gaza'

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
    if protection is not None:
        protection_index = pd.merge(probabilities, protection, left_index=True, right_index=True, how='left').protection.values >= probabilities.reset_index('rp').rp.values
        probabilities.loc[protection_index] = 0

    # average weighted by probability
    res = df.mul(probabilities, axis=0).reset_index('rp', drop=True)
    res = res.groupby(level=list(range(res.index.nlevels))).sum()
    if type(df) is pd.Series:
        res.name = df.name
    return res


def gather_capital_data(root_dir_, include_legacy_sids=False):
    # Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
    # pwt_data = load_input_data(root_dir, "pwt90.xlsx", sheet_name="Data")
    capital_data = load_input_data(root_dir_, "PWT_macro_economic_data/pwt1001.xlsx", sheet_name="Data")
    capital_data = capital_data.rename({'countrycode': 'iso3'}, axis=1)

    # !! NOTE: PWT variable for capital stock has been renamed from 'ck' to 'cn' in the 10.0.0 version
    capital_data = capital_data[['iso3', 'cgdpo', 'cn', 'year']].dropna()

    # retain only the most recent year
    capital_data = capital_data.groupby("iso3").apply(lambda x: x.loc[(x['year']) == np.nanmax(x['year']), :])
    capital_data = capital_data.reset_index(drop=True).set_index('iso3')

    # get capital data for SIDS from GAR
    if include_legacy_sids:
        sids_list = load_input_data(root_dir_, "gar_name_sids.csv")
        sids_list = df_to_iso3(sids_list, 'country')
        sids_list = sids_list[sids_list.isaSID == "SIDS"].dropna().reset_index().iso3
        sids_capital_gar = load_input_data(root_dir_, "GAR_capital.csv")[['country', 'GDP', 'K']]
        sids_capital_gar = df_to_iso3(sids_capital_gar, 'country').drop('country', axis=1)
        sids_capital_gar.dropna(inplace=True)
        sids_capital_gar = sids_capital_gar.set_index("iso3")
        sids_capital_gar = sids_capital_gar.loc[np.intersect1d(sids_list.values, sids_capital_gar.index.values), :]
        sids_capital_gar = sids_capital_gar.replace(0, np.nan).dropna()
        sids_capital_gar.rename({'K': 'cn', 'GDP': 'cgdpo'}, axis=1, inplace=True)

        # merge capital data from PWT and GAR (SIDS)
        # compute average productivity of capital
        capital_data = pd.merge(capital_data, sids_capital_gar, on='iso3', how='outer')
        capital_data['cgdpo'] = capital_data.cgdpo_x.fillna(capital_data.cgdpo_y)
        capital_data['cn'] = capital_data.cn_x.fillna(capital_data.cn_y)
        capital_data.drop(['cgdpo_x', 'cgdpo_y', 'cn_x', 'cn_y'], axis=1, inplace=True)

    capital_data.drop('year', axis=1, inplace=True)
    capital_data["avg_prod_k"] = capital_data.cgdpo / capital_data.cn
    capital_data = capital_data.dropna()

    return capital_data


def integrate_and_find_recovery_rate(v: float, consump_util: float, discount_rate: float, average_productivity: float,
                                     lambda_increment: float, years_to_recover: int) -> float:
    """Find recovery rate (lambda) given the value of `v` (household vulnerability).

    Args:
        v (float): Household vulnerability.
        consump_util (float): Consumption utility.
        discount_rate (float): Discount rate.
        average_productivity (float): Average productivity.
        lambda_increment (float): Lambda increment for the integration.
        years_to_recover (int): Number of years to recover.

    Returns:
        float: Recovery rate (lambda).
    """

    # No existing solution found, so we need to optimize
    tot_weeks = 52 * years_to_recover
    dt = years_to_recover / tot_weeks

    _lambda = 0
    last_dwdlambda = 0

    while True:
        dwdlambda = 0
        for _t in np.linspace(0, years_to_recover, tot_weeks):
            factor = average_productivity + _lambda
            part1 = (average_productivity - factor * v * np.e**(-_lambda * _t))**(-consump_util)
            part2 = _t * factor - 1
            part3 = np.e**(-_t * (discount_rate + _lambda))
            dwdlambda += part1 * part2 * part3 * dt

        if (last_dwdlambda < 0 and dwdlambda > 0) or (last_dwdlambda > 0 and dwdlambda < 0) or _lambda > 10:
            return _lambda

        last_dwdlambda = dwdlambda
        _lambda += lambda_increment


def df_to_iso3(df_, column_name_, any_to_wb_=None, verbose_=False):
    if 'iso3' in df_:
        raise Exception("iso3 column already exists")

    def get_iso3(name):
        # hard coded country names:
        hard_coded = {
            'congo, dem. rep.': 'COD',
            'democratic republic of the congo': 'COD',
            'congo, rep.': 'COG',
            'congo brazzaville': 'COG',
            'congo, republic of the': 'COG',
            'cape verde': 'CPV',
            'hong kong sar, china': 'HKG',
            'china, hong kong special administrative region': 'HKG',
            'hong kong, china': 'HKG',
            'macao sar, china': 'MAC',
            'china, macao special administrative region': 'MAC',
            'macau, china': 'MAC',
            'korea, rep.': 'KOR',
            'korea, south': 'KOR',
            "korea, dem. people's rep.": 'PRK',
            'korea, dem. rep.': 'PRK',
            'st. vincent and the grenadines': 'VCT',
            'st. vincent ': 'VCT',
            'swaziland': 'SWZ',
            'bolivia (plurinational state of)': 'BOL',
            'faeroe islands': 'FRO',
            'iran (islamic republic of)': 'IRN',
            'iran, islamic rep.': 'IRN',
            'iran': 'IRN',
            'micronesia (federated states of)': 'FSM',
            'micronesia, fed. sts': 'FSM',
            'micronesia, fed. sts.': 'FSM',
            'the former yugoslav republic of macedonia': 'MKD',
            'macedonia, fyr': 'MKD',
            'united states virgin islands': 'VIR',
            'virgin islands (u.s.)': 'VIR',
            'venezuela (bolivarian republic of)': 'VEN',
            'venezuela, rb': 'VEN',
            'netherlands antilles': 'ANT',
            'st. kitts and nevis': 'KNA',
            'st. lucia': 'LCA',
            'st. martin (french part)': 'MAF',
            'bahamas, the': 'BHS',
            'curacao': 'CUW',
            'egypt, arab rep.': 'EGY',
            'gambia, the': 'GMB',
            'lao pdr': 'LAO',
            'turkiye': 'TUR',
            'turkey (turkiye)': 'TUR',
            'türkiye': 'TUR',
            'west bank and gaza': 'PSE',
            'gaza strip': 'PSE',
            'west bank': 'PSE',
            'occupied palestinian territory': 'PSE',
            'yemen, rep.': 'YEM',
            'kosovo': 'XKX',
            'tanzania, united rep.': 'TZA',
            'bosnia herzegovina': 'BIH',
        }
        if str.lower(name) in hard_coded:
            return hard_coded[str.lower(name)]

        # try to find in pycountry
        if pc.countries.get(name=name) is not None:
            return pc.countries.get(name=name).alpha_3
        elif pc.countries.get(common_name=name) is not None:
            return pc.countries.get(common_name=name).alpha_3
        elif pc.countries.get(official_name=name) is not None:
            return pc.countries.get(official_name=name).alpha_3
        else:
            try:
                fuzzy_search_res = pc.countries.search_fuzzy(name)
            except LookupError:
                if any_to_wb_ is not None and name in any_to_wb_.index and any_to_wb_.loc[name] != name:
                    if verbose_:
                        print(f"Warning: {name} not found in pycountry, but found in any_to_wb. Retry with "
                              f"{any_to_wb_.loc[name]}")
                    return get_iso3(any_to_wb_.loc[name])
                else:
                    if verbose_:
                        print(f"Warning: {name} not found in pycountry, and fuzzy search failed")
                    return None
            if len(fuzzy_search_res) == 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found {fuzzy_search_res[0].name}")
                return fuzzy_search_res[0].alpha_3
            elif len(fuzzy_search_res) > 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found multiple matches: {fuzzy_search_res}")
                return None

    df = df_.copy()
    df['iso3'] = df[column_name_].apply(lambda x: get_iso3(x))
    if df.iso3.isna().any():
        print(f"Warning: ISO3 could not be found for {len(df[df.iso3.isna()].country.unique())} countries.")
    return df


def agg_to_economy_level(df, seriesname, economy):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T * df["n"]).T.groupby(level=economy).sum()


def interpolate_rps(hazard_ratios, protection_list, default_rp):
    """Extends return periods in hazard_ratios to a finer grid as defined in protection_list, by extrapolating to rp=0.
    hazard_ratios: dataframe with columns of return periods, index with countries and hazards.
    protection_list: list of protection levels that hazard_ratios should be extended to.
    default_rp: function does not do anything if default_rp is already present in hazard_ratios.
    """
    # check input
    if hazard_ratios is None:
        print("hazard_ratios is None, skipping...")
        return None
    if default_rp in hazard_ratios.index:
        print(f"default_rp={default_rp} already in hazard_ratios, skipping...")
        return hazard_ratios
    hazard_ratios_ = hazard_ratios.copy(deep=True)
    protection_list_ = copy.deepcopy(protection_list)

    flag_stack = False
    if "rp" in get_list_of_index_names(hazard_ratios_):
        original_index_names = hazard_ratios_.index.names
        hazard_ratios_ = hazard_ratios_.unstack("rp")
        flag_stack = True

    if type(protection_list_) is pd.DataFrame:
        protection_list_ = protection_list_.squeeze()
    if type(protection_list_) is pd.Series:
        protection_list_ = protection_list_.unique().tolist()

    # in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(hazard_ratios_.columns) is pd.MultiIndex:
        keys = hazard_ratios_.columns.get_level_values(0).unique()
        res = pd.concat(
            {col: interpolate_rps(hazard_ratios_[col], protection_list_, default_rp) for col in keys},
            axis=1
        )
    else:
        # actual function
        # figures out all the return periods to be included
        all_rps = list(set(protection_list_ + hazard_ratios_.columns.tolist()))

        res = hazard_ratios_.copy()

        # extrapolates linearly towards the 0 return period exposure (this creates negative exposure that is tackled after
        # interp.) (mind the 0 rp when computing probabilities)
        if len(res.columns) == 1:
            res[0] = res.squeeze()
        else:
            res[0] = (
                # exposure of smallest return period
                res.iloc[:, 0] -
                # smallest return period * (exposure of second-smallest rp - exposure of smallest rp) /
                res.columns[0] * (res.iloc[:, 1] - res.iloc[:, 0]) /
                # (second-smallest rp - smallest rp)
                (res.columns[1] - res.columns[0])
            )

        # add new, interpolated values for fa_ratios, assuming constant exposure on the right
        x = res.columns.values
        y = res.values
        res = pd.DataFrame(
            data=interp1d(x, y, bounds_error=False)(all_rps),
            index=res.index,
            columns=all_rps
        ).sort_index(axis=1).clip(lower=0).ffill(axis=1)
        res.columns.name = "rp"

    if flag_stack:
        res = res.stack("rp").reset_index().set_index(original_index_names).sort_index()
    return res


def recompute_after_policy_change(macro_, cat_info_, hazard_ratios_, econ_scope_, axfin_impact_, pi_):
    macro_ = macro_.copy(deep=True)
    cat_info_ = cat_info_.copy(deep=True)
    hazard_ratios_ = hazard_ratios_.copy(deep=True)

    # TODO: check whether any of the recomputation is still necessary
    # here we assume that gdp = consumption = prod_from_k
    macro_["gdp_pc_pp"] = macro_["avg_prod_k"] * agg_to_economy_level(cat_info_, "k", econ_scope_)
    cat_info_["c"] = ((1 - macro_["tau_tax"]) * macro_["avg_prod_k"] * cat_info_["k"] +
                      cat_info_["gamma_SP"] * macro_["tau_tax"] * macro_["avg_prod_k"]
                      * agg_to_economy_level(cat_info_, "k", econ_scope_))

    # recompute diversified_share after policy change
    cat_info_['diversified_share'] = cat_info_.social + cat_info_.axfin * axfin_impact_

    macro_["tau_tax"], cat_info_["gamma_SP"] = social_to_tx_and_gsp(econ_scope_, cat_info_)

    # Recompute consumption from k and new gamma_SP and tau_tax
    cat_info_["c"] = ((1 - macro_["tau_tax"]) * macro_["avg_prod_k"] * cat_info_["k"] +
                      cat_info_["gamma_SP"] * macro_["tau_tax"] * macro_["avg_prod_k"]
                      * agg_to_economy_level(cat_info_, "k", econ_scope_))

    # Calculation of macroeconomic resilience (Gamma in the technical paper)
    # \Gamma = (\mu + 3/N) / (\rho + 3/N)
    hazard_ratios_["macro_multiplier_Gamma"] = ((macro_["avg_prod_k"] + hazard_ratios_["recovery_rate"]) /
                                           (macro_["rho"] + hazard_ratios_["recovery_rate"]))

    hazard_ratios_["v_ew"] = hazard_ratios_["v"] * (1 - pi_ * hazard_ratios_["ew"])
    hazard_ratios_.drop(['ew', 'v'], inplace=True, axis=1)

    # TODO: no longer interpolate return periods, but consider any rp below the maximum protection level as protected
    # interpolates data to a more granular grid for return periods that includes all protection values that are
    # potentially not the same in hazard_ratios.
    # hazard_ratios_ = interpolate_rps(hazard_ratios_, protection_, default_rp=default_rp_)

    return macro_, cat_info_, hazard_ratios_
