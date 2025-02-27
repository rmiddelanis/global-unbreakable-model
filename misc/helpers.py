import os
from functools import partial

import numpy as np
import pandas as pd
import pycountry as pc


# from sorted_nicely import sorted_nicely

 
def get_list_of_index_names(df):
    """returns name of index in a data frame as a list. (single element list if the dataframe as a single index)""
    """
    
    if df.index.name is None:
        return list(df.index.names)
    else:
        return [df.index.name] #do not use list( ) as list breaks strings into list of chars

 
def concat_categories(p,np, index):
    """works like pd.concat with keys but swaps the index so that the new index is innermost instead of outermost
    http://pandas.pydata.org/pandas-docs/stable/merging.html#concatenating-objects
    """
    
    if index.name is None:
        raise Exception("index should be named")
        
    
    y= pd.concat([p, np], 
        keys = index, 
        names=[index.name]+get_list_of_index_names(p)
            )#.sort_index()
    
    #puts new index at the end            
    y=y.reset_index(index.name).set_index(index.name, append=True).sort_index()
    
    #makes sure a series is returned when possible
    return y.squeeze()


def average_over_rp(df, protection=None):
    """Aggregation of the outputs over return periods"""

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


def get_country_name_dicts(root_dir):
    # Country dictionaries
    any_to_wb = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/any_name_to_wb_name.csv"), index_col="any").squeeze()

    # iso3 to wb country name table
    iso3_to_wb = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/iso3_to_wb_name.csv"), index_col="iso3").squeeze()

    iso2_iso3 = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/iso2_to_iso3.csv"), index_col="iso2").squeeze()

    return any_to_wb, iso3_to_wb, iso2_iso3


def df_to_iso3(df_, column_name_, any_to_wb_=None, verbose_=True):
    if 'iso3' in df_:
        raise Exception("iso3 column already exists")

    def get_iso3(name):
        # hard coded country names:
        hard_coded = {
            'congo, dem. rep.': 'COD',
            'congo, democratic republic': 'COD',
            'democratic republic of the congo': 'COD',
            'congo, rep.': 'COG',
            'congo brazzaville': 'COG',
            'congo, republic of the': 'COG',
            'cape verde': 'CPV',
            'côte d’ivoire': 'CIV',
            'côte d\'ivoire': 'CIV',
            'hong kong sar, china': 'HKG',
            'china, hong kong special administrative region': 'HKG',
            'hong kong, china': 'HKG',
            'macao sar, china': 'MAC',
            'china, macao special administrative region': 'MAC',
            'macau, china': 'MAC',
            'macao, china': 'MAC',
            'taiwan, china': 'TWN',
            'korea, rep.': 'KOR',
            'republic of korea': 'KOR',
            'korea, south': 'KOR',
            "korea, dem. people's rep.": 'PRK',
            'korea, dem. rep.': 'PRK',
            'korea (the democratic people\'s republic of)': 'PRK',
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
    if df.iso3.isna().any() and verbose_:
        print(f"Warning: ISO3 could not be found for {len(df[df.iso3.isna()].country.unique())} countries.")
    return df


def load_income_groups(root_dir_):
    income_groups_file = "WB_country_classification/country_classification.xlsx"
    income_groups = pd.read_excel(os.path.join(root_dir_, "data/raw/", income_groups_file), header=0)[["Code", "Region", "Income group"]]
    income_groups = income_groups.dropna().rename({'Code': 'iso3'}, axis=1)
    income_groups = income_groups.set_index('iso3').squeeze()
    income_groups.loc['VEN'] = ['Latin America & Caribbean', 'Upper middle income']
    income_groups.rename({'Income group': 'Country income group'}, axis=1, inplace=True)
    income_groups.replace({
        'Low income': 'LICs',
        'Lower middle income': 'LMICs',
        'Upper middle income': 'UMICs',
        'High income': 'HICs'
    }, inplace=True)
    income_groups.replace({
        'South Asia': 'SAR',
        'Europe & Central Asia': 'ECA',
        'Middle East & North Africa': 'MNA',
        'Sub-Saharan Africa': 'SSA',
        'Latin America & Caribbean': 'LAC',
        'East Asia & Pacific': 'EAP',
        'North America': 'NAM'
    }, inplace=True)
    return income_groups


def update_data_coverage(root_dir_, variable, available_countries, imputed_countries=None):
    data_availability_path = os.path.join(root_dir_, "data/processed/data_coverage.csv")
    available_countries = list(available_countries)
    if imputed_countries is None:
        imputed_countries = []
    else:
        imputed_countries = list(imputed_countries)
    if os.path.exists(data_availability_path):
        data_coverage = pd.read_csv(data_availability_path, index_col='iso3')
    else:
        data_coverage = pd.DataFrame()
        data_coverage.index.name = 'iso3'
    if variable in data_coverage.columns:
        data_coverage.drop(variable, axis=1, inplace=True)
    v_coverage = pd.Series(index=available_countries + imputed_countries, dtype=str)
    v_coverage.index.name = 'iso3'
    v_coverage.loc[available_countries] = 'available'
    v_coverage.loc[imputed_countries] = 'imputed'
    data_coverage = pd.concat([data_coverage, v_coverage.rename(variable)], axis=1)
    data_coverage.to_csv(data_availability_path)
