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

import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
import pycountry as pc


def get_world_bank_countries(wb_raw_data_path, download):
    wb_raw_path = os.path.join(wb_raw_data_path, "world_bank_countries.csv")
    if download or not os.path.exists(wb_raw_path):
        url = "https://api.worldbank.org/v2/country"
        params = {
            "format": "json",
            "per_page": 500  # ensures all countries are retrieved in one page
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve data: {response.status_code}")

        data = response.json()

        if not data or len(data) < 2:
            raise Exception("Unexpected API response format.")

        countries = data[1]
        country_data = {country['id']: (str.strip(country['name']), str.strip(country['region']['value']), str.strip(country['incomeLevel']['id'] + 's')) for country in countries if 'id' in country}
        country_df = pd.DataFrame.from_dict(country_data, orient='index', columns=['name', 'region', 'income_group'])
        country_df.index.name = 'iso3'
        country_df.to_csv(wb_raw_path)
    else:
        country_df = pd.read_csv(wb_raw_path, index_col='iso3')

    country_df = country_df[country_df.region != 'Aggregates']
    country_df['region'] = country_df.region.replace(
        {'East Asia & Pacific': 'EAP', 'Europe & Central Asia': 'ECA', 'Latin America & Caribbean': 'LAC',
         'Middle East, North Africa, Afghanistan & Pakistan': 'MNA', 'North America': 'NMA', 'South Asia': 'SAR',
         'Sub-Saharan Africa': 'SSA'}
    )
    country_df['income_group'] = country_df.income_group.replace({'LMCs': 'LMICs', 'UMCs': 'UMICs'})

    if country_df.loc['VEN', 'income_group'] == 'INXs':  # use most recent available value for VEN
        country_df.loc['VEN', 'income_group'] = 'UMICs'
    if country_df.loc['ETH', 'income_group'] == 'INXs':  # use most recent available value for ETH
        country_df.loc['ETH', 'income_group'] = 'LICs'
    return country_df


def get_population_scope_indices(scope, df):
    """
        Get the indices of population scope based on the given scope and DataFrame.

        Args:
            scope (list of tuple): A list of tuples where each tuple contains two values representing the lower and upper bounds of the scope.
            df (pd.DataFrame): Input DataFrame with a multi-index that includes 'income_cat'.

        Returns:
            list: A list of indices corresponding to the 'income_cat' levels within the specified scope.

        Raises:
            KeyError: If 'income_cat' is not found in the DataFrame index.
        """
    income_cat_indices = np.sort(df.index.get_level_values('income_cat').unique())
    indices = []
    for a, b in scope:
        lower, upper = sorted([a, b])
        indices += income_cat_indices[(income_cat_indices > lower) & (income_cat_indices <= upper)].tolist()
    return indices


def get_list_of_index_names(df):
    """
    Get the names of the index in a DataFrame as a list.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: A list containing the names of the index. If the DataFrame has a single index, the list will contain one element.
    """
    
    if df.index.name is None:
        return list(df.index.names)
    else:
        return [df.index.name]

 
def concat_categories(p, np, index):
    """
    Concatenate two pandas objects with keys, swapping the index so that the new index is innermost.

    Args:
        p (pd.DataFrame or pd.Series): First pandas object.
        np (pd.DataFrame or pd.Series): Second pandas object.
        index (pd.Index): Index to use as the innermost index.

    Returns:
        pd.DataFrame or pd.Series: Concatenated object with the new index as the innermost level.

    Raises:
        Exception: If the provided index is not named.
    """
    
    if index.name is None:
        raise Exception("index should be named")
        
    y = pd.concat([p, np],
        keys = index, 
        names = [index.name]+get_list_of_index_names(p))
    
    #puts new index at the end            
    y = y.reset_index(index.name).set_index(index.name, append=True).sort_index()
    
    #makes sure a series is returned when possible
    return y.squeeze()


def average_over_rp(d_in, protection_=None, zero_rp=1):
    """
    Aggregate outputs over return periods, weighted by probabilities.

    Args:
        d_in (pd.DataFrame or pd.Series): Input DataFrame or Series with a 'rp' level in the index.
        protection_ (optional): Protection levels to adjust probabilities. Can be a pandas Series or DataFrame.

    Returns:
        pd.DataFrame or pd.Series: Aggregated object with probabilities applied.
    """

    if isinstance(d_in, pd.Series):
        df_in = d_in.to_frame()
    else:
        df_in = d_in.copy()

    if 'rp' not in df_in.index.names:
        raise ValueError("Need index level 'rp' to average over return periods.")

    if zero_rp is not None:
        if zero_rp <= 0:
            raise ValueError("zero_rp should be > 0")
        elif zero_rp not in df_in.index.get_level_values('rp'):
            pairs = df_in.index.droplevel("rp").unique()
            new_index = pd.MultiIndex.from_arrays(
                [pairs.get_level_values(0), pairs.get_level_values(1), [zero_rp] * len(pairs)],
                names=df_in.index.names
            )
            new_rows = pd.DataFrame(0, index=new_index, columns=df_in.columns)
            df = pd.concat([df_in, new_rows]).sort_index().copy()
        else:
            print(f"Warning: zero_rp={zero_rp} was provided for return period averaging, but return period is "
                  f"already in df_in. Ignoring zero_rp.")
            df = df_in.copy()
    else:
        df = df_in.copy()

    group_cols = [c for c in df.reset_index().columns if c not in df.columns and c != "rp"]

    # removes events below the protection level
    if protection_ is not None:
        if isinstance(protection_, pd.DataFrame):
            protection = protection_.protection.copy().round(3)
        else:
            protection = protection_.copy().round(3)
        protection = protection[protection > min(df.index.get_level_values('rp').unique())]
        protected_index = protection.rename('rp').to_frame().set_index('rp', append=True).index
        protected_levels = pd.DataFrame(np.nan, index=protected_index.difference(df.index), columns=df.columns)
        df = pd.concat([df, protected_levels]).sort_index()

        if group_cols:
            df = df.reset_index().set_index(group_cols).groupby(group_cols).apply(lambda g: g.reset_index().drop(columns=group_cols).set_index('rp').interpolate(method='index'))
        else:
            df = df.sort_index().interpolate(method='index')

        df = df[((df.reset_index('rp').rp - protection).fillna(0) >= 0).values]

    def calculate_rp_average(g):
        g_ = g.sort_index().copy()
        g_.loc[np.inf] = g_.iloc[-1]
        rp_weights = pd.Series(1 / g_.index, index=g_.index).diff(-1).loc[g.sort_index().index]
        return pd.DataFrame((g_.values[:-1] + g_.values[1:]) / 2, index=g.sort_index().index, columns=g_.columns).mul(rp_weights, axis=0).sum()

    res = df.groupby(group_cols, group_keys=False).apply(lambda g: calculate_rp_average(g.droplevel(group_cols)))
    res.loc[d_in.droplevel('rp').index.unique().difference(res.index)] = 0

    if isinstance(d_in, pd.Series):
        res.name = d_in.name
        res = res.squeeze()
    return res


def get_country_name_dicts(root_dir):
    """
    Load country name mappings and return dictionaries for name conversions.

    Args:
        root_dir (str): Root directory containing the country name mapping files.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Mapping from any name to World Bank name.
            - pd.Series: Mapping from ISO3 to World Bank name.
            - pd.Series: Mapping from ISO2 to ISO3.
    """
    # Country dictionaries
    any_to_wb = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/any_name_to_wb_name.csv"), index_col="any").squeeze()

    # iso3 to wb country name table
    iso3_to_wb = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/iso3_to_wb_name.csv"), index_col="iso3").squeeze()

    iso2_iso3 = pd.read_csv(os.path.join(root_dir, "data/raw/country_name_mappings/iso2_to_iso3.csv"), index_col="iso2").squeeze()

    return any_to_wb, iso3_to_wb, iso2_iso3


def df_to_iso3(df_, column_name_, any_to_wb_=None, verbose_=True):
    """
    Add an ISO3 column to a DataFrame based on a country name column.

    Args:
        df_ (pd.DataFrame): Input DataFrame.
        column_name_ (str): Name of the column containing country names.
        any_to_wb_ (pd.Series, optional): Mapping from any name to World Bank name. Defaults to None.
        verbose_ (bool, optional): Whether to print warnings for unmatched names. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with an added 'iso3' column.
    """
    if 'iso3' in df_:
        raise Exception("iso3 column already exists")

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
        'cote d\'ivoire': 'CIV',
        'ivory coast': 'CIV',
        'ethiopia pdr': 'ETH',
        'hong kong sar, china': 'HKG',
        'china, hong kong special administrative region': 'HKG',
        'hong kong, china': 'HKG',
        'china, hong kong sar': 'HKG',
        'macao sar, china': 'MAC',
        'china, macao sar': 'MAC',
        'china, macao special administrative region': 'MAC',
        'macau, china': 'MAC',
        'macao, china': 'MAC',
        'taiwan, china': 'TWN',
        'china, taiwan province of': 'TWN',
        'china, mainland': 'CHN',
        'korea, rep.': 'KOR',
        'republic of korea': 'KOR',
        'korea, south': 'KOR',
        "korea, dem. people's rep.": 'PRK',
        'netherlands (kingdom of the)': 'NLD',
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
        'laos': 'LAO',
        'turkiye': 'TUR',
        'turkey (turkiye)': 'TUR',
        'turkey': 'TUR',
        'türkiye': 'TUR',
        'west bank and gaza': 'PSE',
        'gaza strip': 'PSE',
        'west bank': 'PSE',
        'occupied palestinian territory': 'PSE',
        'palestine': 'PSE',
        'yemen, rep.': 'YEM',
        'kosovo': 'XKX',
        'tanzania, united rep.': 'TZA',
        'bosnia herzegovina': 'BIH',
    }

    def get_iso3(name):
        # hard coded country names:
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
                    hard_coded[str.lower(name)] = None
                    return None
            if len(fuzzy_search_res) == 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found {fuzzy_search_res[0].name}")
                    hard_coded[str.lower(name)] = fuzzy_search_res[0].alpha_3
                return fuzzy_search_res[0].alpha_3
            elif len(fuzzy_search_res) > 1:
                if verbose_:
                    print(f"Warning: {name} not found in pycountry, but fuzzy search found multiple matches: {fuzzy_search_res}")
                hard_coded[str.lower(name)] = None
                return None

    df = df_.copy()
    df['iso3'] = df[column_name_].apply(lambda x: get_iso3(x))
    if df.iso3.isna().any() and verbose_:
        print(f"Warning: ISO3 could not be found for {len(df[df.iso3.isna()][column_name_].unique())} countries.")
    return df


def load_income_groups(root_dir_):
    """
    Load income group classifications for countries.

    Args:
        root_dir_ (str): Root directory containing the income group classification file.

    Returns:
        pd.DataFrame: DataFrame with income group and region information for each country.
    """
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
    """
    Update the data coverage file with availability and imputation information for a variable.

    Args:
        root_dir_ (str): Root directory containing the data coverage file.
        variable (str): Name of the variable to update.
        available_countries (list): List of countries where the variable is available.
        imputed_countries (list, optional): List of countries where the variable is imputed. Defaults to None.

    Returns:
        None
    """
    available_countries = list(available_countries)
    data_availability_path = os.path.join(root_dir_, "data/processed/data_coverage.csv")
    if not os.path.exists(data_availability_path) or (variable == '__purge__' and available_countries == []):
        data_coverage = pd.DataFrame()
        data_coverage.index.name = 'iso3'
    else:
        data_coverage = pd.read_csv(data_availability_path, index_col='iso3')

    if variable != '__purge__' and len(available_countries) > 0:
        if imputed_countries is None:
            imputed_countries = []
        else:
            imputed_countries = list(imputed_countries)
        if variable in data_coverage.columns:
            data_coverage.drop(variable, axis=1, inplace=True)
        v_coverage = pd.Series(index=available_countries + imputed_countries, dtype=str)
        v_coverage.index.name = 'iso3'
        v_coverage.loc[available_countries] = 'available'
        v_coverage.loc[imputed_countries] = 'imputed'
        data_coverage = pd.concat([data_coverage, v_coverage.rename(variable)], axis=1)
    data_coverage.to_csv(data_availability_path)


def calculate_average_recovery_duration(df, aggregation_level, hazard_protection_=None, agg_rp=None):
    df = df[['n', 't_reco_95']].xs('a', level='affected_cat').copy()

    if agg_rp is not None:
        if hazard_protection_ is not None:
            df['rp'] = df.reset_index().rp.values
            df = pd.merge(df, hazard_protection_, left_index=True, right_index=True, how='left')
            df.loc[df[df.rp <= df.protection].index, 'n'] = 0
            df = df.drop(columns=['protection', 'rp'])
        return df.xs(agg_rp, level='rp').groupby(aggregation_level).apply(lambda x: np.average(x.t_reco_95, weights=x.n) if np.sum(x.n) > 0 else 0).squeeze().rename('t_reco_avg')

    if type(aggregation_level) is str:
        aggregation_level = [aggregation_level]

    # compute population-weighted average recovery duration of affected households

    df_ = (df.t_reco_95 * df.n).groupby(aggregation_level + ['rp']).sum() / df.n.groupby(aggregation_level + ['rp']).sum()
    df_[df.n.groupby(aggregation_level + ['rp']).sum() == 0] = 0
    df = df_.rename('t_reco_avg')
    if 1 not in df.index.get_level_values('rp'):
        # for aggregation, assume that all events with rp < 10 have the same recovery duration as the event with rp = 10
        rp_1 = df.xs(df.index.get_level_values('rp').min(), level='rp').copy().to_frame()
        rp_1['rp'] = 1
        rp_1 = rp_1.set_index('rp', append=True).reorder_levels(df.index.names).squeeze()
        df = pd.concat([df, rp_1]).sort_index()

    # compute probability of each return period
    return_periods = df.index.get_level_values('rp').unique()

    rp_probabilities = pd.Series(1 / return_periods - np.append(1 / return_periods, 0)[1:], index=return_periods)
    # match return periods and their frequency
    probabilities = pd.Series(data=df.reset_index('rp').rp.replace(rp_probabilities).values, index=df.index,
                              name='probability').to_frame()

    if hazard_protection_ is not None:
        probabilities['rp'] = probabilities.reset_index().rp.values
        probabilities = pd.merge(probabilities, hazard_protection_, left_index=True, right_index=True, how='left')
        probabilities.loc[probabilities[probabilities.rp <= probabilities.protection].index, 'probability'] = 0
        probabilities = probabilities.drop(columns=['protection', 'rp'])

    # average weighted by probability
    res = pd.merge(df, probabilities, left_index=True, right_index=True, how='left')
    res = ((res.t_reco_avg * res.probability).groupby(aggregation_level).sum() / res.probability.groupby(aggregation_level).sum()).rename('t_reco_avg')
    if type(df) is pd.Series:
        res.name = df.name

    return res
