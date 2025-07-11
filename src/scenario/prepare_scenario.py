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


import argparse
import os
import numpy as np
import yaml
from scipy.optimize import curve_fit
from src.scenario.data_processing.gather_findex_data import get_liquidity_from_findex, gather_axfin_data
from src.scenario.data_processing.gather_gem_data import gather_gem_data
from src.scenario.data_processing.gather_gir_data import load_giri_hazard_loss_rel
from src.scenario.data_processing.get_wb_data import get_wb_data, get_wb_mrv, broadcast_to_population_resolution
import pandas as pd
from src.misc.helpers import get_country_name_dicts, df_to_iso3, load_income_groups, update_data_coverage, \
    get_population_scope_indices
from time import time

from src.scenario.data_processing.process_flopros_data import process_flopros_data
from src.scenario.data_processing.process_peb_data import process_peb_data


def get_cat_info_and_tau_tax(cat_info_, wb_data_macro_, avg_prod_k_, resolution, axfin_impact_,
                             scale_non_diversified_income_=None, min_diversified_share_=None, scale_income_=None,
                             scale_gini_index_=None):
    """
    Computes diversified income share, tax rate (tau_tax), and capital per income category for a given dataset.

    Parameters:
        cat_info_ (pd.DataFrame): DataFrame containing income category information, including transfers, income shares, and financial inclusion.
        wb_data_macro_ (pd.DataFrame): DataFrame containing macroeconomic data, including GDP per capita and population.
        avg_prod_k_ (pd.Series): Average productivity of capital.
        resolution (float): Population resolution for scaling calculations.
        axfin_impact_ (float): Impact factor for financial access (axfin) on diversified income share.
        scale_non_diversified_income_ (dict, optional): Dictionary with 'scope' and 'parameter' to scale non-diversified income.
        min_diversified_share_ (dict, optional): Dictionary with 'scope' and 'parameter' to set a minimum diversified share.
        scale_income_ (dict, optional): Dictionary with 'scope' and 'parameter' to scale income.
        scale_gini_index_ (dict, optional): Dictionary with 'parameter' to adjust income shares for Gini index reduction.

    Returns:
        tuple: A tuple containing:
            - wb_data_macro_ (pd.DataFrame): Updated macroeconomic data with adjusted GDP per capita.
            - cat_info_ (pd.DataFrame): Updated income category data with computed fields.
            - tau_tax_ (pd.Series): Tax rate per country.
    """
    print("Computing diversified income share and tax...")
    cat_info_['diversified_share'] = cat_info_.transfers + cat_info_.axfin * axfin_impact_
    if scale_non_diversified_income_ is not None:
        scope = get_population_scope_indices(scale_non_diversified_income_['scope'], cat_info_)
        parameter = scale_non_diversified_income_['parameter']
        cat_info_.loc[pd.IndexSlice[:, scope], 'diversified_share'] = 1 - (1 - cat_info_['diversified_share']) * parameter
    cat_info_['diversified_share'] = cat_info_['diversified_share'].clip(upper=1, lower=0)
    if min_diversified_share_ is not None:
        scope = get_population_scope_indices(min_diversified_share_['scope'], cat_info_)
        parameter = min_diversified_share_['parameter']
        cat_info_.loc[pd.IndexSlice[:, scope], 'diversified_share'] = cat_info_['diversified_share'].clip(lower=parameter)

    # reduces the gini index by redistributing a certain percentage of each quintile's income equally among all quintiles
    if scale_gini_index_ is not None:
        cat_info_.income_share = cat_info_.income_share * scale_gini_index_['parameter'] + (cat_info_.income_share * (1 - scale_gini_index_['parameter'])).groupby('iso3').sum() * resolution

    cat_info_['n'] = resolution
    cat_info_['c'] = cat_info_.income_share / cat_info_.n * wb_data_macro_.gdp_pc_pp

    if scale_income_ is not None:
        scope = get_population_scope_indices(scale_income_['scope'], cat_info_)
        parameter = scale_income_['parameter']
        cat_info_.loc[pd.IndexSlice[:, scope], 'c'] *= parameter
        wb_data_macro_.gdp_pc_pp = cat_info_.groupby('iso3').c.mean()
        cat_info_['income_share'] = cat_info_.c * cat_info_.n / wb_data_macro_.gdp_pc_pp

    # compute tau tax and gamma_sp from social
    tau_tax_, cat_info_["gamma_SP"] = social_to_tx_and_gsp(cat_info_)

    # compute capital per income category
    cat_info_["k"] = (1 - cat_info_.diversified_share) * cat_info_.c / ((1 - tau_tax_) * avg_prod_k_)

    cat_info_ = cat_info_.loc[cat_info_.drop('n', axis=1).dropna(how='all').index]
    return wb_data_macro_, cat_info_, tau_tax_


def load_protection(index_, root_dir_, protection_data="FLOPROS", min_rp=1, hazard_types="Flood+Storm surge",
                    flopros_protection_file="flopros_protection_processed.csv",
                    protection_level_assumptions_file="WB_country_classification/protection_level_assumptions.csv",
                    income_groups_file="WB_country_classification/country_classification.xlsx",
                    income_groups_file_historical="WB_country_classification/country_classification_historical.xlsx",
                    force_recompute_=True):
    """
    Loads and processes protection level data for countries based on the specified protection data source.

    Parameters:
        index_ (pd.Index): MultiIndex containing country and hazard information.
        root_dir_ (str): Root directory of the project where data files are located.
        protection_data (str, optional): Source of protection data. Options are 'FLOPROS', 'country_income', or 'None'.
        min_rp (int, optional): Minimum protection level to apply. Defaults to 1.
        hazard_types (str, optional): Hazard types to include, separated by '+'. Defaults to "Flood+Storm surge".
        flopros_protection_file (str, optional): File name for FLOPROS protection data. Defaults to "flopros_protection_processed.csv", generated by script process_flopros_data.py.
        protection_level_assumptions_file (str, optional): File name for protection level assumptions. Defaults to "WB_country_classification/protection_level_assumptions.csv".
        income_groups_file (str, optional): File name for income group classification. Defaults to "WB_country_classification/country_classification.xlsx".
        income_groups_file_historical (str, optional): File name for historical income group classification. Defaults to "WB_country_classification/country_classification_historical.xlsx".

    Returns:
        pd.Series: A Series indexed by country and hazard, containing protection levels.
    """
    protection_path = os.path.join(root_dir_, "data/processed/", flopros_protection_file)
    if force_recompute_ or not os.path.exists(protection_path):
        process_flopros_data(
            flopros_path=os.path.join(root_dir_, "data/raw/FLOPROS"),
            population_path=os.path.join(root_dir_, "data/raw/GPW/gpw_v4_population_density_adjusted_rev11_2pt5_min.nc"),
            gadm_path=os.path.join(root_dir_, "data/raw/GADM/gadm_410-levels.gpkg"),
            outpath=os.path.join(root_dir_, "data/processed/")
        )
    else:
        print("Loading protection data...")
    if 'rp' in index_.names:
        index_ = index_.droplevel('rp').drop_duplicates()
    if 'income_cat' in index_.names:
        index_ = index_.droplevel('income_cat').drop_duplicates()
    prot = pd.Series(index=index_, name="protection", data=0.)
    if protection_data in ['FLOPROS', 'country_income']:
        if protection_data == 'FLOPROS':
            prot_data = pd.read_csv(protection_path, index_col=0)
            prot_data = prot_data.drop('country', axis=1)
            prot_data.rename({'MerL_Riv': 'Flood', 'MerL_Co': 'Storm surge'}, axis=1, inplace=True)
            update_data_coverage(root_dir_, 'flopros', prot_data.index, None)
        elif protection_data == 'country_income':  # assumed a function of the country's income group
            prot_assumptions = pd.read_csv(os.path.join(root_dir_, "data/raw/", protection_level_assumptions_file),
                                               index_col="Income group").squeeze()
            prot_data = pd.read_csv(os.path.join(root_dir_, "data/raw/", income_groups_file), header=0)[["Code", "Income group"]]
            prot_data = prot_data.dropna(how='all').rename({'Code': 'iso3', 'Income group': 'protection'}, axis=1)
            prot_data = prot_data.set_index('iso3').squeeze()
            prot_data.replace(prot_assumptions, inplace=True)

            # use historical data for countries that are not in the income_groups_file
            prot_data_hist = pd.read_excel(os.path.join(root_dir_, "data/raw/", income_groups_file_historical),
                                             sheet_name='Country Analytical History', header=5)
            prot_data_hist = prot_data_hist.rename({prot_data_hist.columns[0]: 'iso3'}, axis=1).dropna(subset='iso3')
            prot_data_hist.set_index('iso3', inplace=True)
            prot_data_hist.drop(prot_data_hist.columns[0], axis=1, inplace=True)
            prot_data_hist = prot_data_hist.stack().rename('protection').replace(prot_assumptions)
            prot_data_hist.index.names = ['iso3', 'year']
            prot_data_hist = prot_data_hist.reset_index()
            prot_data_hist = prot_data_hist.loc[prot_data_hist.groupby('iso3')['year'].idxmax()]
            prot_data_hist = prot_data_hist.set_index('iso3')['protection']
            prot_data = prot_data.fillna(prot_data_hist).dropna()

            hazards = hazard_types.split('+')
            prot_data = pd.concat([prot_data] * len(hazards), axis=1)
            prot_data.columns = hazards
        else:
            raise ValueError(f"Unknown value for protection_data: {protection_data}")
        prot_data = prot_data.stack()
        prot_data.index.names = ['iso3', 'hazard']
        prot_data.name = 'protection'
        prot.loc[np.intersect1d(prot_data.index, prot.index)] = prot_data
    elif protection_data == 'None':
        print("No protection data applied.")
    else:
        raise ValueError(f"Unknown value for protection_data: {protection_data}")
    if min_rp is not None:
        prot.loc[prot < min_rp] = min_rp
    return prot


def load_findex_liquidity_and_axfin(root_dir_, any_to_wb_, gni_pc_pp, resolution, force_recompute_=True, verbose_=True, scale_liquidity_=None):
    """
    Loads or computes liquidity and access to finance (axfin) data from FINDEX.

    Args:
        root_dir_ (str): Root directory of the project.
        any_to_wb_ (dict): Mapping of country names to World Bank codes.
        gni_pc_pp (pd.Series): Gross national income per capita (PPP).
        resolution (float): Population resolution for broadcasting data.
        force_recompute_ (bool): Whether to force recomputation of data. Defaults to True. Method loads data from file if False.
        verbose_ (bool): Whether to print verbose output. Defaults to True.
        scale_liquidity_ (dict, optional): Scaling parameters for liquidity. Defaults to None.

    Returns:
        pd.DataFrame: Liquidity and access to finance data indexed by country and income category.
    """

    outpath = os.path.join(root_dir_, 'data', 'processed', 'findex_liquidity_and_axfin.csv')
    if not force_recompute_ and os.path.exists(outpath):
        print("Loading liquidity and axfin data from file...")
        liquidity_and_axfin = pd.read_csv(outpath)
        liquidity_and_axfin.set_index(['iso3', 'income_cat'], inplace=True)
    else:
        print("Recomputing liquidity and axfin data from FINDEX...")
        findex_data_paths = {
            2021: os.path.join(root_dir_, "data/raw/", 'FINDEX', 'WLD_2021_FINDEX_v03_M.csv'),
            2017: os.path.join(root_dir_, "data/raw/", 'FINDEX', 'WLD_2017_FINDEX_v02_M.csv'),
            2014: os.path.join(root_dir_, "data/raw/", 'FINDEX', 'WLD_2014_FINDEX_v01_M.csv'),
            2011: os.path.join(root_dir_, "data/raw/", 'FINDEX', 'WLD_2011_FINDEX_v02_M.csv'),
        }
        liquidity_ = get_liquidity_from_findex(root_dir_, any_to_wb_, findex_data_paths, gni_pc_pp, verbose=verbose_)
        liquidity_ = liquidity_[['liquidity_share', 'liquidity']].prod(axis=1).rename('liquidity')

        axfin_ = gather_axfin_data(root_dir_, any_to_wb_, findex_data_paths, verbose=verbose_)
        axfin_ = axfin_.iloc[axfin_.reset_index().groupby(['iso3', 'income_cat']).year.idxmax()].axfin.droplevel('year')

        liquidity_and_axfin = pd.merge(liquidity_, axfin_, left_index=True, right_index=True, how='inner')

        update_data_coverage(root_dir_, 'findex', liquidity_and_axfin.index.get_level_values('iso3').unique(), None)

        liquidity_and_axfin.to_csv(outpath)
    liquidity_and_axfin = broadcast_to_population_resolution(liquidity_and_axfin, resolution)
    if scale_liquidity_ is not None:
        scope = get_population_scope_indices(scale_liquidity_['scope'], liquidity_and_axfin)
        liquidity_and_axfin.loc[pd.IndexSlice[:, scope], 'liquidity'] *= scale_liquidity_['parameter']
    return liquidity_and_axfin


def calc_real_estate_share_of_value_added(root_dir, any_to_wb, verbose=True, most_recent_value=True):
    """
    Calculates the share of real estate activities in value added and GDP.

    Args:
        root_dir (str): Root directory of the project.
        any_to_wb (dict): Mapping of country names to World Bank codes.
        verbose (bool): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: Real estate share of value added indexed by country.
    """

    value_added = pd.read_csv(os.path.join(root_dir, "data/raw/UNdata/2025-02-13_value_added_by_industry.csv"))[['Country or Area', 'Item', 'Year', 'Series', 'SNA System', 'Fiscal Year Type', 'Value']]
    value_added.rename({'Country or Area': 'country'}, axis=1, inplace=True)

    # prioritize the Western calendare year if available, select the FY type with the highest priority
    value_added['FY_prio'] = value_added['Fiscal Year Type'].replace({
        'Western calendar year': 6,
        'Fiscal year beginning 21 March': 5,
        'Fiscal year ending 30 September': 4,
        'Fiscal year beginning 1 April': 3,
        'Fiscal year ending 15 July': 2,
        'Fiscal year beginning 1 July': 1,
        'Fiscal year ending 30 June': 0,
    })
    value_added = value_added.loc[value_added.groupby(['country', 'Item', 'Year', 'Series']).FY_prio.idxmax()].drop(['FY_prio', 'Fiscal Year Type'], axis=1)

    # keep both values of 1993 and 2008 SNA systems, then select the most recent series (4 digit numbers are the 2008
    # system, while 3 digit numbers are the 1993 system). "A higher number indicates more recent data"
    # retain the most recent available series
    value_added = value_added.loc[value_added.groupby(['country', 'Item', 'Year']).Series.idxmax()].drop(['Series', 'SNA System'], axis=1)

    value_added = value_added.set_index(['country', 'Item', 'Year']).Value.rename('Value Added')

    # add Zanzibar to Tanzania
    tza = (value_added.loc['Tanzania - Mainland'] + value_added.loc['Zanzibar']).dropna().reset_index()
    tza['country'] = 'Tanzania'
    tza = tza.set_index(['country', 'Item', 'Year'])['Value Added']
    value_added = pd.concat([value_added, tza]).drop(['Tanzania - Mainland', 'Zanzibar'], level='country')

    # compute the share of real estate activities to GDP
    real_est_gdp_share = value_added.xs('Real estate activities', level='Item') / value_added.xs('Equals: GROSS DOMESTIC PRODUCT', level='Item')
    real_est_value_added_share = value_added.xs('Real estate activities', level='Item') / value_added.xs('Equals: VALUE ADDED, GROSS, at basic prices', level='Item')
    merged = pd.merge(real_est_value_added_share, real_est_gdp_share, left_index=True, right_index=True, how='outer')

    merged.drop('Sint Maarten', inplace=True)

    real_est_value_added_share = merged.iloc[:, 0].fillna(merged.iloc[:, 1]).dropna().rename('real_estate_share_of_value_added').reset_index()

    # retain only the most recent year
    if most_recent_value:
        real_est_value_added_share = real_est_value_added_share.loc[real_est_value_added_share.groupby('country').Year.idxmax()].drop('Year', axis=1)

    # replace country names with iso3 codes
    real_est_value_added_share = df_to_iso3(real_est_value_added_share, 'country', any_to_wb, verbose_=verbose).set_index('iso3').drop('country', axis=1)

    return real_est_value_added_share


def load_home_ownership_rates(root_dir, any_to_wb, verbose):
    """
    Loads and processes home ownership rates from various sources.

    Args:
        root_dir (str): Root directory of the project.
        any_to_wb (dict): Mapping of country names to World Bank codes.
        verbose (bool): Whether to print verbose output.

    Returns:
        pd.Series: Home ownership rates indexed by country.
    """
    un_tenure = pd.read_csv(os.path.join(root_dir, "data/raw/UNdata/2025-02-18_household_tenure.csv"))
    un_tenure.rename(columns={'Country or Area': 'country'}, inplace=True)
    un_tenure = un_tenure[['country', 'Year', 'Area', 'Type of housing unit', 'Tenure', 'Value']].dropna()
    un_tenure = un_tenure[un_tenure.Area == 'Total'].drop('Area', axis=1)

    un_tenure = un_tenure.set_index(['country', 'Year', 'Type of housing unit', 'Tenure']).Value

    un_hor = (un_tenure.xs('Member of household owns the housing unit', level='Tenure') / un_tenure.xs('Total', level='Tenure')).dropna()

    un_hor = un_hor.reset_index()

    un_hor['Type_prio'] = un_hor['Type of housing unit'].replace({
        'Total': 4,
        'Conventional dwellings': 3,
        'Other housing units': 2,
        'Unknown (type of housing unit)': 1,
    })
    un_hor = un_hor.loc[un_hor.groupby(['country', 'Year'])['Type_prio'].idxmax()].drop(['Type_prio', 'Type of housing unit'], axis=1)
    un_hor = un_hor.loc[un_hor.groupby('country').Year.idxmax()].drop('Year', axis=1)

    un_hor = df_to_iso3(un_hor, 'country', any_to_wb, verbose_=verbose).dropna().set_index('iso3').Value.rename('home_ownership_rate')

    hor_oecd = pd.read_excel(os.path.join(root_dir, "data/raw/Home_ownership_rates/home_ownership_rates.xlsx"), sheet_name="OECD", header=0)
    hor_eurostat = pd.read_excel(os.path.join(root_dir, "data/raw/Home_ownership_rates/home_ownership_rates.xlsx"), sheet_name="Eurostat", header=0)
    hor_cahf = pd.read_excel(os.path.join(root_dir, "data/raw/Home_ownership_rates/home_ownership_rates.xlsx"), sheet_name="CAHF", header=0)

    hor = None
    for df in [hor_oecd, hor_eurostat, hor_cahf]:
        df.columns.name = 'income_cat'
        df = df_to_iso3(df, 'country', any_to_wb, verbose_=verbose).dropna()
        df = df.set_index('iso3').drop('country', axis=1).stack().rename('home_ownership_rate')
        if hor is None:
            hor = df
        else:
            hor = pd.merge(hor, df, left_index=True, right_index=True, how='outer')
            hor['home_ownership_rate'] = hor['home_ownership_rate_x'].fillna(hor['home_ownership_rate_y'])
            hor.drop(['home_ownership_rate_x', 'home_ownership_rate_y'], axis=1, inplace=True)
    hor = pd.merge(hor.unstack('income_cat').mean(axis=1).rename('home_ownership_rate'), un_hor, left_index=True, right_index=True, how='outer')
    hor = hor['home_ownership_rate_x'].fillna(hor['home_ownership_rate_y']).rename('home_ownership_rate')
    return hor


def estimate_real_est_k_to_va_shares_ratio(root_dir, any_to_wb, verbose):
    """
    Estimates the ratio of real estate capital to value-added shares.

    Args:
        root_dir (str): Root directory of the project.
        any_to_wb (dict): Mapping of country names to World Bank codes.
        verbose (bool): Whether to print verbose output.

    Returns:
        float: Estimated ratio of real estate capital to value-added shares.
    """
    estat_capital_industry = pd.read_csv(os.path.join(root_dir, "data/raw/Eurostat/eurostat__nama_10_nfa_st__capital_stock.csv"))[['nace_r2', 'asset10', 'geo', 'TIME_PERIOD', 'OBS_VALUE']]
    estat_capital_industry.rename({'OBS_VALUE': 'capital_stock', 'geo': 'country', 'TIME_PERIOD': 'year'}, axis=1, inplace=True)
    estat_capital_industry = df_to_iso3(estat_capital_industry, 'country', any_to_wb, verbose_=verbose)
    estat_capital_industry = estat_capital_industry.set_index(['iso3', 'year', 'nace_r2', 'asset10'])['capital_stock']
    estat_capital_industry *= 1e6

    estat_k_real_est_share = estat_capital_industry.loc[pd.IndexSlice[:, :, 'Real estate activities', 'Total fixed assets (gross)']] / estat_capital_industry.loc[pd.IndexSlice[:, :, 'Total - all NACE activities', 'Total fixed assets (gross)']]
    estat_k_real_est_share = estat_k_real_est_share.rename('k_real_estate_share')

    estat_value_added = pd.read_csv(os.path.join(root_dir, "data/raw/Eurostat/eurostat__nama_10_a64__value_added.csv"))[['nace_r2', 'geo', 'TIME_PERIOD', 'OBS_VALUE']]
    estat_value_added.rename({'OBS_VALUE': 'value_added', 'geo': 'country', 'TIME_PERIOD': 'year'}, axis=1, inplace=True)
    estat_value_added = df_to_iso3(estat_value_added, 'country', any_to_wb, verbose_=verbose)
    estat_value_added = estat_value_added.set_index(['iso3', 'year', 'nace_r2'])['value_added']
    estat_value_added *= 1e6

    estat_real_estate_share_of_value_added = estat_value_added.xs('Real estate activities', level='nace_r2') / estat_value_added.xs('Total - all NACE activities', level='nace_r2')
    estat_real_estate_share_of_value_added = estat_real_estate_share_of_value_added.rename('real_estate_share_of_value_added')

    estat_data = pd.merge(estat_real_estate_share_of_value_added, estat_k_real_est_share, left_index=True,right_index=True, how='outer').dropna(how='all')

    def kappa_hous_m(idx, m_):
        real_estate_share_theta_h = estat_data.loc[idx, 'real_estate_share_of_value_added']
        return m_ * real_estate_share_theta_h
    m_fit_data = estat_data.dropna(subset=['real_estate_share_of_value_added', 'k_real_estate_share'])
    m_opt_result = curve_fit(kappa_hous_m, m_fit_data.index, m_fit_data['k_real_estate_share'], p0=1)
    m_opt = m_opt_result[0][0]
    return m_opt


def calc_asset_shares(root_dir_, any_to_wb_, scale_self_employment=None,
                      verbose=True, force_recompute=True, download=True,
                      guess_missing_countries=False):
    """
    Calculates asset shares, including private, household, and owner-occupied shares.

    Args:
        root_dir_ (str): Root directory of the project.
        any_to_wb_ (dict): Mapping of country names to World Bank codes.
        scale_self_employment (dict, optional): Scaling parameters for self-employment. Defaults to None.
        verbose (bool): Whether to print verbose output. Defaults to True.
        force_recompute (bool): Whether to force recomputation of data. Defaults to True.
        guess_missing_countries (bool): Whether to guess missing data for countries. Defaults to False.

    Returns:
        pd.DataFrame: Asset shares indexed by country.
    """
    capital_shares_before_adjustment_path = os.path.join(root_dir_, "data/processed/capital_shares.csv")
    if not force_recompute and os.path.exists(capital_shares_before_adjustment_path):
        print("Loading capital shares from file...")
        capital_shares = pd.read_csv(capital_shares_before_adjustment_path, index_col='iso3')
    else:
        print("Recomputing capital shares...")
        imf_capital_data_file = "IMF_capital/IMFInvestmentandCapitalStockDataset2021.xlsx"
        imf_data = pd.read_excel(os.path.join(root_dir_, "data/raw/", imf_capital_data_file), sheet_name='Dataset')
        imf_data = imf_data.rename(columns={'isocode': 'iso3'}).set_index(['iso3', 'year'])

        imf_data_n = imf_data[['kgov_n', 'kpriv_n', 'kppp_n']].dropna(subset=['kgov_n', 'kpriv_n']).fillna(0)
        k_pub_share_n = (imf_data_n[['kgov_n', 'kppp_n']].sum(axis=1) / imf_data_n.sum(axis=1)).dropna()

        imf_data_ppp = imf_data[['kgov_rppp', 'kpriv_rppp', 'kppp_rppp']].dropna(subset=['kgov_rppp', 'kpriv_rppp']).fillna(0)
        k_pub_share_ppp = (imf_data_ppp[['kgov_rppp', 'kppp_rppp']].sum(axis=1) / imf_data_ppp.sum(axis=1)).dropna()

        k_pub_share = pd.merge(k_pub_share_ppp.rename(0), k_pub_share_n.rename(1), left_index=True, right_index=True, how='outer')
        k_pub_share = k_pub_share[0].fillna(k_pub_share[1]).rename('k_pub_share')
        k_pub_share = k_pub_share.reset_index().loc[k_pub_share.reset_index().groupby('iso3').year.idxmax()].set_index('iso3')['k_pub_share']

        self_employment = get_wb_mrv('SL.EMP.SELF.ZS', 'self_employment', os.path.join(root_dir_, "data/raw/WB_socio_economic_data/API"), download=download) / 100
        self_employment = df_to_iso3(self_employment.reset_index(), 'country', any_to_wb_, verbose_=verbose).dropna(subset='iso3')
        self_employment = self_employment.set_index('iso3').drop('country', axis=1).squeeze()

        home_ownership_rates = load_home_ownership_rates(root_dir_, any_to_wb_, verbose)

        real_estate_share_of_value_added = calc_real_estate_share_of_value_added(root_dir_, any_to_wb_, verbose).squeeze()
        real_est_k_to_va_shares_ratio = estimate_real_est_k_to_va_shares_ratio(root_dir_, any_to_wb_, verbose)

        capital_shares = pd.merge(self_employment, k_pub_share, left_index=True, right_index=True, how='outer')
        capital_shares = pd.merge(home_ownership_rates, capital_shares, left_index=True, right_index=True, how='outer')
        capital_shares = pd.merge(real_estate_share_of_value_added, capital_shares, left_index=True, right_index=True, how='outer')
        capital_shares['real_est_k_to_va_shares_ratio'] = real_est_k_to_va_shares_ratio

        for v in ['k_pub_share', 'real_estate_share_of_value_added', 'home_ownership_rate', 'self_employment']:
            available_countries = capital_shares[capital_shares[v].notna()].index.get_level_values('iso3').unique()
            update_data_coverage(root_dir_, v, available_countries, None)

        # fill na-values with the median of the respective country income group
        if guess_missing_countries:
            income_groups = load_income_groups(root_dir_)
            merged = pd.merge(capital_shares, income_groups, left_index=True, right_index=True, how='outer')

            # keep track of imputed data
            for v in ['real_estate_share_of_value_added', 'home_ownership_rate']:
                available_countries = merged[merged[v].notna()].index.get_level_values('iso3').unique()
                imputed_countries = merged[merged[v].isna()].index.get_level_values('iso3').unique()
                update_data_coverage(root_dir_, v, available_countries, imputed_countries)

            # fill missing values with the median of the respective country income group
            capital_shares['real_estate_share_of_value_added'] = capital_shares['real_estate_share_of_value_added'].fillna(
                merged.groupby('Country income group')['real_estate_share_of_value_added'].transform('median'))
            capital_shares['home_ownership_rate'] = capital_shares['home_ownership_rate'].fillna(
                merged.groupby('Country income group')['home_ownership_rate'].transform('median'))

        capital_shares['k_real_est_share'] = capital_shares['real_estate_share_of_value_added'] * capital_shares['real_est_k_to_va_shares_ratio']
        capital_shares['k_labor_share'] = 1 - capital_shares['k_pub_share'] - capital_shares['k_real_est_share'] * capital_shares['home_ownership_rate']
        capital_shares['owner_occupied_share_of_value_added'] = capital_shares['home_ownership_rate'] * capital_shares['real_estate_share_of_value_added']

        capital_shares.to_csv(capital_shares_before_adjustment_path)

    if scale_self_employment is not None:
        capital_shares['self_employment'] *= scale_self_employment['parameter']

    capital_shares['k_self_share'] = capital_shares['k_labor_share'] * capital_shares['self_employment']
    capital_shares['k_priv_share'] = capital_shares['k_labor_share'] * (1 - capital_shares['self_employment'])
    capital_shares['k_household_share'] = capital_shares['owner_occupied_share_of_value_added'] * capital_shares['real_est_k_to_va_shares_ratio'] + capital_shares['k_self_share']

    return capital_shares.dropna(subset='k_household_share')[['k_priv_share', 'k_household_share', 'owner_occupied_share_of_value_added', 'self_employment', 'real_est_k_to_va_shares_ratio']]


def apply_poverty_exposure_bias(root_dir_, exposure_fa_, resolution, guess_missing_countries, population_data_=None,
                                peb_data_path="exposure_bias_per_quintile.csv", force_recompute_=True):
    """
    Applies poverty exposure bias to exposure data.

    Args:
        root_dir_ (str): Root directory of the project.
        exposure_fa_ (pd.Series): Exposure data indexed by country, hazard, return period, and income category.
        resolution (float): Population resolution for broadcasting data.
        guess_missing_countries (bool): Whether to guess missing data for countries.
        population_data_ (pd.DataFrame, optional): Population data. Defaults to None.
        peb_data_path (str): Path to the poverty exposure bias data file. Defaults to "exposure_bias_per_quintile.csv".

    Returns:
        pd.Series: Exposure data with poverty exposure bias applied.
    """
    bias_path = os.path.join(root_dir_, "data/processed/", peb_data_path)
    if force_recompute_ or not os.path.exists(bias_path):
        print("Recomputing poverty exposure biases...")
        process_peb_data(
            root_dir=root_dir_,
            exposure_data_path="data/raw/PEB/exposure bias.dta",
            poverty_data_path="data/raw/PEB/poverty_data/",
            outpath=os.path.join(root_dir_, "data/processed/"),
            wb_macro_path="data/processed/wb_data_macro.csv",
            exclude_povline=13.7,
        )
    else:
        print("Loading poverty exposure biases from file...")
    bias = pd.read_csv(bias_path)
    bias['income_cat'] = bias.income_cat.apply(lambda x: int(x.split('q')[1]))
    bias['income_cat'] = np.round(bias.income_cat / bias.income_cat.max(), 2)
    bias = bias.set_index(['iso3', 'hazard', 'income_cat'])
    bias = broadcast_to_population_resolution(bias, resolution).squeeze()
    eb_hazards = bias.index.get_level_values('hazard').unique()
    missing_index = pd.MultiIndex.from_tuples(
        np.setdiff1d(exposure_fa_.index.droplevel('rp').unique(), bias.index.unique()), names=bias.index.names)
    bias = pd.concat((bias, pd.Series(index=missing_index, data=np.nan)), axis=0).rename(bias.name).sort_index()
    bias = bias.loc[np.intersect1d(bias.index, exposure_fa_.index.droplevel('rp').unique())]

    # if guess_missing_countries, use average EB from global data for countries that don't have EB. Otherwise use 1.
    if guess_missing_countries:
        if population_data_ is not None:
            bias_pop = pd.merge(bias.dropna(), population_data_, left_index=True, right_index=True, how='left')

            # keep track of imputed data
            imputed_countries = missing_index.drop(np.setdiff1d(missing_index.get_level_values('hazard').unique(), eb_hazards), level='hazard').get_level_values('iso3').unique()
            available_countries = np.setdiff1d(exposure_fa_.index.get_level_values('iso3').unique(), imputed_countries)
            update_data_coverage(root_dir_, 'exposure_bias', available_countries, imputed_countries)

            bias_wavg = (bias_pop.product(axis=1).groupby(['hazard', 'income_cat']).sum()
                         / bias_pop['pop'].groupby(['hazard', 'income_cat']).sum()).rename('exposure_bias')
            bias = pd.merge(bias, bias_wavg, left_index=True, right_index=True, how='left')
            bias["exposure_bias_x"] = bias["exposure_bias_x"].fillna(bias.exposure_bias_y)
            bias.drop('exposure_bias_y', axis=1, inplace=True)
            bias = bias.rename({'exposure_bias_x': 'exposure_bias'}, axis=1).squeeze()
            bias = bias.swaplevel('income_cat', 'iso3').swaplevel('hazard', 'iso3').sort_index()
        else:
            raise Exception("Population data not available. Cannot use average PE.")
    bias.fillna(1, inplace=True)
    exposure_fa_with_peb_ = (exposure_fa_ * bias).swaplevel('rp', 'income_cat').sort_index().rename(exposure_fa_.name)
    return exposure_fa_with_peb_


def compute_exposure_and_vulnerability(root_dir_, fa_threshold_, resolution, verbose=True, force_recompute_=False,
                                       apply_exposure_bias=True, population_data=None, scale_exposure=None,
                                       scale_vulnerability=None, early_warning_data=None, reduction_vul=.2,
                                       no_ew_hazards="Earthquake+Tsunami"):
    """
    Computes exposure and vulnerability data for countries and hazards.

    Args:
        root_dir_ (str): Root directory of the project.
        fa_threshold_ (float): Threshold for exposure data. Specified in settings yml file. Should be lower or equal to 1.
        resolution (float): Population resolution for broadcasting data.
        verbose (bool): Whether to print verbose output. Defaults to True.
        force_recompute_ (bool): Whether to force recomputation of data. Defaults to False.
        apply_exposure_bias (bool): Whether to apply poverty exposure bias. Defaults to True.
        population_data (pd.DataFrame, optional): Population data. Defaults to None.
        scale_exposure (dict, optional): Scaling parameters for exposure. Defaults to None.
        scale_vulnerability (dict, optional): Scaling parameters for vulnerability. Defaults to None.
        early_warning_data (pd.Series, optional): Early warning (EW) data. Defaults to None. EW can reduce vulnerability by up to factor reduction_vul.
        reduction_vul (float): Reduction in vulnerability due to early warning. Defaults to 0.2.
        no_ew_hazards (str): Hazards without early warning. Defaults to "Earthquake+Tsunami".

    Returns:
        pd.DataFrame: Exposure and vulnerability data indexed by country, hazard, and income category.
    """
    hazard_ratios_path = os.path.join(root_dir_, 'data', 'processed', 'hazard_ratios.csv')
    if not force_recompute_ and os.path.exists(hazard_ratios_path):
            print("Loading exposure and vulnerability from file...")
            hazard_ratios = pd.read_csv(hazard_ratios_path, index_col=[0, 1, 2, 3]).squeeze()
    else:
        print("Computing exposure and vulnerability...")
        # load total hazard losses per return period (on the country level)
        hazard_loss_rel = load_giri_hazard_loss_rel(
            gir_filepath_=os.path.join(root_dir_, "data/raw/GIR_hazard_loss_data/export_all_metrics.csv.zip"),
            extrapolate_rp_=False,
            climate_scenario="Existing climate",
            verbose=verbose,
        )
        update_data_coverage(root_dir_, 'hazard_loss', hazard_loss_rel.index.get_level_values('iso3').unique(), None)

        # load GEM data
        _, gem_building_classes = load_gem_data(
            root_dir_=root_dir_,
            verbose=verbose,
        )
        update_data_coverage(root_dir_, 'gem_building_classes', gem_building_classes.index.get_level_values('iso3').unique(), None)

        # load building vulnerability classification and compute vulnerability per income class
        vulnerability_unadjusted_ = process_vulnerability_data(
            root_dir_=root_dir_,
            building_classes=gem_building_classes,
            resolution=resolution,
            use_gmd_to_distribute=True,
            fill_missing_gmd_with_country_average=False,
            vulnerability_bounds='gem_extremes',
        )

        exposure_fa_ = (hazard_loss_rel / vulnerability_unadjusted_.xs('tot', level='income_cat')).dropna().rename('fa')

        vulnerability_quintiles = vulnerability_unadjusted_.drop('tot', level='income_cat')

        # merge vulnerability with hazard_ratios
        fa_v_merged = pd.merge(exposure_fa_, vulnerability_quintiles, left_index=True, right_index=True)
        fa_v_merged['fa'] = fa_v_merged['fa'].clip(lower=0, upper=fa_threshold_)

        hazard_ratios = fa_v_merged[['fa', 'v']]
        hazard_ratios.to_csv(hazard_ratios_path)

    # apply poverty exposure bias
    if apply_exposure_bias:
        hazard_ratios['fa'] = apply_poverty_exposure_bias(
            root_dir_=root_dir_,
            exposure_fa_=hazard_ratios['fa'],
            resolution=resolution,
            guess_missing_countries=True,
            population_data_=population_data,
            force_recompute_=force_recompute_,
        ).clip(lower=0, upper=fa_threshold_).values

    for policy_dict, col_name in zip([scale_vulnerability, scale_exposure], ['v', 'fa']):
        if policy_dict is not None:
            scope = get_population_scope_indices(policy_dict['scope'], hazard_ratios)
            parameter = policy_dict['parameter']
            scale_total = policy_dict['scale_total']
            if not scale_total:
                hazard_ratios.loc[pd.IndexSlice[:, :, :, scope], col_name] *= parameter
            else:
                average = hazard_ratios[col_name].groupby(['iso3', 'hazard', 'rp']).mean()
                scope_sum = hazard_ratios.loc[pd.IndexSlice[:, :, :, scope], col_name].groupby(['iso3', 'hazard', 'rp']).sum()
                none_scope_sum = hazard_ratios[col_name].groupby(['iso3', 'hazard', 'rp']).sum() - scope_sum
                scope_factor = (average * parameter - resolution * none_scope_sum) / (resolution * scope_sum)
                hazard_ratios.loc[pd.IndexSlice[:, :, :, scope], col_name] *= scope_factor

    if early_warning_data is not None:
        hazard_ratios["v_ew"] = hazard_ratios["v"] * (1 - reduction_vul * early_warning_data)
        hazard_ratios.loc[pd.IndexSlice[:, no_ew_hazards.split('+'), :, :], "v_ew"] = hazard_ratios.loc[
            pd.IndexSlice[:, no_ew_hazards.split('+'), :, :], "v"]
    return hazard_ratios


def process_vulnerability_data(
        root_dir_, building_classes, resolution, use_gmd_to_distribute=True,
        fill_missing_gmd_with_country_average=False, vulnerability_bounds='gem_extremes',
        building_class_vuln_path="GEM_vulnerability/building_class_to_vulenrability_mapping.csv",
        gmd_vulnerability_distribution_path="GMD_vulnerability_distribution/Dwelling quintile vul ratio.xlsx",
):
    """
    Processes vulnerability data for different building classes and income categories.

    Args:
        root_dir_ (str): Root directory of the project.
        building_classes (pd.DataFrame): DataFrame containing building class distributions by country and hazard.
        resolution (str): Resolution of the data (e.g., national or subnational).
        use_gmd_to_distribute (bool): Whether to use GMD data to distribute vulnerability. Default is True.
        fill_missing_gmd_with_country_average (bool): Whether to fill missing GMD data with country averages. Default is False.
        vulnerability_bounds (str): Method to bound vulnerability values ('gem_extremes' or 'class_extremes'). Default is 'gem_extremes'.
        building_class_vuln_path (str): Path to the building class vulnerability mapping file.
        gmd_vulnerability_distribution_path (str): Path to the GMD vulnerability distribution file.

    Returns:
        pd.DataFrame: Processed vulnerability data indexed by country, hazard, and income category.
    """
    # load vulnerability of building classes
    building_class_vuln = pd.read_csv(os.path.join(root_dir_, "data/raw/", building_class_vuln_path), index_col=0)
    building_class_vuln.index.name = 'hazard'
    building_class_vuln.columns.name = 'vulnerability_class'

    # compute extreme case of vuln. distribution, assuming that the poorest live in the most vulnerable buildings
    quantiles = np.round(np.linspace(.2, 1, 5), 1)
    v_gem = pd.DataFrame(
        index=pd.MultiIndex.from_product((building_classes.index, quantiles), names=['iso3', 'income_cat']),
        columns=pd.Index(building_classes.columns.get_level_values(0).unique(), name='hazard'),
        dtype=float
    )
    for cum_head in quantiles:
        for hazard in v_gem.columns:
            share_h_q = (
                (building_classes[hazard] - (building_classes[hazard].cumsum(axis=1).add(-cum_head, axis=0)).clip(lower=0)).clip(0) -
                (building_classes[hazard] - (building_classes[hazard].cumsum(axis=1).add(-(cum_head - 1 / len(quantiles)), axis=0)).clip(lower=0)).clip(0)) / (1 / len(quantiles))
            v_gem_h_q = (share_h_q * building_class_vuln.loc[hazard]).sum(axis=1, skipna=True)
            v_gem.loc[(slice(None), cum_head), hazard] = v_gem_h_q.values
    v_gem = v_gem.stack().rename('v')
    v_gem = v_gem.reorder_levels(['iso3', 'hazard', 'income_cat']).sort_index()

    # compute total vulnerability per country as the weighted sum of vulnerability classes
    vulnerability_tot = building_classes.mul(building_class_vuln.stack(), axis=1).T.groupby(level='hazard').sum().T
    vulnerability_tot = vulnerability_tot.stack().rename('tot').to_frame()
    vulnerability_tot.columns.name = 'income_cat'
    vulnerability_tot = vulnerability_tot.stack()
    vulnerability_tot = vulnerability_tot.reorder_levels(['iso3', 'hazard', 'income_cat']).sort_index()
    vulnerability_tot = vulnerability_tot.rename('v')

    # always use GEM vulnerability as the baseline
    vulnerability_ = v_gem.copy()

    # use GMD data to distribute GEM national vulnerability
    if use_gmd_to_distribute:
        # load vulnerability distribution as per GMD
        vuln_distr = pd.read_excel(os.path.join(root_dir_, "data/raw/", gmd_vulnerability_distribution_path), sheet_name='Data')
        vuln_distr = vuln_distr.loc[vuln_distr.groupby('code')['year'].idxmax()]
        vuln_distr.rename(
            {'code': 'iso3', 'ratio1': .2, 'ratio2': .4, 'ratio3': .6, 'ratio4': .8, 'ratio5': 1},
            axis=1, inplace=True
        )
        vuln_distr = vuln_distr.set_index('iso3')[[.2, .4, .6, .8, 1]]

        available_countries = vuln_distr.index
        update_data_coverage(root_dir_, 'gmd_vulnerability_rel', available_countries, None)

        # fill missing countries in the GMD data with average
        if fill_missing_gmd_with_country_average:
            missing_index = np.setdiff1d(vulnerability_.index.get_level_values('iso3').unique(), vuln_distr.index)

            imputed_countries = missing_index
            update_data_coverage(root_dir_, 'gmd_vulnerability_rel', available_countries, imputed_countries)

            vuln_distr = pd.concat((vuln_distr, pd.DataFrame(index=pd.Index(missing_index, name='iso3'),
                                                             columns=vuln_distr.columns, data=np.nan)), axis=0)
            vuln_distr.fillna(vuln_distr.mean(axis=0), inplace=True)
        vuln_distr.columns.name = 'income_cat'
        vuln_distr = vuln_distr.stack().rename('v_rel').sort_index()

        # compute vulnerability per income quintile as the product of national-level vulnerability and the relative
        # vulnerability distribution
        vulnerability_.update((vulnerability_tot.droplevel('income_cat') * vuln_distr).rename('v'))

        # merge vulnerability with the maximum (fragile or from GEM) and minimum (robust or from GEM) vulnerability, as
        # well as the extreme case of vulnerability; then, clip with the min / max vulnerability
        if vulnerability_bounds:
            if vulnerability_bounds == 'class_extremes':
                vulnerability_ = pd.merge(
                    vulnerability_,
                    building_class_vuln.rename({'fragile': 'v_max', 'robust': 'v_min'}, axis=1)[['v_min', 'v_max']],
                    left_index=True, right_index=True, how='left'
                )
            elif vulnerability_bounds == 'gem_extremes':
                v_gem_minmax = v_gem.unstack('income_cat')
                v_gem_minmax = v_gem_minmax.rename({v_gem_minmax.columns.min(): 'v_max', v_gem_minmax.columns.max(): 'v_min'}, axis=1)[['v_min', 'v_max']]
                vulnerability_ = pd.merge(vulnerability_, v_gem_minmax, left_index=True, right_index=True, how='left')
            else:
                raise ValueError(f"Unknown value for vulnerability_bounds: {vulnerability_bounds}")
            vulnerability_ = vulnerability_['v'].clip(lower=vulnerability_['v_min'], upper=vulnerability_['v_max'])

            # recompute vulnerability_tot due to clipping
            vulnerability_tot = vulnerability_.groupby(['iso3', 'hazard']).mean().to_frame()
            vulnerability_tot['income_cat'] = 'tot'
            vulnerability_tot = vulnerability_tot.set_index('income_cat', append=True).squeeze()
        elif vulnerability_[vulnerability_ > 1].any():
            print(f"Warning. No vulnerability bounds provided. {len(vulnerability_[vulnerability_ > 1])} entries with"
                  f"excess vulnerability values > 1!")
    vulnerability_ = pd.concat((broadcast_to_population_resolution(vulnerability_, resolution), vulnerability_tot), axis=0).sort_index()
    return vulnerability_


def compute_borrowing_ability(root_dir_, any_to_wb_, finance_preparedness_=None, cat_ddo_filepath="CatDDO/catddo.csv",
                              verbose=True, force_recompute=True):
    """
    Processes vulnerability data for different building classes and income categories.

    Args:
        root_dir_ (str): Root directory of the project.
        building_classes (pd.DataFrame): DataFrame containing building class distributions by country and hazard.
        resolution (str): Resolution of the data (e.g., national or subnational).
        use_gmd_to_distribute (bool): Whether to use GMD data to distribute vulnerability. Default is True.
        fill_missing_gmd_with_country_average (bool): Whether to fill missing GMD data with country averages. Default is False.
        vulnerability_bounds (str): Method to bound vulnerability values ('gem_extremes' or 'class_extremes'). Default is 'gem_extremes'.
        building_class_vuln_path (str): Path to the building class vulnerability mapping file.
        gmd_vulnerability_distribution_path (str): Path to the GMD vulnerability distribution file.

    Returns:
        pd.DataFrame: Processed vulnerability data indexed by country, hazard, and income category.
    """
    outpath = os.path.join(root_dir_, "data/processed/borrowing_ability.csv")
    if not force_recompute and os.path.exists(outpath):
        print("Loading borrowing ability data from file...")
        borrowing_ability_ = pd.read_csv(outpath, index_col='iso3')
        return borrowing_ability_
    print("Computing borrowing ability...")
    credit_ratings = load_credit_ratings(root_dir_, any_to_wb_, verbose=verbose)
    borrowing_ability_ = credit_ratings.rename('borrowing_ability')
    if finance_preparedness_ is not None:
        borrowing_ability_ = pd.concat((borrowing_ability_, finance_preparedness_), axis=1).mean(axis=1).rename('borrowing_ability')
    catddo_countries = pd.read_csv(os.path.join(root_dir_, "data/raw/", cat_ddo_filepath), index_col=0, header=None)
    catddo_countries['contingent_countries'] = 1
    catddo_countries = catddo_countries.squeeze()
    catddo_countries.index.name = 'iso3'
    borrowing_ability_ = pd.concat((borrowing_ability_, catddo_countries), axis=1)
    borrowing_ability_ = borrowing_ability_.contingent_countries.fillna(borrowing_ability_.borrowing_ability).rename('borrowing_ability')
    borrowing_ability_.to_csv(outpath)

    update_data_coverage(root_dir_, 'borrowing_ability', borrowing_ability_.index, None)
    return borrowing_ability_


def load_hfa_data(root_dir_):
    """
    Loads Hyogo Framework for Action (HFA) data to assess early warning systems and disaster preparedness.

    Args:
        root_dir_ (str): Root directory of the project.

    Returns:
        pd.DataFrame: HFA data with columns for early warning, preparedness, and financial reserves.
    """
    # HFA (Hyogo Framework for Action) data to assess the role of early warning system
    # 2015 hfa
    hfa15 = pd.read_csv(os.path.join(root_dir_, "data/raw/HFA/HFA_all_2013_2015.csv"), index_col='ISO 3')
    # READ THE LAST HFA DATA
    hfa_newest = pd.read_csv(os.path.join(root_dir_, "data/raw/HFA/HFA_all_2011_2013.csv"), index_col='ISO 3')
    # READ THE PREVIOUS HFA DATA
    hfa_previous = pd.read_csv(os.path.join(root_dir_, "data/raw/HFA/HFA_all_2009_2011.csv"), index_col='ISO 3')
    # most recent values... if no 2011-2013 reporting, we use 2009-2011

    # concat to harmonize indices
    hfa_oldnew = pd.concat([hfa_newest, hfa_previous, hfa15], axis=1, keys=['new', 'old', "15"])
    hfa_data = hfa_oldnew["15"].fillna(hfa_oldnew["new"].fillna(hfa_oldnew["old"]))

    # access to early warning normalized between zero and 1.
    # P2-C3: "Early warning systems are in place for all major hazards, with outreach to communities"
    hfa_data["ew"] = 1 / 5 * hfa_data["P2-C3"]

    # q_s in the report, ability to scale up support to affected population after the disaster
    # normalized between zero and 1
    # P4C2: Do social safety nets exist to increase the resilience of risk prone households and communities?
    # P5C2: Disaster preparedness plans and contingency plans are in place at all administrative levels, and regular
    # training drills and rehearsals are held to test and develop disaster response programs.
    # P4C5: Disaster risk reduction measures are integrated into post‐disaster recovery and rehabilitation processes.
    hfa_data["prepare_scaleup"] = (hfa_data["P4-C2"] + hfa_data["P5-C2"] + hfa_data["P4-C5"]) / 3 / 5

    # P5-C3: "Financial reserves and contingency mechanisms are in place to enable effective response and recovery when
    # required"
    hfa_data["finance_pre"] = (1 + hfa_data["P5-C3"]) / 6

    hfa_data = hfa_data[["ew", "prepare_scaleup", "finance_pre"]]
    hfa_data.fillna(0, inplace=True)
    hfa_data.index.name = 'iso3'

    return hfa_data


def load_wrp_data(any_to_wb_, wrp_data_path_, root_dir_, outfile=None, verbose=True):
    """
    Loads World Risk Poll (WRP) data for disaster preparedness and early warning systems.

    Args:
        any_to_wb_ (dict): Mapping of country names to World Bank ISO3 codes.
        wrp_data_path_ (str): Path to the WRP data file.
        root_dir_ (str): Root directory of the project.
        outfile (str, optional): Path to save the processed data. Default is None.
        verbose (bool): Whether to print verbose output. Default is True.

    Returns:
        pd.DataFrame: Processed WRP data with indicators for disaster preparedness and early warning.
    """

    # previously used HFA indicators for "ability to scale up the support to affected population after the disaster":
    # P4C2: Do social safety nets exist to increase the resilience of risk prone households and communities?
    # P5C2: Disaster preparedness plans and contingency plans are in place at all administrative levels, and regular
    #       training drills and rehearsals are held to test and develop disaster response programs.
    # P4C5: Disaster risk reduction measures are integrated into post‐disaster recovery and rehabilitation processes.
    # disaster preparedness related questions (see lrf_wrp_2021_full_data_dictionary.xlsx):

    # WRP indicators for disaster preparedness
    # Q16A	    Well Prepared to Deal With a Disaster: The National Government
    # Q16B	    Well Prepared to Deal With a Disaster: Hospitals
    # (Q16C)	Well Prepared to Deal With a Disaster: You and Your Family
    # Q16D	    Well Prepared to Deal With a Disaster: Local Government
    cat_preparedness_cols = ['Q16A', 'Q16D']

    # previously used HFA indicators for early warning:
    # P2-C3: "Early warning systems are in place for all major hazards, with outreach to communities"

    # early warning related questions (see lrf_wrp_2021_full_data_dictionary.xlsx):
    # Q19A	Received Warning About Disaster From Internet/Social Media
    # Q19B	Received Warning About Disaster From Local Government or Police
    # Q19C	Received Warning About Disaster From Radio, TV, or Newspapers
    # Q19D	Received Warning About Disaster From Local Community Organization
    early_warning_cols = ['Q19A', 'Q19B', 'Q19C', 'Q19D']

    all_cols = cat_preparedness_cols + early_warning_cols

    # read data
    wrp_data_ = pd.read_csv(str(os.path.join(root_dir_, 'data/raw/', wrp_data_path_)), dtype=object)[['Country', 'WGT', 'Year'] + all_cols]
    wrp_data_ = wrp_data_.replace(' ', np.nan)
    wrp_data_[all_cols + ['WGT']] = wrp_data_[all_cols + ['WGT']].astype(float)
    wrp_data_[['Country', 'Year']] = wrp_data_[['Country', 'Year']].astype(str)
    wrp_data_.set_index(['Country', 'Year'], inplace=True)

    # drop rows with no data
    wrp_data_ = wrp_data_.dropna(how='all', subset=all_cols)

    # 97 = Does Not Apply
    # 98 = Don't Know
    # 99 = Refused
    # consider all of the above as 'no'
    # 1 = Yes
    # 2 = No (set to 0)
    # 3 = It depends (only for Q16A-D, set to 0.5)
    wrp_data_[all_cols] = wrp_data_[all_cols].replace({97: 0, 98: 0, 99: 0, 2: 0, 3: 0.5})

    # consider early warning available, if at least one of the early warning questions was answered with 'yes'
    wrp_data_['ew'] = wrp_data_[early_warning_cols].sum(axis=1, skipna=False)
    wrp_data_.loc[~wrp_data_.ew.isna(), 'ew'] = (wrp_data_.loc[~wrp_data_.ew.isna(), 'ew'] > 0).astype(float)
    early_warning_cols = ['ew']

    indicators = []
    for indicator, cols in [('prepare_scaleup', cat_preparedness_cols), ('ew', early_warning_cols)]:
        data_ = wrp_data_[cols + ['WGT']].dropna(how='all', subset=cols)

        # calculate weighted mean of each sub-indicator. Then take the mean of the sub-indicators to get the indicator
        data_ = data_[cols].mul(data_.WGT, axis=0).groupby(['Country', 'Year']).sum().div(
            data_.WGT.groupby(['Country', 'Year']).sum(), axis=0).mean(axis=1).rename(indicator)
        data_ = data_.to_frame().reset_index()
        data_ = data_.loc[data_.groupby('Country').Year.idxmax()].drop('Year', axis=1).set_index('Country')
        indicators.append(data_)
    indicators = pd.concat(indicators, axis=1)

    indicators = df_to_iso3(indicators.reset_index(), 'Country', any_to_wb_, verbose_=verbose)
    indicators = indicators.set_index('iso3').drop('Country', axis=1)

    if outfile is not None:
        indicators.to_csv(outfile)

    return indicators


def load_credit_ratings(root_dir_, any_to_wb_, tradingecon_ratings_path="credit_ratings/2023-12-13_tradingeconomics_ratings.csv",
                        cia_ratings_raw_path="credit_ratings/2024-01-30_cia_ratings_raw.txt",
                        ratings_scale_path="credit_ratings/credit_ratings_scale.csv",
                        verbose=True):
    """
    Loads and processes credit ratings data from multiple sources.

    Args:
        root_dir_ (str): Root directory of the project.
        any_to_wb_ (dict): Mapping of country names to World Bank ISO3 codes.
        tradingecon_ratings_path (str): Path to the Trading Economics ratings file. Default is "credit_ratings/2023-12-13_tradingeconomics_ratings.csv".
        cia_ratings_raw_path (str): Path to the CIA ratings raw file. Default is "credit_ratings/2024-01-30_cia_ratings_raw.txt".
        ratings_scale_path (str): Path to the credit ratings scale file. Default is "credit_ratings/credit_ratings_scale.csv".
        verbose (bool): Whether to print verbose output. Default is True.

    Returns:
        pd.Series: Processed credit ratings data indexed by ISO3 country codes.
    """
    # Trading Economics ratings
    if verbose:
        print("Warning. Check that [date]_tradingeconomics_ratings.csv is up to date. If not, download it from "
              "http://www.tradingeconomics.com/country-list/rating")
    te_ratings = pd.read_csv(os.path.join(root_dir_, "data/raw/", tradingecon_ratings_path), dtype="str", encoding="utf8", na_values=['NR'])
    te_ratings = te_ratings.dropna(how='all')
    te_ratings = te_ratings[["country", "S&P", "Moody's"]]

    # drop EU
    te_ratings = te_ratings[te_ratings.country != 'European Union']

    # rename Congo to Congo, Dem. Rep.
    te_ratings.replace({'Congo': 'Congo, Dem. Rep.'}, inplace=True)

    # change country name to iso3
    te_ratings = df_to_iso3(te_ratings, "country", any_to_wb_, verbose_=verbose)
    te_ratings = te_ratings.set_index("iso3").drop("country", axis=1)

    # make lower case and strip
    te_ratings = te_ratings.map(lambda x: str.strip(x).lower() if type(x) is str else x)

    # CIA ratings
    if verbose:
        print("Warning. Check that [date]_cia_ratings_raw.csv is up to date. If not, copy the text from "
                "https://www.cia.gov/the-world-factbook/field/credit-ratings/ into [date]_cia_ratings_raw.csv")
    cia_ratings_raw = open(os.path.join(root_dir_, "data/raw/", cia_ratings_raw_path), 'r').read().strip().split('\n')
    cia_ratings = pd.DataFrame(columns=['country', 'agency', 'rating']).set_index(['country', 'agency'])
    current_country = None
    for line in cia_ratings_raw:
        if "rating:" in line:
            agency, rating_year = line.split(" rating: ")
            rating = rating_year.strip().split(" (")[0]
            cia_ratings.loc[(current_country, agency), 'rating'] = rating.strip().lower()
        elif "note:" not in line and len(line) > 0:
            current_country = line
    cia_ratings.reset_index(inplace=True)
    cia_ratings.replace({"Standard & Poors": "S&P", 'n/a': np.nan, 'nr': np.nan}, inplace=True)

    # drop EU
    cia_ratings = cia_ratings[cia_ratings.country != 'European Union']

    # change country name to iso3
    cia_ratings = df_to_iso3(cia_ratings, "country", any_to_wb_, verbose_=verbose)
    cia_ratings = cia_ratings.set_index(["iso3", "agency"]).drop("country", axis=1).squeeze().unstack('agency')

    # merge ratings
    ratings = pd.merge(te_ratings, cia_ratings, left_index=True, right_index=True, how='outer')
    for agency in np.intersect1d(te_ratings.columns, cia_ratings.columns):
        ratings[agency] = ratings[agency + "_x"].fillna(ratings[agency + "_y"])
        ratings.drop([agency + "_x", agency + "_y"], axis=1, inplace=True)

    # Transforms ratings letters into 1-100 numbers
    rating_scale = pd.read_csv(os.path.join(root_dir_, "data/raw/", ratings_scale_path))
    ratings["S&P"].replace(rating_scale["s&p"].values, rating_scale["score"].values, inplace=True)
    ratings["Moody's"].replace(rating_scale["moodys"].values, rating_scale["score"].values, inplace=True)
    ratings["Fitch"].replace(rating_scale["fitch"].values, rating_scale["score"].values, inplace=True)

    # average rating over all agencies
    ratings["rating"] = ratings.mean(axis=1) / 100

    # set ratings to 0 for countries with no rating
    if ratings.rating.isna().any():
        print("No rating available for regions:", "; ".join(ratings[ratings.rating.isna()].index), ". Setting rating to 0.")
        ratings.rating.fillna(0, inplace=True)

    return ratings.rating


def load_disaster_preparedness_data(root_dir_, any_to_wb_, include_hfa_data=True, guess_missing_countries=True,
                                    force_recompute=True, verbose=True, early_warning_file=None):
    """
    Loads and processes disaster preparedness data, including HFA and WRP data.

    Args:
        root_dir_ (str): Root directory of the project.
        any_to_wb_ (dict): Mapping of country names to World Bank ISO3 codes.
        include_hfa_data (bool): Whether to include HFA data. Default is True.
        guess_missing_countries (bool): Whether to guess missing data based on income groups and regions. Default is True.
        force_recompute (bool): Whether to force recomputation of data. Default is True.
        verbose (bool): Whether to print verbose output. Default is True.
        early_warning_file (str): Filepath for alternative early warning data (optional).

    Returns:
        pd.DataFrame: Processed disaster preparedness data indexed by ISO3 country codes.
    """
    dp_path = os.path.join(root_dir_, "data/processed/disaster_preparedness.csv")

    if not force_recompute and os.path.exists(dp_path):
        print("Loading disaster preparedness data from file...")
        disaster_preparedness_ = pd.read_csv(dp_path, index_col='iso3').drop(columns='ew')
        early_warning_ = pd.read_csv(early_warning_file if early_warning_file else dp_path)
        early_warning_ = early_warning_.set_index(list(np.intersect1d(early_warning_.columns, ['iso3', 'hazard']))[::-1]).ew
    else:
        print("Recomputing disaster preparedness data...")
        disaster_preparedness_ = load_wrp_data(any_to_wb_, "WRP/lrf_wrp_2021_full_data.csv.zip", root_dir_,
                                               verbose=verbose)
        if include_hfa_data:
            # read HFA data
            hfa_data = load_hfa_data(root_dir_=root_dir_)
            # merge HFA and WRP data: mean of the two data sets where both are available
            disaster_preparedness_['ew'] = pd.concat((hfa_data['ew'], disaster_preparedness_['ew']), axis=1).mean(axis=1)
            disaster_preparedness_['prepare_scaleup'] = pd.concat(
                (hfa_data['prepare_scaleup'], disaster_preparedness_['prepare_scaleup']), axis=1).mean(axis=1)
            disaster_preparedness_ = pd.merge(disaster_preparedness_, hfa_data.finance_pre, left_index=True, right_index=True, how='outer')

        for v in ['ew', 'prepare_scaleup', 'finance_pre']:
            update_data_coverage(root_dir_, v, disaster_preparedness_.dropna(subset=v).index, None)

        if guess_missing_countries:
            income_groups = load_income_groups(root_dir_)
            if verbose:
                print("Guessing missing countries' disaster preparedness data based on income group and region averages.")
            merged = pd.merge(disaster_preparedness_, income_groups, left_index=True, right_index=True, how='outer')
            merged.dropna(subset=['Region', 'Country income group'], inplace=True)

            # keep track of imputed data
            for v in ['finance_pre', 'ew', 'prepare_scaleup']:
                available_countries = merged[~merged[v].isna()].index.get_level_values('iso3').unique()
                imputed_countries = merged[merged[v].isna()].index.get_level_values('iso3').unique()
                update_data_coverage(root_dir_, v, available_countries, imputed_countries)

            fill_values = merged.groupby(['Region', 'Country income group']).mean()
            fill_values = fill_values.fillna(merged.drop('Region', axis=1).groupby('Country income group').mean())
            merged = merged.fillna(merged.apply(lambda x: fill_values.loc[(x['Region'], x['Country income group'])], axis=1))
            disaster_preparedness_ = merged[disaster_preparedness_.columns]
        disaster_preparedness_ = disaster_preparedness_.dropna()
        disaster_preparedness_.to_csv(dp_path)
        early_warning_ = disaster_preparedness_.pop('ew')
    return disaster_preparedness_, early_warning_


def load_gem_data(
        root_dir_,
        gem_repo_root_dir="data/raw/GEM_vulnerability/global_exposure_model",
        hazus_gem_mapping_path="data/raw/GEM_vulnerability/hazus-gem_mapping.csv",
        gem_fields_path = "./data/raw/GEM_vulnerability/gem_taxonomy_fields.json",
        vulnarebility_class_mapping = "./data/raw/GEM_vulnerability/gem-to-vulnerability_mapping_per_hazard.xlsx",
        verbose=True,
    ):
    """
    Loads GEM (Global Exposure Model) data for vulnerability and building class distributions.

    Args:
        root_dir_ (str): Root directory of the project.
        gem_repo_root_dir (str): Path to the GEM repository. Default is "data/raw/GEM_vulnerability/global_exposure_model".
        hazus_gem_mapping_path (str): Path to the HAZUS-GEM mapping file. Default is "data/raw/GEM_vulnerability/hazus-gem_mapping.csv".
        gem_fields_path (str): Path to the GEM taxonomy fields file. Default is "./data/raw/GEM_vulnerability/gem_taxonomy_fields.json".
        vulnarebility_class_mapping (str): Path to the vulnerability class mapping file. Default is "./data/raw/GEM_vulnerability/gem-to-vulnerability_mapping_per_hazard.xlsx".
        verbose (bool): Whether to print verbose output. Default is True.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Residential share of total replacement cost by country.
            - pd.DataFrame: Building class distributions by country and hazard.
    """
    # load distribution of vulnerability classes per country
    gem_res, gem_building_classes = gather_gem_data(
        gem_repo_root_dir_=os.path.join(root_dir_, gem_repo_root_dir),
        hazus_gem_mapping_path_=os.path.join(root_dir_, hazus_gem_mapping_path),
        gem_fields_path_=os.path.join(root_dir_, gem_fields_path),
        vuln_class_mapping_=os.path.join(root_dir_, vulnarebility_class_mapping),
        vulnerability_class_output_=None,
        weight_by='total_replacement_cost',
        verbose=verbose
    )
    gem_building_classes.columns.names = ['hazard', 'vulnerability_class']
    gem_building_classes = gem_building_classes.droplevel('country', axis=0)
    if 'default' in gem_building_classes.columns:
        gem_building_classes = gem_building_classes.drop('default', axis=1, level=0)

    residential_share = gem_res.groupby(['iso3', 'building_type']).total_replacement_cost.sum().xs('Res', level='building_type') / gem_res.groupby(['iso3']).total_replacement_cost.sum()
    residential_share.rename('residential_share', inplace=True)

    return residential_share, gem_building_classes


def get_average_capital_productivity(root_dir_, force_recompute=True):
    """
    Computes the average productivity of capital using Penn World Table data.

    Args:
        root_dir_ (str): Root directory of the project.
        force_recompute (bool): Whether to force recomputation of data. Default is True.

    Returns:
        pd.Series: Average productivity of capital indexed by ISO3 country codes.
    """

    # Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
    outpath = os.path.join(root_dir_, "data/processed/avg_prod_k.csv")
    if not force_recompute and os.path.exists(outpath):
        print("Loading capital data from file...")
        return pd.read_csv(outpath, index_col='iso3').squeeze()

    print("Recomputing capital data...")
    capital_data = pd.read_excel(os.path.join(root_dir_, "data/raw/PWT_macro_economic_data/pwt1001.xlsx"), sheet_name="Data")
    capital_data = capital_data.rename({'countrycode': 'iso3'}, axis=1)

    # !! NOTE: PWT variable for capital stock has been renamed from 'ck' to 'cn' in the 10.0.0 version
    capital_data = capital_data[['iso3', 'cgdpo', 'cn', 'year']].dropna()

    # retain only the most recent year
    capital_data = capital_data.groupby("iso3").apply(lambda x: x.loc[(x['year']) == np.nanmax(x['year']), :])
    capital_data = capital_data.reset_index(drop=True).set_index('iso3')

    capital_data.drop('year', axis=1, inplace=True)
    capital_data["avg_prod_k"] = capital_data.cgdpo / capital_data.cn
    capital_data = capital_data.dropna()

    capital_data.avg_prod_k.to_csv(outpath)

    update_data_coverage(root_dir_, 'avg_prod_k', capital_data.dropna(subset='avg_prod_k').index, None)

    return capital_data.avg_prod_k


def social_to_tx_and_gsp(cat_info):
    """
    Computes tax rate and social protection income fraction from data by income category.

    Args:
        cat_info (pd.DataFrame): DataFrame containing category information with columns 'social', 'c', and 'n'.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Tax rate (tau_tax) indexed by ISO3 country codes.
            - pd.Series: Social protection income fraction (gamma_SP) indexed by ISO3 country codes.
    """

    # tax is the sum of all transfers paid over the sum of all income
    tx_tax = (cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum()
              / cat_info[["c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum())
    tx_tax.name = 'tau_tax'

    # income from social protection PER PERSON as fraction of total PER CAPITA social protection
    gsp = (cat_info[["diversified_share", "c"]].prod(axis=1, skipna=False)
           / cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum())
    gsp.name = 'gamma_SP'

    return tx_tax, gsp


def prepare_scenario(scenario_params):
    """
    Prepares data for a disaster risk scenario based on specified parameters.

    Args:
        scenario_params (dict): Dictionary containing scenario parameters, including:
            - run_params (dict): Parameters for running the scenario.
            - macro_params (dict): Macro-economic parameters.
            - hazard_params (dict): Hazard-related parameters.
            - policy_params (dict): Policy-related parameters.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Macro-economic data.
            - pd.DataFrame: Data by income category.
            - pd.DataFrame: Hazard ratios data.
            - pd.DataFrame: Hazard protection data.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    if not os.path.exists(os.path.join(root_dir, 'data/processed')):
        os.makedirs(os.path.join(root_dir, 'data/processed'))

    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # unpack scenario parameters
    if scenario_params is None:
        scenario_params = {}

    # Parameter dicts
    run_params = scenario_params.get('run_params', {})
    macro_params = scenario_params.get('macro_params', {})
    hazard_params = scenario_params.get('hazard_params', {})
    policy_params = scenario_params.get('policy_params', {})

    # Set defaults
    run_params['outpath'] = run_params.get('outpath', None)
    run_params['force_recompute'] = run_params.get('force_recompute', False)
    run_params['download'] = run_params.get('download', False)
    run_params['verbose'] = run_params.get('verbose', True)
    run_params['countries'] = run_params.get('countries', 'all')
    run_params['hazards'] = run_params.get('hazards', 'all')
    run_params['ppp_reference_year'] = run_params.get('ppp_reference_year', 2021)
    run_params['include_spl'] = run_params.get('include_spl', False)

    macro_params['income_elasticity_eta'] = macro_params.get('income_elasticity_eta', 1.5)
    macro_params['discount_rate_rho'] = macro_params.get('discount_rate_rho', .06)
    macro_params['axfin_impact'] = macro_params.get('axfin_impact', .1)
    macro_params['reconstruction_capital'] = macro_params.get('reconstruction_capital', 'self_hous')
    macro_params['reduction_vul'] = macro_params.get('reduction_vul', .2)
    macro_params['early_warning_file'] = macro_params.get('early_warning_file', None)

    hazard_params['hazard_protection'] = hazard_params.get('hazard_protection', 'FLOPROS')
    hazard_params['no_exposure_bias'] = hazard_params.get('no_exposure_bias', False)
    hazard_params['fa_threshold'] = hazard_params.get('fa_threshold', .9)

    timestamp = time()
    # read WB data
    wb_data_macro, wb_data_cat_info = get_wb_data(
        root_dir=root_dir,
        ppp_reference_year=run_params['ppp_reference_year'],
        include_remittances=True,
        impute_missing_data=True,
        drop_incomplete=True,
        force_recompute=run_params['force_recompute'],
        download=run_params['download'],
        verbose=run_params['verbose'],
        include_spl=run_params['include_spl'],
        resolution=run_params['resolution'],
    )

    # print duration
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # load liquidity from Findex data
    liquidity_and_axfin = load_findex_liquidity_and_axfin(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        gni_pc_pp=wb_data_macro.gni_pc_pp,
        resolution=run_params['resolution'],
        force_recompute_=run_params['force_recompute'],
        verbose_=run_params['verbose'],
        scale_liquidity_=policy_params.pop('scale_liquidity', None),
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    cat_info = pd.merge(wb_data_cat_info, liquidity_and_axfin, left_index=True, right_index=True, how='inner')

    disaster_preparedness, early_warning = load_disaster_preparedness_data(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        include_hfa_data=True,
        guess_missing_countries=True,
        force_recompute=run_params['force_recompute'],
        verbose=run_params['verbose'],
        early_warning_file=macro_params['early_warning_file'],
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # compute country borrowing ability
    if 'finance_pre' in disaster_preparedness.columns:
        borrowing_ability = compute_borrowing_ability(
            root_dir_=root_dir,
            any_to_wb_=any_to_wb,
            finance_preparedness_=disaster_preparedness.finance_pre,
            verbose=run_params['verbose'],
            force_recompute=run_params['force_recompute'],
        )
        disaster_preparedness = pd.merge(disaster_preparedness, borrowing_ability, left_index=True, right_index=True, how='left')
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # load average productivity of capital
    avg_prod_k = get_average_capital_productivity(
        root_dir_=root_dir,
        force_recompute=run_params['force_recompute']
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # Load capital shares
    capital_shares = calc_asset_shares(
        root_dir_=root_dir,
        any_to_wb_=any_to_wb,
        scale_self_employment=policy_params.pop('scale_self_employment', None),
        verbose=run_params['verbose'],
        force_recompute=run_params['force_recompute'],
        download=run_params['download'],
        guess_missing_countries=True,
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # compute exposure and vulnerability
    hazard_ratios = compute_exposure_and_vulnerability(
        root_dir_=root_dir,
        force_recompute_=run_params['force_recompute'],
        resolution=run_params['resolution'],
        verbose=run_params['verbose'],
        fa_threshold_=hazard_params['fa_threshold'],
        apply_exposure_bias=not hazard_params['no_exposure_bias'],
        population_data=wb_data_macro['pop'],
        scale_exposure=policy_params.pop('scale_exposure', None),
        scale_vulnerability=policy_params.pop('scale_vulnerability', None),
        early_warning_data=early_warning,
        no_ew_hazards="Earthquake+Tsunami",
        reduction_vul=macro_params['reduction_vul'],
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    # get data per income categories
    macro, cat_info, tau_tax = get_cat_info_and_tau_tax(
        cat_info_=cat_info,
        wb_data_macro_=wb_data_macro,
        avg_prod_k_=avg_prod_k,
        resolution=run_params['resolution'],
        axfin_impact_=macro_params['axfin_impact'],
        scale_non_diversified_income_=policy_params.pop('scale_non_diversified_income', None),
        min_diversified_share_=policy_params.pop('min_diversified_share', None),
        scale_income_=policy_params.pop('scale_income', None),
        scale_gini_index_=policy_params.pop('scale_gini_index', None),
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    hazard_protection = load_protection(
        index_=hazard_ratios.index,
        root_dir_=root_dir,
        protection_data=hazard_params['hazard_protection'],
        min_rp=0,
        force_recompute_=run_params['force_recompute'],
    )
    if run_params['force_recompute']:
        print(f"Duration: {time() - timestamp:.2f} seconds.\n")
        timestamp = time()

    hazard_protection = pd.merge(hazard_protection, early_warning, left_index=True, right_index=True, how='outer')

    macro = macro.join(disaster_preparedness, how='left')
    macro = macro.join(avg_prod_k, how='left')
    macro = macro.join(tau_tax, how='left')
    macro = macro.join(capital_shares, how='left')
    macro['rho'] = macro_params['discount_rate_rho']
    macro['income_elasticity_eta'] = macro_params['income_elasticity_eta']

    # clean and harmonize data frames
    macro.dropna(inplace=True)
    cat_info.dropna(inplace=True)
    hazard_ratios.dropna(inplace=True)
    hazard_protection.dropna(inplace=True)

    # retain common (and selected) countries only
    countries = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index and c in hazard_protection.index and c not in run_params.get('exclude_countries', [])]

    # keep track of imputed data
    data_coverage = pd.read_csv(os.path.join(root_dir, "data/processed/data_coverage.csv"), index_col='iso3')
    data_coverage = data_coverage.loc[np.intersect1d(countries, list(data_coverage.index))]
    data_coverage.to_csv(os.path.join(root_dir, "data/processed/data_coverage.csv"))

    if run_params['countries'] != 'all':
        if len(np.intersect1d(countries, run_params['countries'])) == 0:
            print("None of the selected countries found in data. Keeping all countries.")
        else:
            countries = np.intersect1d(countries, run_params['countries'])
            if len(countries) < len(run_params['countries']):
                print("Not all selected countries found in data.")
    macro = macro.loc[countries]
    cat_info = cat_info.loc[countries]
    hazard_ratios = hazard_ratios.loc[countries]
    hazard_protection = hazard_protection.loc[countries]

    # retain selected hazards only
    if run_params['hazards'] != 'all':
        hazards = run_params['hazards']
        hazard_ratios = hazard_ratios[hazard_ratios.index.get_level_values('hazard').isin(hazards)]
        hazard_protection = hazard_protection[hazard_protection.index.get_level_values('hazard').isin(hazards)]

    # Save all data
    print(macro.shape[0], 'countries in analysis')
    if run_params['outpath'] is not None:
        if not os.path.exists(run_params['outpath']):
            os.makedirs(run_params['outpath'])

        data_coverage.to_csv(os.path.join(run_params['outpath'], "data_coverage.csv"))

        # save protection by country and hazard
        hazard_protection.to_csv(
            os.path.join(run_params['outpath'], "scenario__hazard_protection.csv"),
            encoding="utf-8",
            header=True
        )

        # save macro-economic country economic data
        macro.to_csv(
            os.path.join(run_params['outpath'], "scenario__macro.csv"),
            encoding="utf-8",
            header=True
        )

        # save consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        cat_info.to_csv(
            os.path.join(run_params['outpath'], "scenario__cat_info.csv"),
            encoding="utf-8",
            header=True
        )

        # save exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        hazard_ratios.to_csv(
            os.path.join(run_params['outpath'], "scenario__hazard_ratios.csv"),
            encoding="utf-8",
            header=True
        )
    return macro, cat_info, hazard_ratios, hazard_protection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('settings', type=str, help='Path to the settings file')
    args = parser.parse_args()

    scenario_macro_data, scenario_cat_info_data, scenario_hazard_ratios_data, scenario_hazard_protection_data = prepare_scenario(
        scenario_params=yaml.safe_load(open(args.settings, 'r'))['scenario_params']
    )
