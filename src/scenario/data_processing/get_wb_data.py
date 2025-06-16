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
import pandas as pd
from pandas_datareader import wb
import statsmodels.api as sm
from src.misc.helpers import get_country_name_dicts, df_to_iso3, update_data_coverage, get_world_bank_countries
import numpy as np

from datetime import date

YEAR_TODAY = date.today().year

AGG_REGIONS = ['Africa Eastern and Southern', 'Africa Western and Central', 'Arab World', 'Caribbean small states',
               'Central Europe and the Baltics', 'Early-demographic dividend', 'East Asia & Pacific',
               'East Asia & Pacific (excluding high income)', 'East Asia & Pacific (IDA & IBRD countries)',
               'Euro area', 'Europe & Central Asia', 'Europe & Central Asia (excluding high income)',
               'Europe & Central Asia (IDA & IBRD countries)', 'European Union', 'Fragile and conflict affected situations',
               'Heavily indebted poor countries (HIPC)', 'High income', 'IBRD only', 'IDA & IBRD total', 'IDA blend',
               'IDA only', 'IDA total', 'Late-demographic dividend', 'Latin America & Caribbean',
               'Latin America & Caribbean (excluding high income)', 'Latin America & the Caribbean (IDA & IBRD countries)',
               'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
               'Middle East & North Africa', 'Middle East & North Africa (excluding high income)',
               'Middle East & North Africa (IDA & IBRD countries)', 'Middle income', 'North America', 'Not classified',
               'OECD members', 'Other small states', 'Pacific island small states', 'Post-demographic dividend',
               'Pre-demographic dividend', 'Small states', 'South Asia', 'South Asia (IDA & IBRD)',
               'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa (IDA & IBRD countries)',
               'Upper middle income', 'World']


def get_wb_series(wb_name, colname, wb_raw_data_path, download):
    """
    Retrieves a World Bank series and renames its column.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str or float): Column name for the data_processing.

    Returns:
        pd.DataFrame: DataFrame containing the World Bank series.
    """
    return get_wb_df(wb_name, colname, wb_raw_data_path, download)[colname]


def get_wb_mrv(wb_name, colname, wb_raw_data_path, download):
    """
    Retrieves the most recent value of a World Bank series.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str): Column name for the data_processing.

    Returns:
        pd.Series: Series containing the most recent values of the World Bank series.
    """
    # if year is not in index, it is already the most recent value
    return get_most_recent_value(get_wb_df(wb_name, colname, wb_raw_data_path, download), drop_year=True)


def get_most_recent_value(data, drop_year=True):
    """
    Extracts the most recent value for each group in the data_processing.

    Args:
        data (pd.DataFrame or pd.Series): Input data_processing.
        drop_year (bool): Whether to drop the year column. Defaults to True.

    Returns:
        pd.Series: Data with the most recent values for each group.
    """
    if 'year' in data.index.names:
        levels_new = data.index.droplevel('year').names
        res = data.dropna().reset_index()
        if drop_year:
            res = res.loc[res.groupby(levels_new)['year'].idxmax()].drop(columns='year').set_index(levels_new).squeeze()
        else:
            res = res.loc[res.groupby(levels_new)['year'].idxmax()].set_index(levels_new).squeeze()
        return res
    else:
        print("Warning: No 'year' in index names, returning data_processing as is.")
    return data


def get_wb_df(wb_name, colname, wb_raw_data_path, download):
    """
    Downloads a World Bank dataset and renames its column.

    Args:
        wb_name (str): World Bank indicator name.
        colname (str): Column name for the data_processing.

    Returns:
        pd.DataFrame: DataFrame containing the World Bank dataset.
    """
    wb_raw_path = os.path.join(wb_raw_data_path, f"{wb_name}.csv")
    if download or not os.path.exists(wb_raw_path):
        # return all values
        wb_raw = wb.download(indicator=wb_name, start=2000, end=YEAR_TODAY, country="all")
        wb_raw.to_csv(wb_raw_path)
    else:
        wb_raw = pd.read_csv(wb_raw_path)
        wb_raw = wb_raw.set_index(list(np.intersect1d(wb_raw.columns, ['country', 'year'])))
    # sensible name for the column
    return wb_raw.rename(columns={wb_raw.columns[0]: colname})


def broadcast_to_population_resolution(data, resolution):
    """
    Scales data_processing to a specified population resolution.

    Args:
        data (pd.DataFrame or pd.Series): Input data_processing to be scaled.
        resolution (float): Resolution to scale the data_processing to.

    Returns:
        pd.DataFrame or pd.Series: Data scaled to the specified resolution.

    Raises:
        ValueError: If the input data_processing is not a DataFrame or Series.
    """
    # scale to resolution
    if type(data) == pd.DataFrame:
        return pd.concat([broadcast_to_population_resolution(data[col], resolution) for col in data.columns], axis=1)
    elif type(data) == pd.Series:
        series_name = data.name
        data_ = data.unstack('income_cat').copy()
        data_ = pd.concat([data_] + [data_.rename(columns={c: np.round(c - (i + 1) * resolution, len(str(resolution).split('.')[1])) for c in data_.columns}) for i in range(int(.2 / resolution) - 1)], axis=1)
        data_ = data_.stack().sort_index()
        data_.name = series_name
    else:
        raise ValueError("Data must be a DataFrame or Series")
    return data_


def guess_missing_transfers_shares(cat_info_df_, root_dir_, country_classification_, any_to_wb, wb_raw_data_path,
                                   download, verbose=True, reg_data_outpath=None):
    """
    Predicts missing transfer shares using regression models. Regression specification is hard-coded.

    Args:
        cat_info_df_ (pd.DataFrame): DataFrame containing category information with transfer shares.
        root_dir_ (str): Root directory of the project.
        any_to_wb (dict): Mapping of country names to World Bank ISO3 codes.
        verbose (bool): Whether to print verbose output. Defaults to True.
        reg_data_outpath (str, optional): Path to save regression data_processing. Defaults to None.

    Returns:
        pd.DataFrame: Updated category information with predicted transfer shares.
    """
    regression_spec = {
        .2: 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + FSY + MNA', # R2=0.557
        .4: 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + EAP + ECA + MNA',  # R2=0.558
        .6: 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + EAP + ECA + MNA',  # R2=0.503
        .8: 'transfers ~ exp_SP_GDP + remittances_GDP + HICs + UMICs + EAP + LAC + MNA',  # R2=0.468
        1: 'transfers ~ exp_SP_GDP + remittances_GDP + UMICs + EAP',  # R2=0.328
    }

    remittances = get_wb_mrv('BX.TRF.PWKR.DT.GD.ZS', 'remittances_GDP', wb_raw_data_path, download).dropna().astype(float)
    remittances = df_to_iso3(remittances.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    remittances = remittances.set_index('iso3', drop=True).drop('country', axis=1)

    unemployment = df_to_iso3(get_wb_mrv('SL.UEM.TOTL.ZS', 'unemployment', wb_raw_data_path, download).reset_index(), 'country', any_to_wb, verbose)
    unemployment = unemployment.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()

    fsy_countries = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/fsy_countries.csv'), header=None)
    fsy_countries = df_to_iso3(fsy_countries, 0, any_to_wb, verbose).iso3.values

    ilo_sp_exp = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/ILO_WSPR_SP_exp.csv'),
                             index_col='iso3', na_values=['...', '…']).drop('country', axis=1)
    x = pd.concat([remittances, ilo_sp_exp,
                   pd.get_dummies(country_classification_['region']), pd.get_dummies(country_classification_['income_group']),
                   unemployment], axis=1)
    x['FSY'] = False
    x.loc[fsy_countries, 'FSY'] = True
    y = cat_info_df_.transfers.unstack('income_cat') * 100
    regression_data = pd.concat([x, y], axis=1).dropna(how='all')

    if reg_data_outpath is not None:
        regression_data.to_csv(reg_data_outpath)

    transfers_predicted = None
    for income_cat, spec in regression_spec.items():
        variables = spec.split('~')[1].strip().split(' + ')
        regression_data_ = regression_data[variables + [income_cat]].dropna().copy()
        model = sm.OLS(
            endog=regression_data_[[income_cat]].astype(float),
            exog=sm.add_constant(regression_data_.drop(columns=income_cat)).astype(float)
        ).fit()
        print(f"############ Regression for {income_cat} ############")
        print(model.summary())
        prediction_data_ = regression_data[variables].dropna()
        predicted = model.predict(sm.add_constant(prediction_data_).astype(float)).rename(income_cat)
        if transfers_predicted is None:
            transfers_predicted = predicted
        else:
            transfers_predicted = pd.concat([transfers_predicted, predicted], axis=1)
    transfers_predicted.columns.name = 'income_cat'
    transfers_predicted = transfers_predicted.stack().dropna().sort_index().rename('transfers_predicted')
    transfers_predicted = transfers_predicted.clip(0, 100) / 100

    cat_info_df_ = pd.concat([cat_info_df_, transfers_predicted], axis=1)
    imputed_countries = cat_info_df_[cat_info_df_.transfers.isna() & cat_info_df_.transfers_predicted.notna()].index.get_level_values('iso3').unique()
    available_countries = np.setdiff1d(cat_info_df_.index.get_level_values('iso3').unique(), imputed_countries)
    update_data_coverage(root_dir_, 'transfers', available_countries, imputed_countries)

    cat_info_df_['transfers'] = cat_info_df_['transfers'].fillna(cat_info_df_['transfers_predicted'])
    cat_info_df_.drop('transfers_predicted', axis=1, inplace=True)

    return cat_info_df_


def download_quintile_data(name, id_q1, id_q2, id_q3, id_q4, id_q5, wb_raw_data_path, download, most_recent_value=True,
                           upper_bound=None, lower_bound=None):
    """
    Downloads World Bank quintile data_processing and processes it.

    Args:
        name (str): Name of the data_processing.
        id_q1 (str): World Bank indicator ID for the first quintile.
        id_q2 (str): World Bank indicator ID for the second quintile.
        id_q3 (str): World Bank indicator ID for the third quintile.
        id_q4 (str): World Bank indicator ID for the fourth quintile.
        id_q5 (str): World Bank indicator ID for the fifth quintile.
        most_recent_value (bool): Whether to use the most recent value. Defaults to True.
        upper_bound (float, optional): Upper bound for the data_processing. Defaults to None.
        lower_bound (float, optional): Lower bound for the data_processing. Defaults to None.

    Returns:
        pd.Series: Processed quintile data_processing indexed by country, year, and income category.
    """
    data_q1 = get_wb_series(id_q1, .2, wb_raw_data_path, download)
    data_q2 = get_wb_series(id_q2,.4, wb_raw_data_path, download)
    data_q3 = get_wb_series(id_q3, .6, wb_raw_data_path, download)
    data_q4 = get_wb_series(id_q4, .8, wb_raw_data_path, download)
    data_q5 = get_wb_series(id_q5, 1, wb_raw_data_path, download)
    data = pd.concat([data_q1, data_q2, data_q3, data_q4, data_q5], axis=1).stack().rename(name)
    data.index.names = ['country', 'year', 'income_cat']
    # note: setting upper and lower bounds to nan s.th. the more recent available value is used
    if upper_bound is not None:
        data[data > upper_bound] = np.nan
    if lower_bound is not None:
        data[data < lower_bound] = np.nan
    if most_recent_value:
        data = get_most_recent_value(data)
    return data


def get_wb_data(root_dir, ppp_reference_year=2021, include_remittances=True, impute_missing_data=False,
                drop_incomplete=True, force_recompute=True, verbose=True, save=True, include_spl=False, resolution=.2,
                download=False):
    """
    Downloads and processes World Bank socio-economic data_processing, including macroeconomic and income-level data_processing.

    Args:
        root_dir (str): Root directory of the project.
        ppp_reference_year (int): Reference year for PPP data_processing. Defaults to 2021.
        include_remittances (bool): Whether to include remittance data_processing. Defaults to True.
        impute_missing_data (bool): Whether to impute missing data_processing. Defaults to False.
        drop_incomplete (bool): Whether to drop countries with incomplete data_processing. Defaults to True.
        force_recompute (bool): Whether to force recomputation of data_processing. Defaults to True.
        verbose (bool): Whether to print verbose output. Defaults to True.
        save (bool): Whether to save the processed data_processing. Defaults to True.
        include_spl (bool): Whether to include shared prosperity line data_processing. Defaults to False.
        resolution (float): Resolution for income shares. Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Macroeconomic data_processing indexed by ISO3 country codes.
            - pd.DataFrame: Category-level data_processing indexed by ISO3 country codes and income categories.
    """
    macro_path = os.path.join(root_dir, "data/processed/wb_data_macro.csv")
    cat_info_path = os.path.join(root_dir, "data/processed/wb_data_cat_info.csv")
    rem_ade_path = os.path.join(root_dir, "data/processed/adequacy_remittances.csv")
    transfers_regr_data_path = os.path.join(root_dir, "data/processed/social_shares_regressors.csv")
    wb_raw_data_path = os.path.join(root_dir, "data/raw/WB_socio_economic_data/API")
    if not force_recompute and os.path.exists(macro_path) and os.path.exists(cat_info_path):
        print("Loading World Bank data_processing from file...")
        macro_df = pd.read_csv(macro_path, index_col='iso3')
        cat_info_df = pd.read_csv(cat_info_path, index_col=['iso3', 'income_cat'])
        return macro_df, cat_info_df
    print("Downloading World Bank data_processing...")
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # World Development Indicators
    if ppp_reference_year == 2021:
        gdp_pc_pp = get_wb_series('NY.GDP.PCAP.PP.KD', "gdp_pc_pp", wb_raw_data_path, download)  # Gdp per capita ppp (source: International Comparison Program)
        gni_pc_pp = get_wb_series('NY.GNP.PCAP.PP.KD', 'gni_pc_pp', wb_raw_data_path, download)
    elif ppp_reference_year == 2017:
        wb_wdi = pd.read_excel(os.path.join(root_dir, "./data/raw/WB_socio_economic_data/WB-WDI.xlsx"), sheet_name='Data')
        wb_wdi = wb_wdi.rename({'Economy Name': 'country'}, axis=1)
        wb_wdi['country'] = wb_wdi['country'].replace({'Vietnam': 'Viet Nam'})

        gni_pc_pp = wb_wdi[wb_wdi['Indicator ID'] == 'WB.WDI.NY.GNP.PCAP.PP.KD'].set_index('country').iloc[:, 8:]
        gni_pc_pp.columns.name = 'year'
        gni_pc_pp = gni_pc_pp.stack().rename('gni_pc_pp')

        gdp_pc_pp = wb_wdi[wb_wdi['Indicator ID'] == 'WB.WDI.NY.GDP.PCAP.PP.KD'].set_index('country').iloc[:, 8:]
        gdp_pc_pp.columns.name = 'year'
        gdp_pc_pp = gdp_pc_pp.stack().rename('gdp_pc_pp')
    else:
        raise ValueError("PPP reference year not supported")
    gdp_pc_pp = gdp_pc_pp.drop(np.intersect1d(gdp_pc_pp.index.get_level_values('country').unique(), AGG_REGIONS), level='country' if gdp_pc_pp.index.nlevels > 1 else None)
    gdp_pc_pp = df_to_iso3(gdp_pc_pp.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    gdp_pc_pp = gdp_pc_pp.set_index(list(np.intersect1d(['iso3', 'year'], gdp_pc_pp.columns))).drop('country', axis=1)

    gni_pc_pp = gni_pc_pp.drop(np.intersect1d(gni_pc_pp.index.get_level_values('country').unique(), AGG_REGIONS), level='country' if gni_pc_pp.index.nlevels > 1 else None)
    gni_pc_pp = df_to_iso3(gni_pc_pp.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    gni_pc_pp = gni_pc_pp.set_index(list(np.intersect1d(['iso3', 'year'], gni_pc_pp.columns))).drop('country', axis=1)

    pop = get_wb_series('SP.POP.TOTL', "pop", wb_raw_data_path, download)  # population (source: World Development Indicators)
    pop = pop.drop(np.intersect1d(pop.index.get_level_values('country').unique(), AGG_REGIONS), level='country' if pop.index.nlevels > 1 else None)
    pop = df_to_iso3(pop.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    pop = pop.set_index(list(np.intersect1d(['iso3', 'year'], pop.columns))).drop('country', axis=1)

    gini_index = get_wb_series('SI.POV.GINI', 'gini_index', wb_raw_data_path, download)
    gini_index = gini_index.drop(np.intersect1d(gini_index.index.get_level_values('country').unique(), AGG_REGIONS), level='country' if gini_index.index.nlevels > 1 else None).dropna()
    gini_index = df_to_iso3(gini_index.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    gini_index = gini_index.set_index(list(np.intersect1d(['iso3', 'year'], gini_index.columns))).drop('country', axis=1)

    country_classification = get_world_bank_countries(wb_raw_data_path, download)

    # if include_spl, make sure that years of macro data points match
    if include_spl:
        pip_data = pd.read_csv(os.path.join(root_dir, "data/raw/WB_socio_economic_data/2025-03-07_pip_data.csv"))
        pip_data = pip_data.rename({'country_code': 'iso3', 'reporting_year': 'year'}, axis=1)
        pip_data.year = pip_data.year.astype(str)
        pip_data.set_index(['iso3', 'year', 'reporting_level', 'welfare_type'], inplace=True)
        pip_data = pip_data.xs('national', level='reporting_level')
        spl = pd.merge(
            pip_data.xs('income', level='welfare_type').spl,
            pip_data.xs('consumption', level='welfare_type').spl,
            left_index=True, right_index=True, how='outer'
        )
        spl = spl.spl_x.fillna(spl.spl_y).rename('spl').to_frame()

        macro_df = pd.concat([gdp_pc_pp, gni_pc_pp, pop, spl, country_classification], axis=1).dropna()
        macro_df = get_most_recent_value(macro_df)
    else:
        macro_df = pd.concat([get_most_recent_value(gdp_pc_pp), get_most_recent_value(gni_pc_pp), get_most_recent_value(pop), get_most_recent_value(gini_index), country_classification], axis=1)

    # income shares (source: Poverty and Inequality Platform)
    if resolution == .2:
        income_shares = download_quintile_data(name='income_share', id_q1='SI.DST.FRST.20', id_q2='SI.DST.02nd.20',
                                               id_q3='SI.DST.03rd.20', id_q4='SI.DST.04th.20', id_q5='SI.DST.05th.20',
                                               wb_raw_data_path=wb_raw_data_path, download=download,
                                               most_recent_value=False, upper_bound=100, lower_bound=0) / 100
        # make sure income shares add up to 1
        income_shares /= income_shares.unstack('income_cat').sum(axis=1)
        income_shares = df_to_iso3(income_shares.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3').set_index(['iso3', 'year', 'income_cat']).drop('country', axis=1)
        income_shares = get_most_recent_value(income_shares.dropna())
    elif resolution == .1:
        pip_data = pd.read_csv(os.path.join(root_dir, "data/raw/WB_socio_economic_data/2025-03-07_pip_data.csv"))
        pip_data = pip_data.rename({'country_code': 'iso3', 'reporting_year': 'year'}, axis=1)
        pip_data.year = pip_data.year.astype(str)
        pip_data.set_index(['iso3', 'year', 'reporting_level', 'welfare_type'], inplace=True)
        pip_data = pip_data.xs('national', level='reporting_level')
        pip_data = pip_data[[f'decile{i + 1}' for i in range(10)]].dropna()
        pip_data.columns = np.round(np.linspace(.1, 1, 10), 1)
        pip_data = pip_data.stack()
        pip_data.index.names = ['iso3', 'year', 'welfare_type', 'income_cat']
        income_shares = pd.merge(
            pip_data.xs('income', level='welfare_type').rename('income_welfare'),
            pip_data.xs('consumption', level='welfare_type').rename('consumption_welfare'),
            left_index=True, right_index=True, how='outer'
        )
        income_shares = income_shares.income_welfare.fillna(income_shares.consumption_welfare).rename('income_share').to_frame()
        income_shares = get_most_recent_value(income_shares)
    else:
        raise ValueError(f"Resolution {resolution} not supported")
    # ASPIRE

    # Adequacies
    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_remittances = download_quintile_data(name='adequacy_remittances', id_q1='per_pr_allpr.adq_q1_tot',
                                                  id_q2='per_pr_allpr.adq_q2_tot', id_q3='per_pr_allpr.adq_q3_tot',
                                                  id_q4='per_pr_allpr.adq_q4_tot', id_q5='per_pr_allpr.adq_q5_tot',
                                                  wb_raw_data_path=wb_raw_data_path, download=download,
                                                  most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_all_prot_lab = download_quintile_data(name='adequacy_all_prot_lab', id_q1='per_allsp.adq_q1_tot',
                                                   id_q2='per_allsp.adq_q2_tot', id_q3='per_allsp.adq_q3_tot',
                                                   id_q4='per_allsp.adq_q4_tot', id_q5='per_allsp.adq_q5_tot',
                                                   wb_raw_data_path=wb_raw_data_path, download=download,
                                                   most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Coverage
    coverage_remittances = download_quintile_data(name='coverage_remittances', id_q1='per_pr_allpr.cov_q1_tot',
                                                  id_q2='per_pr_allpr.cov_q2_tot', id_q3='per_pr_allpr.cov_q3_tot',
                                                  id_q4='per_pr_allpr.cov_q4_tot', id_q5='per_pr_allpr.cov_q5_tot',
                                                  wb_raw_data_path=wb_raw_data_path, download=download,
                                                  most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    coverage_all_prot_lab = download_quintile_data(name='coverage_all_prot_lab', id_q1='per_allsp.cov_q1_tot',
                                                   id_q2='per_allsp.cov_q2_tot', id_q3='per_allsp.cov_q3_tot',
                                                   id_q4='per_allsp.cov_q4_tot', id_q5='per_allsp.cov_q5_tot',
                                                   wb_raw_data_path=wb_raw_data_path, download=download,
                                                   most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    if include_remittances:
        # fraction of income that is from transfers
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab + coverage_remittances * adequacy_remittances).rename('transfers')
    else:
        # fraction of income that is from transfers
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab).rename('transfers')
    transfers = df_to_iso3(transfers.reset_index(), 'country', any_to_wb, verbose)
    transfers = transfers.dropna(subset='iso3').set_index(['iso3', 'year', 'income_cat']).transfers
    transfers = get_most_recent_value(transfers.dropna())

    # store data coverage
    update_data_coverage(root_dir, '__purge__', [], None)
    for v in macro_df.columns:
        update_data_coverage(root_dir, v, macro_df.dropna(subset=v).index.unique(), None)
    update_data_coverage(root_dir, 'income_share', income_shares.unstack('income_cat').dropna().index.unique(), None)
    update_data_coverage(root_dir, 'transfers', transfers.unstack('income_cat').dropna().index.unique(), None)

    if impute_missing_data:
        transfers = guess_missing_transfers_shares(transfers.to_frame(), root_dir, country_classification, any_to_wb, wb_raw_data_path, download, verbose, transfers_regr_data_path).squeeze()

    transfers = broadcast_to_population_resolution(transfers, resolution)

    cat_info_df = pd.concat([income_shares, transfers], axis=1).sort_index()

    complete_macro = macro_df.dropna().index.get_level_values('iso3').unique()
    complete_cat_info = cat_info_df.isna().any(axis=1).replace(True, np.nan).unstack('income_cat').dropna(how='any').index.unique()
    complete_countries = np.intersect1d(complete_macro, complete_cat_info)
    if verbose:
        print(f"Full data_processing for {len(complete_countries)} countries.")
    if drop_incomplete:
        dropped = list(set(list(macro_df.index.get_level_values('iso3').unique()) +
                           list(cat_info_df.index.get_level_values('iso3').unique())) - set(complete_countries))
        if verbose:
            print(f"Dropped {len(dropped)} countries with missing data_processing: {dropped}")
        macro_df = macro_df.loc[complete_countries]
        cat_info_df = cat_info_df.loc[complete_countries]

    if save:
        macro_df.to_csv(macro_path)
        cat_info_df.to_csv(cat_info_path)
        pd.concat([adequacy_remittances, adequacy_all_prot_lab, coverage_remittances, coverage_all_prot_lab], axis=1).to_csv(rem_ade_path)
    return macro_df, cat_info_df
