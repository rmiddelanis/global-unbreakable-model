# Downloads wb data
import os
import pandas as pd
from pandas_datareader import wb
from misc.helpers import get_country_name_dicts, df_to_iso3
import numpy as np

from datetime import date

YEAR_TODAY = date.today().year


def clean_merge_update(df_, series_other_, any_to_wb):
    if df_.index.names is None:
        raise ValueError('df_ index names are None')
    if series_other_.index.names != df_.index.names:
        series_other_.index.names = df_.index.names
    if type(series_other_) is not pd.Series:
        raise ValueError('series_other_ is not a series')
    if series_other_.name is None:
        raise ValueError('series_other_ has no name')

    df_other_ = series_other_.reset_index()
    df_other_.country = df_other_.country.apply(lambda x: str.lower(x)).replace(any_to_wb)
    series_other_clean = df_other_.set_index(['country', 'income_cat'])[series_other_.name]

    res = pd.merge(df_, series_other_clean, on=series_other_.index.names, how="outer")
    res[series_other_clean.name] = res[series_other_clean.name + '_x'].fillna(res[series_other_clean.name + '_y'])
    res.drop([series_other_clean.name + '_x', series_other_clean.name + '_y'], axis=1, inplace=True)
    return res


def download_cat_info(name, id_q1, id_q2, id_q3, id_q4, id_q5, most_recent_value=True, upper_bound=None,
                      lower_bound=None):
    data_q1 = get_wb_series(id_q1, 'q1')
    data_q2 = get_wb_series(id_q2, 'q2')
    data_q3 = get_wb_series(id_q3, 'q3')
    data_q4 = get_wb_series(id_q4, 'q4')
    data_q5 = get_wb_series(id_q5, 'q5')
    data = pd.concat([data_q1, data_q2, data_q3, data_q4, data_q5], axis=1).dropna().stack().rename(name)
    data.index.names = ['country', 'year', 'income_cat']
    # note: setting upper and lower bounds to nan s.th. the more recent available value is used
    if upper_bound is not None:
        data[data > upper_bound] = np.nan
    if lower_bound is not None:
        data[data < lower_bound] = np.nan
    if most_recent_value:
        data = get_most_recent_value(data)
    return data


def get_wb_data(root_dir, include_remittances=True, use_additional_data=False, drop_incomplete=True,
                force_recompute=True, verbose=True):
    macro_path = os.path.join(root_dir, "data/processed/wb_data_macro.csv")
    cat_info_path = os.path.join(root_dir, "data/processed/wb_data_cat_info.csv")
    if not force_recompute and os.path.exists(macro_path) and os.path.exists(cat_info_path):
        print("Loading World Bank data from file...")
        macro_df_ = pd.read_csv(macro_path, index_col='iso3')
        cat_info_df_ = pd.read_csv(cat_info_path, index_col=['iso3', 'income_cat'])
        return macro_df_, cat_info_df_
    print("Downloading World Bank data...")
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # World Development Indicators
    gdp_pc_pp = get_wb_mrv('NY.GDP.PCAP.pp.kd', "gdp_pc_pp")  # Gdp per capita ppp
    pop = get_wb_mrv('SP.POP.TOTL', "pop")  # population
    # urbanization_rate = get_wb_mrv("SP.URB.TOTL.IN.ZS", "urbanization_rate") / 100

    # create output data frames
    # macro_df_ = pd.concat([gdp_pc_pp, pop, urbanization_rate], axis=1).reset_index()
    macro_df_ = pd.concat([gdp_pc_pp, pop], axis=1).reset_index()
    macro_df_.country = macro_df_.country.replace(any_to_wb)
    macro_df_ = macro_df_.dropna(subset='country').set_index('country')

    cat_info_df_ = pd.DataFrame(index=pd.MultiIndex.from_product((macro_df_.index, ['q1', 'q2', 'q3', 'q4', 'q5'])),
                               columns=['income_share', 'social'])
    cat_info_df_.index.names = ['country', 'income_cat']

    # income shares
    income_shares = download_cat_info(name='income_share', id_q1='SI.DST.FRST.20', id_q2='SI.DST.02nd.20',
                                      id_q3='SI.DST.03rd.20', id_q4='SI.DST.04th.20', id_q5='SI.DST.05th.20',
                                      most_recent_value=True, upper_bound=100, lower_bound=0) / 100
    # make sure income shares add up to 1
    income_shares /= income_shares.unstack('income_cat').sum(axis=1)
    cat_info_df_ = clean_merge_update(cat_info_df_, income_shares, any_to_wb)

    # ASPIRE

    # Adequacies
    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_remittances = download_cat_info(name='adequacy_remittances', id_q1='per_pr_allpr.adq_q1_tot',
                                             id_q2='per_pr_allpr.adq_q2_tot', id_q3='per_pr_allpr.adq_q3_tot',
                                             id_q4='per_pr_allpr.adq_q4_tot', id_q5='per_pr_allpr.adq_q5_tot',
                                             most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Total transfer amount received by all beneficiaries in a population group as a share of the total welfare of
    # beneficiaries in that group
    adequacy_all_prot_lab = download_cat_info(name='adequacy_all_prot_lab', id_q1='per_allsp.adq_q1_tot',
                                              id_q2='per_allsp.adq_q2_tot', id_q3='per_allsp.adq_q3_tot',
                                              id_q4='per_allsp.adq_q4_tot', id_q5='per_allsp.adq_q5_tot',
                                              most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    # Coverage
    coverage_remittances = download_cat_info(name='coverage_remittances', id_q1='per_pr_allpr.cov_q1_tot',
                                             id_q2='per_pr_allpr.cov_q2_tot', id_q3='per_pr_allpr.cov_q3_tot',
                                             id_q4='per_pr_allpr.cov_q4_tot', id_q5='per_pr_allpr.cov_q5_tot',
                                             most_recent_value=False, upper_bound=100, lower_bound=0) / 100

    coverage_all_prot_lab = download_cat_info(name='coverage_all_prot_lab', id_q1='per_allsp.cov_q1_tot',
                                              id_q2='per_allsp.cov_q2_tot', id_q3='per_allsp.cov_q3_tot',
                                              id_q4='per_allsp.cov_q4_tot', id_q5='per_allsp.cov_q5_tot',
                                              most_recent_value=False, upper_bound=100, lower_bound=0) / 100


    if include_remittances:
        # fraction of income that is from transfers
        social = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab +
                                       coverage_remittances * adequacy_remittances).rename('social')
    else:
        # fraction of income that is from transfers
        social = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab).rename('social')
    cat_info_df_ = clean_merge_update(cat_info_df_, social, any_to_wb)

    macro_df_ = df_to_iso3(macro_df_.reset_index(), 'country', any_to_wb, verbose_=verbose)
    macro_df_ = macro_df_[~macro_df_.iso3.isna()].set_index('iso3').drop('country', axis=1)

    cat_info_df_ = df_to_iso3(cat_info_df_.reset_index(), 'country', any_to_wb, verbose_=verbose)
    cat_info_df_ = cat_info_df_[~cat_info_df_.iso3.isna()].set_index(['iso3', 'income_cat']).drop('country', axis=1)

    # legacy additions are not formatted to ISO3; therefore, they cannot be added after the df_to_iso3 function
    if use_additional_data:
        guessed_social = pd.read_csv(os.path.join(root_dir, "data/processed/transfer_shares_predicted.csv"),
                                         index_col=[0, 1]).squeeze()
        cat_info_df_.social.fillna(guessed_social, inplace=True)

    complete_macro = macro_df_.dropna().index.get_level_values('iso3').unique()
    complete_cat_info = cat_info_df_.isna().any(axis=1).replace(True, np.nan).unstack('income_cat').dropna(how='any').index.unique()
    complete_countries = np.intersect1d(complete_macro, complete_cat_info)
    if verbose:
        print(f"Full data for {len(complete_countries)} countries.")
    if drop_incomplete:
        dropped = list(set(list(macro_df_.index.get_level_values('iso3').unique()) +
                           list(cat_info_df_.index.get_level_values('iso3').unique())) - set(complete_countries))
        if verbose:
            print(f"Dropped {len(dropped)} countries with missing data: {dropped}")
        macro_df_ = macro_df_.loc[complete_countries]
        cat_info_df_ = cat_info_df_.loc[complete_countries]

    macro_df_.to_csv(macro_path)
    cat_info_df_.to_csv(cat_info_path)
    return macro_df_, cat_info_df_


def get_wb_series(wb_name, colname='value'):
    """"gets a pandas SERIES (instead of dataframe, for convinience) from wb data with all years and all countries, and a lotof nans"""
    return get_wb_df(wb_name, colname)[colname]


def get_wb_mrv(wb_name, colname):
    """most recent value from WB API"""
    return get_most_recent_value(get_wb_df(wb_name, colname))


def get_most_recent_value(data):
    levels_new = data.index.droplevel('year').names
    res = data.dropna().reset_index()
    return res.loc[res.groupby(levels_new)['year'].idxmax()].drop(columns='year').set_index(levels_new).squeeze()


def get_wb_df(wb_name, colname):
    """gets a dataframe from wb data with all years and all countries, and a lot of nans"""
    # return all values
    wb_raw = (wb.download(indicator=wb_name, start=2000, end=YEAR_TODAY, country="all"))
    # sensible name for the column
    return wb_raw.rename(columns={wb_raw.columns[0]: colname})
