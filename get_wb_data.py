# Downloads wb data
import argparse
import os
from lib import get_country_name_dicts, df_to_iso3
from wb_api_wrapper import *


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


def get_most_recent_value(df):
    if type(df) is not pd.Series:
        raise ValueError('df is not a series')
    res = df.unstack('income_cat').dropna().stack().reset_index()
    idxmax = res.groupby(['country', 'income_cat'])['year'].idxmax().values
    res = res.loc[idxmax].set_index(['country', 'income_cat']).drop('year', axis=1).squeeze()
    res.name = df.name
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
    macro_path = os.path.join(root_dir, "inputs/processed/wb_data_macro.csv")
    cat_info_path = os.path.join(root_dir, "inputs/processed/wb_data_cat_info.csv")
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
    urbanization_rate = get_wb_mrv("SP.URB.TOTL.IN.ZS", "urbanization_rate") / 100

    # create output data frames
    macro_df_ = pd.concat([gdp_pc_pp, pop, urbanization_rate], axis=1).reset_index()
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
    macro_df_.dropna(how="all", inplace=True)

    cat_info_df_ = df_to_iso3(cat_info_df_.reset_index(), 'country', any_to_wb, verbose_=verbose)
    cat_info_df_ = cat_info_df_[~cat_info_df_.iso3.isna()].set_index(['iso3', 'income_cat']).drop('country', axis=1)
    cat_info_df_.dropna(how="all", inplace=True)

    # legacy additions are not formatted to ISO3; therefore, they cannot be added after the df_to_iso3 function
    if use_additional_data:
        guessed_social = pd.read_csv(os.path.join(root_dir, "inputs/raw/social_share_regression/social_predicted.csv"),
                                         index_col=[0, 1]).squeeze()
        cat_info_df_.social.fillna(guessed_social, inplace=True)

    complete_countries = np.intersect1d(macro_df_.dropna().index.get_level_values('iso3').unique(),
                                        cat_info_df_.dropna().index.get_level_values('iso3').unique())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script parameters')
    parser.add_argument('--exclude_remittances', action='store_true', help='Exclude remittances.')
    parser.add_argument('--no_additional_data', action='store_true', help='Do not use additional data.')
    parser.add_argument('--keep_incomplete', action='store_true', help='Do not drop incomplete data.')

    args = parser.parse_args()

    macro_df, cat_info_df = get_wb_data(
        root_dir=os.getcwd(),
        include_remittances=args.exclude_remittances,
        use_additional_data=not args.no_additional_data,
        drop_incomplete=not args.keep_incomplete,
    )
