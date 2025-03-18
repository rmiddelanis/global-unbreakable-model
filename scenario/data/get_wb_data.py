# Downloads wb data
import os
import pandas as pd
from pandas_datareader import wb
import statsmodels.api as sm
from misc.helpers import get_country_name_dicts, df_to_iso3, load_income_groups, update_data_coverage
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


def guess_missing_transfers_shares(cat_info_df_, root_dir_, any_to_wb, verbose=True):
    regression_spec = {
        'q1': 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + FSY + MNA', # R2=0.557
        'q2': 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + EAP + ECA + MNA',  # R2=0.558
        'q3': 'transfers ~ exp_SP_GDP + unemployment + HICs + UMICs + EAP + ECA + MNA',  # R2=0.503
        'q4': 'transfers ~ exp_SP_GDP + remittances_GDP + HICs + UMICs + EAP + LAC + MNA',  # R2=0.468
        'q5': 'transfers ~ exp_SP_GDP + remittances_GDP + UMICs + EAP',  # R2=0.328
    }

    remittances = get_wb_mrv('BX.TRF.PWKR.DT.GD.ZS', 'remittances_GDP').dropna().astype(float)
    remittances = df_to_iso3(remittances.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3')
    remittances = remittances.set_index('iso3', drop=True).drop('country', axis=1)
    fsy_countries = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/fsy_countries.csv'), header=None)
    fsy_countries = df_to_iso3(fsy_countries, 0, any_to_wb, verbose).iso3.values
    income_groups = load_income_groups(root_dir_)
    # gini_index = df_to_iso3(get_wb_mrv('SI.POV.GINI', 'gini_index').reset_index(), 'country', any_to_wb, verbose)
    # gini_index = gini_index.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()
    unemployment = df_to_iso3(get_wb_mrv('SL.UEM.TOTL.ZS', 'unemployment').reset_index(), 'country', any_to_wb, verbose)
    unemployment = unemployment.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()
    # labforce_part_rate = df_to_iso3(get_wb_mrv('SL.TLF.CACT.ZS', 'labforce_part_rate').reset_index(), 'country', any_to_wb, verbose)
    # labforce_part_rate = labforce_part_rate.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()
    # gvt_effectiveness = df_to_iso3(get_wb_mrv('GE.EST', 'gvt_effectiveness').reset_index(), 'country', any_to_wb, verbose)
    # gvt_effectiveness = gvt_effectiveness.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()
    # self_employment = df_to_iso3(get_wb_mrv('SL.EMP.SELF.ZS', 'self_employment').reset_index(), 'country', any_to_wb, verbose)
    # self_employment = self_employment.dropna(subset='iso3').set_index('iso3').drop('country', axis=1).squeeze()
    ilo_sp_exp = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/ILO_WSPR_SP_exp.csv'),
                             index_col='iso3', na_values=['...', '…']).drop('country', axis=1)
    # ilo_sp_coverage = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/ILO_WSPR_SP_coverage.csv'),
    #                               index_col='iso3', na_values=['...', '…']).drop('country', axis=1)
    # ilo_pension_coverage = pd.read_csv(os.path.join(root_dir_, 'data/raw/social_share_regression/ILO_WSPR_old_age_pension_coverage.csv'),
    #                                     index_col='iso3', na_values=['...', '…']).drop('country', axis=1)
    # x = pd.concat([remittances, ilo_sp_exp, ilo_sp_coverage, ilo_pension_coverage,
    #                pd.get_dummies(income_groups['Region']), pd.get_dummies(income_groups['Country income group']),
    #                macro_df_.gdp_pc_pp / 1000, np.log(macro_df_.gdp_pc_pp).rename('ln_gdp_pc_pp'), gini_index,
    #                unemployment, self_employment, labforce_part_rate, gvt_effectiveness], axis=1)
    x = pd.concat([remittances, ilo_sp_exp,
                   pd.get_dummies(income_groups['Region']), pd.get_dummies(income_groups['Country income group']),
                   unemployment], axis=1)
    x['FSY'] = False
    x.loc[fsy_countries, 'FSY'] = True
    y = cat_info_df_.transfers.unstack('income_cat') * 100
    regression_data = pd.concat([x, y], axis=1).dropna(how='all')

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
    # imputed_countries = pd.Series(index=imputed_countries, data=True).rename('transfers_share')
    available_countries = np.setdiff1d(cat_info_df_.index.get_level_values('iso3').unique(), imputed_countries)
    update_data_coverage(root_dir_, 'transfers', available_countries, imputed_countries)

    cat_info_df_['transfers'] = cat_info_df_['transfers'].fillna(cat_info_df_['transfers_predicted'])
    cat_info_df_.drop('transfers_predicted', axis=1, inplace=True)

    return cat_info_df_


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


def get_wb_data(root_dir, ppp_reference_year=2021, include_remittances=True, impute_missing_data=False, drop_incomplete=True,
                force_recompute=True, verbose=True, save=True, match_macro_years=True):
    macro_path = os.path.join(root_dir, "data/processed/wb_data_macro.csv")
    cat_info_path = os.path.join(root_dir, "data/processed/wb_data_cat_info.csv")
    if not force_recompute and os.path.exists(macro_path) and os.path.exists(cat_info_path):
        print("Loading World Bank data from file...")
        macro_df = pd.read_csv(macro_path, index_col='iso3')
        cat_info_df = pd.read_csv(cat_info_path, index_col=['iso3', 'income_cat'])
        return macro_df, cat_info_df
    print("Downloading World Bank data...")
    any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

    # World Development Indicators
    if ppp_reference_year == 2021:
        # gdp_pc_pp = get_wb_mrv('NY.GDP.PCAP.pp.kd', "gdp_pc_pp")  # Gdp per capita ppp (source: International Comparison Program)
        gdp_pc_pp = get_wb_series('NY.GDP.PCAP.pp.kd', "gdp_pc_pp")  # Gdp per capita ppp (source: International Comparison Program)
        gni_pc_pp = get_wb_series('NY.GNP.PCAP.PP.KD', 'gni_pc_pp')
    elif ppp_reference_year == 2017:
        wb_wdi = pd.read_excel(os.path.join(root_dir, "./data/raw/WB_socio_economic_data/WB-WDI.xlsx"), sheet_name='Data')
        wb_wdi = wb_wdi.rename({'Economy Name': 'country'}, axis=1)
        wb_wdi['country'] = wb_wdi['country'].replace({'Vietnam': 'Viet Nam'})

        gni_pc_pp = wb_wdi[wb_wdi['Indicator ID'] == 'WB.WDI.NY.GNP.PCAP.PP.KD'].set_index('country').iloc[:, 8:]
        gni_pc_pp.columns.name = 'year'
        gni_pc_pp = gni_pc_pp.stack().rename('gni_pc_pp')
        # gni_pc_pp = get_most_recent_value(gni_pc_pp).rename('gni_pc_pp')

        gdp_pc_pp = wb_wdi[wb_wdi['Indicator ID'] == 'WB.WDI.NY.GDP.PCAP.PP.KD'].set_index('country').iloc[:, 8:]
        gdp_pc_pp.columns.name = 'year'
        gdp_pc_pp = gdp_pc_pp.stack().rename('gdp_pc_pp')
        # gdp_pc_pp = get_most_recent_value(gdp_pc_pp).rename('gdp_pc_pp')
    else:
        raise ValueError("PPP reference year not supported")
    gdp_pc_pp = gdp_pc_pp.drop(np.intersect1d(gdp_pc_pp.index.get_level_values('country').unique(), AGG_REGIONS), level='country')
    gdp_pc_pp = df_to_iso3(gdp_pc_pp.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3').set_index(['iso3', 'year']).drop('country', axis=1)

    gni_pc_pp = gni_pc_pp.drop(np.intersect1d(gni_pc_pp.index.get_level_values('country').unique(), AGG_REGIONS), level='country')
    gni_pc_pp = df_to_iso3(gni_pc_pp.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3').set_index(['iso3', 'year']).drop('country', axis=1)

    # pop = get_wb_mrv('SP.POP.TOTL', "pop")  # population (source: World Development Indicators)
    pop = get_wb_series('SP.POP.TOTL', "pop")  # population (source: World Development Indicators)
    pop = pop.drop(np.intersect1d(pop.index.get_level_values('country').unique(), AGG_REGIONS), level='country')
    pop = df_to_iso3(pop.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3').set_index(['iso3', 'year']).drop('country', axis=1)

    # create output data frames
    if match_macro_years:
        macro_df = get_most_recent_value(pd.concat([gdp_pc_pp, gni_pc_pp, pop], axis=1).dropna(), drop_year=False)
        macro_df.rename({'year': 'macro_year'}, axis=1, inplace=True)
    else:
        macro_df = pd.concat([get_most_recent_value(gdp_pc_pp), get_most_recent_value(gni_pc_pp), get_most_recent_value(pop)], axis=1)

    # income shares (source: Poverty and Inequality Platform)
    income_shares = download_cat_info(name='income_share', id_q1='SI.DST.FRST.20', id_q2='SI.DST.02nd.20',
                                      id_q3='SI.DST.03rd.20', id_q4='SI.DST.04th.20', id_q5='SI.DST.05th.20',
                                      # most_recent_value=True, upper_bound=100, lower_bound=0) / 100
                                      most_recent_value=False, upper_bound=100, lower_bound=0) / 100
    # make sure income shares add up to 1
    income_shares /= income_shares.unstack('income_cat').sum(axis=1)
    income_shares = df_to_iso3(income_shares.reset_index(), 'country', any_to_wb, verbose).dropna(subset='iso3').set_index(['iso3', 'year', 'income_cat']).drop('country', axis=1)
    income_shares = get_most_recent_value(income_shares.dropna())
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
        # transfers = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab +
        #                                coverage_remittances * adequacy_remittances).rename('transfers')
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab + coverage_remittances * adequacy_remittances).rename('transfers')
    else:
        # fraction of income that is from transfers
        # transfers = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab).rename('transfers')
        transfers = (coverage_all_prot_lab * adequacy_all_prot_lab).rename('transfers')
    transfers = df_to_iso3(transfers.reset_index(), 'country', any_to_wb, verbose)
    transfers = transfers.dropna(subset='iso3').set_index(['iso3', 'year', 'income_cat']).transfers
    transfers = get_most_recent_value(transfers.dropna())

    cat_info_df = pd.concat([income_shares, transfers], axis=1).sort_index()

    # store data coverage
    for v in macro_df.columns:
        update_data_coverage(root_dir, v, macro_df.dropna(subset=v).index.unique(), None)
    for v in cat_info_df.columns:
        update_data_coverage(root_dir, v, cat_info_df[v].unstack('income_cat').dropna().index.unique(), None)

    if impute_missing_data:
        cat_info_df = guess_missing_transfers_shares(cat_info_df, root_dir, any_to_wb, verbose)

    complete_macro = macro_df.dropna().index.get_level_values('iso3').unique()
    complete_cat_info = cat_info_df.isna().any(axis=1).replace(True, np.nan).unstack('income_cat').dropna(how='any').index.unique()
    complete_countries = np.intersect1d(complete_macro, complete_cat_info)
    if verbose:
        print(f"Full data for {len(complete_countries)} countries.")
    if drop_incomplete:
        dropped = list(set(list(macro_df.index.get_level_values('iso3').unique()) +
                           list(cat_info_df.index.get_level_values('iso3').unique())) - set(complete_countries))
        if verbose:
            print(f"Dropped {len(dropped)} countries with missing data: {dropped}")
        macro_df = macro_df.loc[complete_countries]
        cat_info_df = cat_info_df.loc[complete_countries]

    if save:
        macro_df.to_csv(macro_path)
        cat_info_df.to_csv(cat_info_path)
    return macro_df, cat_info_df


def get_wb_series(wb_name, colname='value'):
    """"gets a pandas SERIES (instead of dataframe, for convinience) from wb data with all years and all countries, and a lotof nans"""
    return get_wb_df(wb_name, colname)[colname]


def get_wb_mrv(wb_name, colname):
    """most recent value from WB API"""
    return get_most_recent_value(get_wb_df(wb_name, colname))


def get_most_recent_value(data, drop_year=True):
    levels_new = data.index.droplevel('year').names
    res = data.dropna().reset_index()
    if drop_year:
        res = res.loc[res.groupby(levels_new)['year'].idxmax()].drop(columns='year').set_index(levels_new).squeeze()
    else:
        res = res.loc[res.groupby(levels_new)['year'].idxmax()].set_index(levels_new).squeeze()
    return res


def get_wb_df(wb_name, colname):
    """gets a dataframe from wb data with all years and all countries, and a lot of nans"""
    # return all values
    wb_raw = (wb.download(indicator=wb_name, start=2000, end=YEAR_TODAY, country="all"))
    # sensible name for the column
    return wb_raw.rename(columns={wb_raw.columns[0]: colname})
