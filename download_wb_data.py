# Downloads wb data
import os
from lib_gather_data import get_country_name_dicts, df_to_iso3
from pandas_helper import load_input_data
from wb_api_wrapper import *

include_remitances = True
use_guessed_social = True  # else keeps nans
drop_incomplete = True  # drop countries with missing data

root_dir = os.getcwd()  # get current directory
any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)


def clean_merge_update(df_, series_other_, how='outer'):
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

    res = pd.merge(df_, series_other_clean, on=series_other_.index.names, how=how)
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


# World Development Indicators
gdp_pc_pp = get_wb_mrv('NY.GDP.PCAP.pp.kd', "gdp_pc_pp")  # Gdp per capita ppp
pop = get_wb_mrv('SP.POP.TOTL', "pop")  # population
urbanization_rate = get_wb_mrv("SP.URB.TOTL.IN.ZS", "urbanization_rate") / 100

# create output data frames
macro_df = pd.concat([gdp_pc_pp, pop, urbanization_rate], axis=1).reset_index()
macro_df.country = macro_df.country.replace(any_to_wb)
macro_df = macro_df.dropna(subset='country').set_index('country')

cat_info_df = pd.DataFrame(index=pd.MultiIndex.from_product((macro_df.index, ['q1', 'q2', 'q3', 'q4', 'q5'])),
                           columns=['income_share', 'axfin', 'social'])
cat_info_df.index.names = ['country', 'income_cat']

# income shares
income_shares = download_cat_info(name='income_share', id_q1='SI.DST.FRST.20', id_q2='SI.DST.02nd.20',
                                  id_q3='SI.DST.03rd.20', id_q4='SI.DST.04th.20', id_q5='SI.DST.05th.20',
                                  most_recent_value=True, upper_bound=100, lower_bound=0) / 100
cat_info_df = clean_merge_update(cat_info_df, income_shares)


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

if include_remitances:
    # fraction of income that is from transfers
    social = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab +
                                   coverage_remittances * adequacy_remittances).rename('social')
else:
    # fraction of income that is from transfers
    social = get_most_recent_value(coverage_all_prot_lab * adequacy_all_prot_lab).rename('social')
cat_info_df = clean_merge_update(cat_info_df, social)

# financial inclusion
axfin = download_cat_info(name='axfin', id_q1='fin17a.t.d.7', id_q2='fin17a.t.d.7', id_q3='fin17a.t.d.8',
                          id_q4='fin17a.t.d.8', id_q5='fin17a.t.d.8', most_recent_value=True, upper_bound=100,
                          lower_bound=0) / 100
cat_info_df = clean_merge_update(cat_info_df, axfin)


# FROM HERE: FILL MISSING VALUES IN ASPIRE DATA

# GDP per capita from google (GDP per capita plays no role in the indicator. only usefull to plot the data)
# TODO: what about Syria?!
macro_df.loc["Syrian Arab Republic", "gdp_pc_pp"] = 5100 / 10700 * 10405.

# for SIDS, manual addition from online research
# TODO: these data are not fit for five income categories yet; generally, using *_p for q1 and social_r for q2-5
manual_additions_sids = load_input_data(root_dir=root_dir, filename='sids_missing_data_manual_input.csv',
                                        index_col='country')
manual_additions_sids_axfin = pd.concat([
    manual_additions_sids.axfin_p.rename('q1'), manual_additions_sids.axfin_r.rename('q2'),
    manual_additions_sids.axfin_r.rename('q3'), manual_additions_sids.axfin_r.rename('q4'),
    manual_additions_sids.axfin_r.rename('q5'),], axis=1).stack().rename('axfin')
cat_info_df = clean_merge_update(cat_info_df, manual_additions_sids_axfin)

# for SIDS countries, use share1 for q1, and (1-share1)/4 for q2-5
manual_additions_sids_shares = pd.DataFrame(index=manual_additions_sids.index[~manual_additions_sids.share1.isna()],
                                            columns=['q1', 'q2', 'q3', 'q4', 'q5'])
manual_additions_sids_shares.loc[:, 'q1'].fillna(manual_additions_sids.share1, inplace=True)
manual_additions_sids_shares = manual_additions_sids_shares.assign(
    **{c: ((1 - manual_additions_sids_shares.q1) / 4).values for c in ['q2', 'q3', 'q4', 'q5']}
).stack().rename('income_share')
cat_info_df = clean_merge_update(cat_info_df, manual_additions_sids_shares)


# Social transfer Data from EUsilc (European Union Survey of Income and Living Conditions) and other countries.
# TODO: update SILC data? keep it at all?
# XXX: there is data from ASPIRE in social_ratios. Use fillna instead to update df.
silc_file = load_input_data(root_dir, "social_ratios.csv")

# Change indexes with wold bank names. UK and greece have differnt codes in Europe than ISO2. The first replace is to
# change EL to GR, and change UK to GB. The second one is to change iso2 to iso3, and the third one is to change iso3
# to the wb
silc_file = silc_file.set_index(silc_file.cc.replace({"EL": "GR", "UK": "GB"}).replace(iso2_iso3).replace(iso3_to_wb))
silc_social = pd.concat([silc_file.social_p.rename('q1'), silc_file.social_r.rename('q2'),
                         silc_file.social_r.rename('q3'), silc_file.social_r.rename('q4'),
                         silc_file.social_r.rename('q5')], axis=1).stack().rename('social')
cat_info_df = clean_merge_update(cat_info_df, silc_social)


# shows the country where social_p and social_r are not all NaN.
where = cat_info_df.social.unstack('income_cat').dropna(how='all').isna().sum(axis=1).apply(
    lambda x: np.nan if x == 0 else x
).dropna().index.values
cat_info_df.loc[where, 'social'] = np.nan

# Guess social transfer
# TODO: @Bramka this appears to be the econometric model data mentioned in paper p. 16/17. Is this model still
#   available? Should we use it? @Stephane
if use_guessed_social:
    guessed_social = load_input_data(root_dir, "df_social_transfers_statistics.csv", index_col=0)
    guessed_social = guessed_social[["social_p_est", "social_r_est"]].clip(lower=0, upper=1)
    guessed_social = pd.concat([guessed_social.social_p_est.rename('q1'), guessed_social.social_r_est.rename('q2'),
                                guessed_social.social_r_est.rename('q3'), guessed_social.social_r_est.rename('q4'),
                                guessed_social.social_r_est.rename('q5')], axis=1).stack().rename('social')
    cat_info_df.social.fillna(guessed_social, inplace=True)  # replace the NaN with guessed social transfer.

macro_df = df_to_iso3(macro_df.reset_index(), 'country', any_to_wb)
macro_df = macro_df[~macro_df.iso3.isna()].set_index('iso3').drop('country', axis=1)
macro_df.dropna(how="all", inplace=True)

cat_info_df = df_to_iso3(cat_info_df.reset_index(), 'country', any_to_wb)
cat_info_df = cat_info_df[~cat_info_df.iso3.isna()].set_index(['iso3', 'income_cat']).drop('country', axis=1)
cat_info_df.dropna(how="all", inplace=True)

complete_countries = np.intersect1d(macro_df.dropna().index.get_level_values('iso3').unique(),
                                    cat_info_df.dropna().index.get_level_values('iso3').unique())
print(f"Full data for {len(complete_countries)} countries.")
if drop_incomplete:
    dropped = list(set(list(macro_df.index.get_level_values('iso3').unique()) +
                       list(cat_info_df.index.get_level_values('iso3').unique())) - set(complete_countries))
    print(f"Dropped {len(dropped)} countries with missing data: {dropped}")
    macro_df = macro_df.loc[complete_countries]
    cat_info_df = cat_info_df.loc[complete_countries]

macro_df.to_csv("inputs/WB_socio_economic_data/wb_data_macro.csv")
cat_info_df.to_csv("inputs/WB_socio_economic_data/wb_data_cat_info.csv")
