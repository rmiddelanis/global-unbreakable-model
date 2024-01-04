# Downloads wb data
# %load_ext autoreload
# %autoreload
# %matplotlib inline
import os

import pandas as pd
from pandas import isnull

from lib_gather_data import get_country_name_dicts, df_to_iso3
from pandas_helper import load_input_data
from wb_api_wrapper import *

include_remitances = True
use_guessed_social = True  # else keeps nans

root_dir = os.getcwd()  # get current directory
any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

# Pandas display optionsa

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 10)
pd.set_option('display.max_rows', 500)


# World Development Indicators
gdp_pc_pp = get_wb_mrv('NY.GDP.PCAP.pp.kd', "gdp_pc_pp")  # Gdp per capita ppp
pop = get_wb_mrv('SP.POP.TOTL', "pop")  # population
# ppp_over_mer = get_wb_mrv('PA.NUS.PPPC.RF',"ppp_over_mer")#conversion factor PPP over MER
gdp_pc_cd = get_wb_mrv('ny.gdp.pcap.cd', "gdp_pc_cd")  # gdp per capita mer
# gap2     =get_wb_mrv('1.0.PGap.2.5usd'  ,"gap2")#poverty gap at 2$
# head2    =get_wb_mrv('SI.POV.2DAY'      ,"head2")# povety count at 2$

share1 = get_wb_mrv('SI.DST.FRST.20', "share1") / 100  # share of income bottom 20%
share2 = get_wb_mrv('SI.DST.02nd.20', "share2") / 100  # share of income second
share3 = get_wb_mrv('SI.DST.03rd.20', "share3") / 100  # share of income 3rd
share4 = get_wb_mrv('SI.DST.04th.20', "share4") / 100  # share of income 4th
share5 = get_wb_mrv('SI.DST.05th.20', "share5") / 100  # share of income 5th

search_wb("coverage.*poor.*all.*ass.*").query("name=='Coverage in poorest quintile (%) - All Social Assistance '")

poor_cov_assistance = mrv(get_wb("per_sa_allsa.cov_q1_tot"))
other_cov_assistance = (mrv(get_wb("per_sa_allsa.cov_q2_tot")) + mrv(get_wb("per_sa_allsa.cov_q3_tot")) + mrv(
    get_wb("per_sa_allsa.cov_q4_tot")) + mrv(get_wb("per_sa_allsa.cov_q5_tot"))) / 4


# Aspire

# Averages
rem1 = get_wb_series('per_pr_allpr.avt_q1_tot', 'rem1')  # Average per capita transfer held by poorest quintile - Private Transfers
rem2 = get_wb_series('per_pr_allpr.avt_q2_tot', 'rem2')  # - Private Transfers
rem3 = get_wb_series('per_pr_allpr.avt_q3_tot', 'rem3')  # - Private Transfers
rem4 = get_wb_series('per_pr_allpr.avt_q4_tot', 'rem4')  # - Private Transfers
rem5 = get_wb_series('per_pr_allpr.avt_q5_tot', 'rem5')  # - Private Transfers

tra1_ = get_wb_series('per_allsp.avt_q1_tot', 'tra1')  # Average per capita transfer held by poorest quintile -All  dolars PPP per day
tra2_ = get_wb_series('per_allsp.avt_q2_tot', 'tra2')  # Average per capita transfer held by -All
tra3_ = get_wb_series('per_allsp.avt_q3_tot', 'tra3')  # Average per capita transfer held by  -All
tra4_ = get_wb_series('per_allsp.avt_q4_tot', 'tra4')  # Average per capita transfer held by  -All
tra5_ = get_wb_series('per_allsp.avt_q5_tot', 'tra5')  # Average per capita transfer held by  -All
# per_pr_allpr.adq_q1_tot


# Adequacies
ade1_remit = get_wb_series('per_pr_allpr.adq_q1_tot', 'ade1_remit') / 100  # Adequacy of benefits for Q1, Remittances
ade2_remit = get_wb_series('per_pr_allpr.adq_q2_tot', 'ade2_remit') / 100  # Adequacy of benefits for Q2, Remittances
ade3_remit = get_wb_series('per_pr_allpr.adq_q3_tot', 'ade3_remit') / 100  # Adequacy of benefits for Q3, Remittances
ade4_remit = get_wb_series('per_pr_allpr.adq_q4_tot', 'ade4_remit') / 100  # Adequacy of benefits for Q4, Remittances
ade5_remit = get_wb_series('per_pr_allpr.adq_q5_tot', 'ade5_remit') / 100  # Adequacy of benefits for Q5, Remittances

ade1_allspl = get_wb_series('per_allsp.adq_q1_tot','ade1_allspl') / 100  # Adequacy of benefits for Q1, All Social Protection and Labor
ade2_allspl = get_wb_series('per_allsp.adq_q2_tot','ade2_allspl') / 100  # Adequacy of benefits for Q2, All Social Protection and Labor
ade3_allspl = get_wb_series('per_allsp.adq_q3_tot','ade3_allspl') / 100  # Adequacy of benefits for Q3, All Social Protection and Labor
ade4_allspl = get_wb_series('per_allsp.adq_q4_tot','ade4_allspl') / 100  # Adequacy of benefits for Q4, All Social Protection and Labor
ade5_allspl = get_wb_series('per_allsp.adq_q5_tot','ade5_allspl') / 100  # Adequacy of benefits for Q5, All Social Protection and Labor

# Coverage
cov1_remit = get_wb_series('per_pr_allpr.cov_q1_tot', 'cov1_remit') / 100  # Coverage for Q1, Remittances
cov2_remit = get_wb_series('per_pr_allpr.cov_q2_tot', 'cov2_remit') / 100  # Coverage for Q2, Remittances
cov3_remit = get_wb_series('per_pr_allpr.cov_q3_tot', 'cov3_remit') / 100  # Coverage for Q3, Remittances
cov4_remit = get_wb_series('per_pr_allpr.cov_q4_tot', 'cov4_remit') / 100  # Coverage for Q4, Remittances
cov5_remit = get_wb_series('per_pr_allpr.cov_q5_tot', 'cov5_remit') / 100  # Coverage for Q5, Remittances

cov1_allspl = get_wb_series('per_allsp.cov_q1_tot','cov1') / 100  # Coverage in poorest quintile (%) -All Social Protection and Labor
cov2_allspl = get_wb_series('per_allsp.cov_q2_tot','cov2') / 100  # Coverage in 2nd quintile (%) -All Social Protection and Labor
cov3_allspl = get_wb_series('per_allsp.cov_q3_tot','cov3') / 100  # Coverage in 3rd quintile (%) -All Social Protection and Labor
cov4_allspl = get_wb_series('per_allsp.cov_q4_tot','cov4') / 100  # Coverage in 4th quintile (%) -All Social Protection and Labor
cov5_allspl = get_wb_series('per_allsp.cov_q5_tot','cov5') / 100  # Coverage in 5th quintile (%) -All Social Protection and Labor

if include_remitances:
    t_1 = mrv(rem1 + tra1_)
    t_2 = mrv(rem2 + tra2_)
    t_3 = mrv(rem3 + tra3_)
    t_4 = mrv(rem4 + tra4_)
    t_5 = mrv(rem5 + tra5_)

    la_1 = mrv((cov1_allspl * ade1_allspl + cov1_remit * ade1_remit))
    la_2 = mrv((cov2_allspl * ade2_allspl + cov2_remit * ade2_remit))
    la_3 = mrv((cov3_allspl * ade3_allspl + cov3_remit * ade3_remit))
    la_4 = mrv((cov4_allspl * ade4_allspl + cov4_remit * ade4_remit))
    la_5 = mrv((cov5_allspl * ade5_allspl + cov5_remit * ade5_remit))

else:
    t_1 = mrv(tra1_)
    t_2 = mrv(tra2_)
    t_3 = mrv(tra3_)
    t_4 = mrv(tra4_)
    t_5 = mrv(tra5_)

    la_1 = mrv((cov1_allspl * ade1_allspl))
    la_2 = mrv((cov2_allspl * ade2_allspl))
    la_3 = mrv((cov3_allspl * ade3_allspl))
    la_4 = mrv((cov4_allspl * ade4_allspl))
    la_5 = mrv((cov5_allspl * ade5_allspl))

y_1 = mrv(rem1 + tra1_) / la_1
y_2 = mrv(rem2 + tra2_) / la_2
y_3 = mrv(rem3 + tra3_) / la_3
y_4 = mrv(rem4 + tra4_) / la_4
y_5 = mrv(rem5 + tra5_) / la_5


search_wb("Saved at a financial institution");

# Findex wave one
# loan40   =get_wb_mrv('WP11651_5.8'      ,"loan40")/100 #Loan in the past year                                  
# loan60   =get_wb_mrv('WP11651_5.9'      ,"loan60")/100 #Loaan in the past year                                  


# Findex wave two
# RM: TODO: THESE DATASETS ARE NOT AVAILABLE IN THE WB API --> using id's below
# loan40 = get_wb_mrv('WP14924_8.8', "loan40") / 100
# loan60 = get_wb_mrv('WP14924_8.9', "loan60") / 100
# saved40 = get_wb_mrv('WP_time_04.8', "saved40") / 100  # Saved at a financial institution in the past year, bottom 40%
# saved60 = get_wb_mrv('WP_time_04.9', "saved60") / 100  # Saved this year, income, top 60% (% age 15+)
saved40 = get_wb_mrv("fin17a.t.d.7", "saved40") / 100  # Saved at a financial institution in the past year, bottom 40%
saved60 = get_wb_mrv('fin17a.t.d.8', "saved60") / 100  # Saved this year, income, top 60% (% age 15+)

search_wb("Urban population ")
urbanization_rate = get_wb_mrv("SP.URB.TOTL.IN.ZS", "urbanization_rate") / 100

df = pd.concat([gdp_pc_pp, pop, share1, urbanization_rate, gdp_pc_cd], axis=1)
df.index.names = ['country']

# We take only savings as an insurance against destitution
df["axfin_p"] = saved40
df["axfin_r"] = saved60

# Comptues share of income from transfers
df["social_p"] = la_1
df["social_r"] = (t_2 + t_3 + t_4 + t_5) / (y_2 + y_3 + y_4 + y_5)


# FROM HERE: FILL MISSING VALUES IN ASPIRE DATA

# GDP per capita from google (GDP per capita plays no role in the indicator. only usefull to plot the data)
# TODO: what about these two countries?!
df.loc["Argentina", "gdp_pc_pp"] = 18600 / 10700 * 10405.
df.loc["Syrian Arab Republic", "gdp_pc_pp"] = 5100 / 10700 * 10405.

# for SIDS, manual addition from online research
manual_additions_sids = load_input_data(root_dir=root_dir, filename='sids_missing_data_manual_input.csv',
                                        index_col='country')
df = df.fillna(manual_additions_sids)

# Social transfer Data from EUsilc (European Union Survey of Income and Living Conditions) and other countries.
# TODO: update SILC data? keep it at all?
# XXX: there is data from ASPIRE in social_ratios. Use fillna instead to update df.
silc_file = load_input_data(root_dir, "social_ratios.csv")

# Change indexes with wold bank names. UK and greece have differnt codes in Europe than ISO2. The first replace is to
# change EL to GR, and change UK to GB. The second one is to change iso2 to iso3, and the third one is to change iso3
# to the wb
silc_file = silc_file.set_index(silc_file.cc.replace({"EL": "GR", "UK": "GB"}).replace(iso2_iso3).replace(iso3_to_wb))

# TODO: Czech Republic not in silc; ignore for now
df.loc[np.intersect1d(df.index, silc_file.index), ["social_p", "social_r"]] = silc_file[["social_p", "social_r"]].loc[
    np.intersect1d(df.index, silc_file.index)]  # Update social transfer from EUsilc.

# shows the country where social_p and social_r are not both NaN.
where = (isnull(df.social_r) & ~isnull(df.social_p)) | (isnull(df.social_p) & ~isnull(df.social_r))
print("social_p and social_r are not both NaN for " + "; ".join(df.loc[where].index))
df.loc[isnull(df.social_r), ['social_p', 'social_r']] = np.nan
df.loc[isnull(df.social_p), ['social_p', 'social_r']] = np.nan

# Guess social transfer
# TODO: @Bramka this appears to be the econometric model data mentioned in paper p. 16/17. Is this model still
#   available? Should we use it? @Stephane
guessed_social = load_input_data(root_dir, "df_social_transfers_statistics.csv", index_col=0)[["social_p_est", "social_r_est"]]
guessed_social.columns = ["social_p", "social_r"]
if use_guessed_social:
    df = df.fillna(guessed_social.clip(lower=0, upper=1))  # replace the NaN with guessed social transfer.

df = df_to_iso3(df.reset_index(), 'country', any_to_wb)
df = df[~df.iso3.isna()].set_index('iso3').drop('country', axis=1)
df.dropna(how="all", inplace=True)
df.to_csv("inputs/WB_socio_economic_data/wb_data.csv", encoding="utf8")
