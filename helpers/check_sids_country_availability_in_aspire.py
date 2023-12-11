import tqdm

from wb_api_wrapper import *
import pandas as pd
import os

wb_ids = [
    "NY.GDP.PCAP.pp.kd",
    "SP.POP.TOTL",
    "ny.gdp.pcap.cd",
    "SI.DST.FRST.20",
    "SI.DST.02nd.20",
    "SI.DST.03rd.20",
    "SI.DST.04th.20",
    "SI.DST.05th.20",
    "per_sa_allsa.cov_q1_tot",
    "per_sa_allsa.cov_q2_tot",
    "per_sa_allsa.cov_q3_tot",
    "per_sa_allsa.cov_q4_tot",
    "per_sa_allsa.cov_q5_tot",
    "per_pr_allpr.avt_q1_tot",
    "per_pr_allpr.avt_q2_tot",
    "per_pr_allpr.avt_q3_tot",
    "per_pr_allpr.avt_q4_tot",
    "per_pr_allpr.avt_q5_tot",
    "per_allsp.avt_q1_tot",
    "per_allsp.avt_q2_tot",
    "per_allsp.avt_q3_tot",
    "per_allsp.avt_q4_tot",
    "per_allsp.avt_q5_tot",
    "per_pr_allpr.adq_q1_tot",
    "per_pr_allpr.adq_q2_tot",
    "per_pr_allpr.adq_q3_tot",
    "per_pr_allpr.adq_q4_tot",
    "per_pr_allpr.adq_q5_tot",
    "per_allsp.adq_q1_tot",
    "per_allsp.adq_q2_tot",
    "per_allsp.adq_q3_tot",
    "per_allsp.adq_q4_tot",
    "per_allsp.adq_q5_tot",
    "per_pr_allpr.cov_q1_tot",
    "per_pr_allpr.cov_q2_tot",
    "per_pr_allpr.cov_q3_tot",
    "per_pr_allpr.cov_q4_tot",
    "per_pr_allpr.cov_q5_tot",
    "per_allsp.cov_q1_tot",
    "per_allsp.cov_q2_tot",
    "per_allsp.cov_q3_tot",
    "per_allsp.cov_q4_tot",
    "per_allsp.cov_q5_tot",
    "WP14924_8.8",
    "WP14924_8.9",
    "WP_time_04.8",
    "WP_time_04.9",
    "SP.URB.TOTL.IN.ZS",
]


root_dir = os.getcwd()  # get current directory
input_dir = os.path.join(root_dir, 'inputs')  # get inputs data directory

# Country dictionaries
any_to_wb = pd.read_csv(os.path.join(input_dir, "any_name_to_wb_name.csv"), index_col="any")  # Names to WB names
any_to_wb = any_to_wb[~any_to_wb.index.duplicated(keep='first')]  # drop duplicates

# TODO: keep this? why?
for _c in any_to_wb.index:
    __c = _c.replace(' ', '')
    if __c != _c:
        any_to_wb.loc[__c] = any_to_wb.loc[_c, 'wb_name']

any_to_wb = any_to_wb.squeeze()

# iso3 to wb country name table
iso3_to_wb = pd.read_csv(os.path.join(input_dir, "iso3_to_wb_name.csv")).set_index("iso3").squeeze()
# iso2 to iso3 table
iso2_iso3 = pd.read_csv(os.path.join(input_dir, "names_to_iso.csv"),
                        usecols=["iso2", "iso3"]).drop_duplicates().set_index("iso2").squeeze()

silc_countries = pd.read_csv(os.path.join(input_dir, "social_ratios.csv"))
silc_countries = silc_countries[silc_countries.eusilc == 1].cc
silc_countries = silc_countries.replace({"EL": "GR", "UK": "GB"}).replace(iso2_iso3).replace(iso3_to_wb).values

res = pd.DataFrame(index=silc_countries, columns=wb_ids, data=False)
for wb_id in tqdm.tqdm(wb_ids):
    try:
        data = get_wb_mrv(wb_id, '')
    except ValueError as e:
        print("Warning. Could not find data for wb_id:", wb_id)
    available_countries = np.intersect1d(res.index, data.index)
    res.loc[list(available_countries), wb_id] = True
