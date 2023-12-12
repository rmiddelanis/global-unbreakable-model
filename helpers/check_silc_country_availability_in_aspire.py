import tqdm

from lib_gather_data import get_country_name_dicts
from pandas_helper import load_input_data
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

any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

silc_countries = load_input_data(root_dir, "social_ratios.csv")
silc_countries = silc_countries[silc_countries.eusilc == 1].cc
silc_countries = silc_countries.replace({"EL": "GR", "UK": "GB"}).replace(iso2_iso3).replace(iso3_to_wb).values

res = pd.DataFrame(index=silc_countries, columns=wb_ids, data=False)
for wb_id in tqdm.tqdm(wb_ids):
    try:
        data = get_wb_mrv(wb_id, '')
        available_countries = np.intersect1d(res.index, data.index)
        res.loc[list(available_countries), wb_id] = True
    except ValueError as e:
        print("Warning. Could not find data for wb_id:", wb_id)
        res.drop(wb_id, axis=1, inplace=True)
print(res)