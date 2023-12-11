from pandas_datareader import wb
import pandas as pd

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

res = None
for wb_id in wb_ids:
    wb_metadata = wb.search(wb_id, field='id')
    wb_metadata = wb_metadata[wb_metadata.id.str.fullmatch(wb_id, case=False)]
    wb_metadata = wb_metadata.reset_index().drop('index', axis=1)
    if res is None:
        res = wb_metadata
    else:
        if len(wb_metadata) == 0:
            print(f"Could not find dataset with ID '{wb_id}'")
            wb_metadata.loc[0] = [wb_id, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
        res.loc[len(res)] = wb_metadata.values.flatten()
res['link'] = res['id'].apply(lambda x: f"https://data.worldbank.org/indicator/{x}")
res['comment'] = ''
res = res[['name', 'id', 'sourceOrganization', 'source', 'comment', 'link', 'sourceNote', 'topics']].astype(str)
print(res)