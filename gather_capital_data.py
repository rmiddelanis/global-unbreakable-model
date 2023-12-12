import copy
import os
import numpy as np
import pandas as pd
from lib_gather_data import get_country_name_dicts
from pandas_helper import load_input_data

root_dir = os.getcwd()

any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

# Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
# pwt_data = load_input_data(root_dir, "pwt90.xlsx", sheet_name="Data")
pwt_data = load_input_data(root_dir, "pwt1001.xlsx", sheet_name="Data")
# retain only the most recent year
pwt_data = pwt_data.groupby("country").apply(lambda x: x.loc[(x['year']) == np.max(x['year']), :])
pwt_data = pwt_data.drop("country", axis=1).reset_index().drop("level_1", axis=1)

# !! NOTE: PWT variable for capital stock has been renamed from 'ck' to 'cn' in the 10.0.0 version
# pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'ck']]
pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'cn']].rename({'cn': 'ck'}, axis=1)
pwt_data["country"] = pwt_data.country.replace(any_to_wb)
pwt_data = pwt_data.dropna()
pwt_data.set_index("country", inplace=True)
pwt_data.drop(["countrycode", 'year'], axis=1, inplace=True)

# get capital data for SIDS from GAR
sids_list = load_input_data(root_dir, "gar_name_sids.csv")
sids_list['wbcountry'] = sids_list.reset_index().country.replace(any_to_wb)
sids_list = sids_list[sids_list.isaSID == "SIDS"].dropna().reset_index().wbcountry
sids_capital_gar = load_input_data(root_dir, "GAR_capital.csv")[['country', 'GDP', 'K']]
sids_capital_gar["country"] = sids_capital_gar.country.replace(any_to_wb)
sids_capital_gar.dropna(inplace=True)
sids_capital_gar = sids_capital_gar.set_index("country")
sids_capital_gar = sids_capital_gar.loc[np.intersect1d(sids_list.values, sids_capital_gar.index.values), :]
sids_capital_gar = sids_capital_gar.replace(0, np.nan).dropna()
sids_capital_gar.rename({'K': 'ck', 'GDP': 'cgdpo'}, axis=1, inplace=True)

# merge capital data from PWT and GAR (SIDS)
# compute average productivity of capital
capital_data = pd.merge(pwt_data, sids_capital_gar, on='country', how='outer')
capital_data['cgdpo'] = capital_data.cgdpo_x.fillna(capital_data.cgdpo_y)
capital_data['ck'] = capital_data.ck_x.fillna(capital_data.ck_y)
capital_data.drop(['cgdpo_x', 'cgdpo_y', 'ck_x', 'ck_y'], axis=1, inplace=True)
capital_data["avg_prod_k"] = capital_data.cgdpo / capital_data.ck
capital_data = capital_data.dropna()

capital_data.avg_prod_k.to_csv("intermediate/avg_prod_k_with_gar_for_sids.csv")

# # note: Previous merger was inconsistent; sometimes, the capital data from PWT was kept, sometimes the one from GAR
# # combine capital data from PWT and GAR (SIDS)
# all_K = pd.concat([sids_capital_gar, pwt_data], axis=1)
# all_K = all_K.loc[(all_K != 0).any(axis=1), :]
#
# # compute average productivity of capital
# # note: for avg prod of cap, we prioritize the data from GAR
# all_K["prod_k_1"] = all_K.GDP / all_K.K
# all_K["prod_k_2"] = all_K.cgdpo / all_K.ck
# all_K["avg_prod_k"] = all_K.prod_k_1
# all_K.avg_prod_k = all_K.avg_prod_k.fillna(all_K.prod_k_2)
#
# # note: for GDP and prod cap, we prioritize the data from PWT
# all_K["Y"] = all_K.cgdpo
# all_K.Y = all_K.Y.fillna(all_K.GDP)
# all_K["Ktot"] = all_K.ck
# all_K.Ktot = all_K.Ktot.fillna(all_K.K)
#
# all_K[["avg_prod_k"]].dropna().to_csv("intermediate/avg_prod_k_with_gar_for_sids.csv")
# # all_K.loc[sids_list, "avg_prod_k"].dropna()


# previously, additional data from capital_data.csv was used:

# # Capital data
# # TODO: @Bramka does this data come from Penn World Table (similar to gather_capital_data)?
# #  values don't appear to match PWT data.
# #   --> we stick to the PWT, use this data
# k_data = load_input_data(root_dir, "capital_data.csv", usecols=["code", "cgdpo", "ck"])
#
# # Zair is congo
# k_data = k_data.replace({"ROM": "ROU", "ZAR": "COD"}).rename(columns={"cgdpo": "prod_from_k", "ck": "k"})
#
# # matches names in the dataset with world bank country names
# k_data.set_index("code", inplace=True)
# missing_dict_vals = np.setdiff1d(k_data.index, iso3_to_wb.index)
# if len(missing_dict_vals) > 0:
#     warnings.warn("Countries missing in iso3_to_wb_name.csv: " + " , ".join(missing_dict_vals))
# k_data.index = [iso3_to_wb.loc[i] for i in k_data.index]
# # \mu in the technical paper -- average productivity of capital
# df["avg_prod_k"] = k_data["prod_from_k"] / k_data["k"]

# and then capital_data.csv was updated with the above data from this script:

# sids_k = pd.read_csv("intermediate/avg_prod_k_with_gar_for_sids.csv")
# sids_k = sids_k.rename(columns={"Unnamed: 0": "country"}).set_index("country")
# df = df.fillna(sids_k)
