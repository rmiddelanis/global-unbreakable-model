import os
import warnings
import numpy as np
import pandas as pd

from lib_gather_data import get_country_name_dicts
from pandas_helper import load_input_data

warnings.filterwarnings("always", category=UserWarning)
root_dir = os.getcwd()

any_to_wb, iso3_to_wb, iso2_iso3 = get_country_name_dicts(root_dir)

# GAR names with SIDS spec
# TODO: SIDS = Small Island Developing States?
#   send email to Bramka regarding this
gar_name_sids = load_input_data(root_dir, "gar_name_sids.csv")
gar_name_sids['wbcountry'] = gar_name_sids.reset_index().country.replace(any_to_wb)
sids_list = gar_name_sids[gar_name_sids.isaSID == "SIDS"].dropna().reset_index().wbcountry

# Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
pwt_data = load_input_data(root_dir, "pwt90.xlsx", sheet_name="Data")
# retain only the most recent year
pwt_data = pwt_data.groupby("country").apply(lambda x: x.loc[(x['year']) == np.max(x['year']), :])
pwt_data = pwt_data.drop("country", axis=1).reset_index().drop("level_1", axis=1)
pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'ck']]
pwt_data["country"] = pwt_data.country.replace(any_to_wb)

gar_capital = load_input_data(root_dir, "GAR_capital.csv")
gar_capital["country"] = gar_capital.country.replace(any_to_wb)
gar_capital.dropna(inplace=True)
gar_sids = gar_capital.set_index("country").loc[np.intersect1d(sids_list.values, gar_capital.country.values), :].replace(0, np.nan).dropna()

# combine capital data from PWT and GAR (SIDS)
all_K = pd.concat([gar_sids, pwt_data.set_index("country")], axis=1)
all_K = all_K.loc[(all_K != 0).any(axis=1), :]

# compute average productivity of capital
all_K["prod_k_1"] = all_K.GDP / all_K.K
all_K["prod_k_2"] = all_K.cgdpo / all_K.ck

all_K["avg_prod_k"] = all_K.prod_k_1
all_K.avg_prod_k = all_K.avg_prod_k.fillna(all_K.prod_k_2)

all_K["Y"] = all_K.cgdpo
all_K.Y = all_K.Y.fillna(all_K.GDP)

all_K["Ktot"] = all_K.ck
all_K.Ktot = all_K.Ktot.fillna(all_K.K)

all_K[["avg_prod_k"]].dropna().to_csv("intermediate/avg_prod_k_with_gar_for_sids.csv")
# all_K.loc[sids_list, "avg_prod_k"].dropna()