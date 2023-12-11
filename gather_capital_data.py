import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("always", category=UserWarning)

# Names to WB names
any_to_wb = pd.read_csv("inputs/any_name_to_wb_name.csv", index_col="any").squeeze()
any_to_wb = any_to_wb[~any_to_wb.index.duplicated(keep='first')]  # drop duplicates

# iso3 to wb country name table
iso3_to_wb = pd.read_csv("inputs/iso3_to_wb_name.csv").set_index("iso3").squeeze()

# iso2 to iso3 table
# the tables has more lines than countries to account for several ways of writing country names
iso2_iso3 = pd.read_csv("inputs/names_to_iso.csv", usecols=["iso2", "iso3"]).drop_duplicates().set_index(
    "iso2").squeeze()

# GAR names with SIDS spec
# TODO: SIDS = Small Island Developing States?
#   send email to Bramka regarding this
gar_name_sids = pd.read_csv("inputs/gar_name_sids.csv")
gar_name_sids['wbcountry'] = gar_name_sids.reset_index().country.replace(any_to_wb)
sids_list = gar_name_sids[gar_name_sids.isaSID == "SIDS"].dropna().reset_index().wbcountry

# Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
pwt_data = pd.read_excel("inputs/pwt90.xlsx", "Data")
# retain only the most recent year
pwt_data = pwt_data.groupby("country").apply(lambda x: x.loc[(x['year']) == np.max(x['year']), :])
pwt_data = pwt_data.drop("country", axis=1).reset_index().drop("level_1", axis=1)
pwt_data = pwt_data[['countrycode', 'country', 'year', 'cgdpo', 'ck']]
pwt_data["country"] = pwt_data.country.replace(any_to_wb)

gar_capital = pd.read_csv("inputs/GAR_capital.csv")
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