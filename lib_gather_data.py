import pandas as pd
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories


def mystriper(string):
    """strip blanks and converts everythng to lower case"""
    if type(string) == str:
        return str.strip(string).lower()
    else:
        return string


# weighted average
def wavg(data, weights):
    df_matched = pd.DataFrame({"data": data, "weights": weights}).dropna()
    return (df_matched.data * df_matched.weights).sum() / df_matched.weights.sum()


# gets share per agg category from the data in one of the sheets in PAGER_XL
def get_share_from_sheet(PAGER_XL, pager_code_to_aggcat, iso3_to_wb, sheet_name='Rural_Non_Res'):
    data = pd.read_excel(PAGER_XL, sheet_name=sheet_name).set_index("ISO-3digit")  # data as provided in PAGER
    # rename column to aggregate category
    data_agg = data[pager_code_to_aggcat.index].rename(
        columns=pager_code_to_aggcat)  # only pick up the columns that are the indice in paper_code_to_aggcat, and change each name to median, fragile etc. based on pager_code_to_aggcat

    # group by category and sum
    # data_agg = data_agg.groupby(level=0).sum(axis=1)  # sum each category up and shows only three columns with fragile, median and robust.
    data_agg = data_agg.T.groupby(level=0).sum().T

    data_agg = data_agg.set_index(data_agg.reset_index()["ISO-3digit"].replace(iso3_to_wb));

    data_agg.index.name = "country"
    return data_agg[data_agg.index.isin(iso3_to_wb)]  # keeps only countries


def social_to_tx_and_gsp(economy, cat_info):
    """(tx_tax, gamma_SP) from cat_info[["social","c","n"]] """

    # paper equation 4: \tau = (\Sigma_i t_i) / (\Sigma_i \mu k_i)
    # --> tax is the sum of all transfers paid over the sum of all income (excluding transfers ?!)
    # TODO: doesn't the calculation below include transfers in income?!
    tx_tax = cat_info[["social", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum() / \
             cat_info[["c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum()

    # paper equation 5: \gamma_i = t_i / (\Sigma_i \mu \tau k_i)
    gsp = cat_info[["social", "c"]].prod(axis=1, skipna=False) / \
          cat_info[["social", "c", "n"]].prod(axis=1, skipna=False).groupby(level=economy).sum()

    return tx_tax, gsp
