from pandas_datareader import wb
import numpy as np
from datetime import date
import pandas as pd

start_year = 2000
today_year = date.today().year


def get_wb_df(wb_name, colname):
    """gets a dataframe from wb data with all years and all countries, and a lot of nans"""
    # return all values
    wb_raw = (wb.download(indicator=wb_name, start=start_year, end=today_year, country="all"))
    # sensible name for the column
    # wb_raw.rename(columns={wb_raw.columns[0]: colname},inplace=True)
    return wb_raw.rename(columns={wb_raw.columns[0]: colname})


def get_wb_series(wb_name, colname='value'):
    """"gets a pandas SERIES (instead of dataframe, for convinience) from wb data with all years and all countries, and a lotof nans"""
    return get_wb_df(wb_name, colname)[colname]


def get_wb_mrv(wb_name, colname):
    """most recent value from WB API"""
    return mrv(get_wb_df(wb_name, colname))


def mrv(data):
    # try:
    #     if data.shape[1] > 1:
    #         data = data.unstack()
    # except IndexError:
    #     pass  # data is already a series;
    #
    # """most recent values from a dataframe. assumes one column is called 'year'"""
    # # removes nans, and takes the most revent value. hop has a horrible shape
    # hop = data.reset_index().dropna().groupby("country").apply(mrv_gp)
    # # reshapes hop as simple dataframe indexed by country
    # hop = hop.reset_index().drop("level_1", axis=1).set_index("country")

    """most recent values from a dataframe. assumes country and year columns or multiindex of these."""
    # reset index and clean data
    res = data
    if isinstance(res.index, pd.MultiIndex):
        res = data.reset_index()
    res = res.dropna()

    # select latest year value
    latest_index = res[["country", "year"]].groupby("country").idxmax().values.flatten()
    res = res.loc[latest_index].drop("year", axis=1)
    res = res.set_index("country")
    return res.squeeze()
