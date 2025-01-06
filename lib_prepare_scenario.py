import copy
import os
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from lib import get_country_name_dicts, df_to_iso3
from pandas_helper import get_list_of_index_names


def social_to_tx_and_gsp(cat_info):
    """(tx_tax, gamma_SP) from cat_info[["social","c","n"]] """

    # paper equation 4: \tau = (\Sigma_i t_i) / (\Sigma_i \mu k_i)
    # --> tax is the sum of all transfers paid over the sum of all income
    tx_tax = (cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum()
              / cat_info[["c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum())
    tx_tax.name = 'tau_tax'

    # income from social protection PER PERSON as fraction of PER CAPITA social protection
    # paper equation 5: \gamma_i = t_i / (\Sigma_i \mu \tau k_i)
    gsp = (cat_info[["diversified_share", "c"]].prod(axis=1, skipna=False)
           / cat_info[["diversified_share", "c", "n"]].prod(axis=1, skipna=False).groupby(level="iso3").sum())
    gsp.name = 'gamma_SP'

    return tx_tax, gsp


def average_over_rp(df, protection=None):
    """Aggregation of the outputs over return periods"""

    # compute probability of each return period
    return_periods = df.index.get_level_values('rp').unique()
    rp_probabilities = pd.Series(1 / return_periods - np.append(1 / return_periods, 0)[1:], index=return_periods)

    # match return periods and their frequency
    probabilities = pd.Series(data=df.reset_index('rp').rp.replace(rp_probabilities).values, index=df.index,
                              name='probability')

    # removes events below the protection level
    if protection is not None:
        protection_index = pd.merge(probabilities, protection, left_index=True, right_index=True, how='left').protection.values >= probabilities.reset_index('rp').rp.values
        probabilities.loc[protection_index] = 0

    # average weighted by probability
    res = df.mul(probabilities, axis=0).reset_index('rp', drop=True)
    res = res.groupby(level=list(range(res.index.nlevels))).sum()
    if type(df) is pd.Series:
        res.name = df.name
    return res


def gather_capital_data(root_dir_, force_recompute=True):
    # Penn World Table data. Accessible from https://www.rug.nl/ggdc/productivity/pwt/
    outpath = os.path.join(root_dir_, "inputs/processed/capital_data.csv")
    if not force_recompute and os.path.exists(outpath):
        print("Loading capital data from file...")
        return pd.read_csv(outpath, index_col='iso3')

    print("Recomputing capital data...")
    capital_data = pd.read_excel(os.path.join(root_dir_, "inputs/raw/PWT_macro_economic_data/pwt1001.xlsx"), sheet_name="Data")
    capital_data = capital_data.rename({'countrycode': 'iso3'}, axis=1)

    # !! NOTE: PWT variable for capital stock has been renamed from 'ck' to 'cn' in the 10.0.0 version
    capital_data = capital_data[['iso3', 'cgdpo', 'cn', 'year']].dropna()

    # retain only the most recent year
    capital_data = capital_data.groupby("iso3").apply(lambda x: x.loc[(x['year']) == np.nanmax(x['year']), :])
    capital_data = capital_data.reset_index(drop=True).set_index('iso3')

    capital_data.drop('year', axis=1, inplace=True)
    capital_data["avg_prod_k"] = capital_data.cgdpo / capital_data.cn
    capital_data = capital_data.dropna()

    capital_data.to_csv(outpath)

    return capital_data


def agg_to_economy_level(df, seriesname, economy):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T * df["n"]).T.groupby(level=economy).sum()


def interpolate_rps(hazard_ratios, protection_list, default_rp):
    """Extends return periods in hazard_ratios to a finer grid as defined in protection_list, by extrapolating to rp=0.
    hazard_ratios: dataframe with columns of return periods, index with countries and hazards.
    protection_list: list of protection levels that hazard_ratios should be extended to.
    default_rp: function does not do anything if default_rp is already present in hazard_ratios.
    """
    # check input
    if hazard_ratios is None:
        print("hazard_ratios is None, skipping...")
        return None
    if default_rp in hazard_ratios.index:
        print(f"default_rp={default_rp} already in hazard_ratios, skipping...")
        return hazard_ratios
    hazard_ratios_ = hazard_ratios.copy(deep=True)
    protection_list_ = copy.deepcopy(protection_list)

    flag_stack = False
    if "rp" in get_list_of_index_names(hazard_ratios_):
        original_index_names = hazard_ratios_.index.names
        hazard_ratios_ = hazard_ratios_.unstack("rp")
        flag_stack = True

    if type(protection_list_) is pd.DataFrame:
        protection_list_ = protection_list_.squeeze()
    if type(protection_list_) is pd.Series:
        protection_list_ = protection_list_.unique().tolist()

    # in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(hazard_ratios_.columns) is pd.MultiIndex:
        keys = hazard_ratios_.columns.get_level_values(0).unique()
        res = pd.concat(
            {col: interpolate_rps(hazard_ratios_[col], protection_list_, default_rp) for col in keys},
            axis=1
        )
    else:
        # actual function
        # figures out all the return periods to be included
        all_rps = list(set(protection_list_ + hazard_ratios_.columns.tolist()))

        res = hazard_ratios_.copy()

        # extrapolates linearly towards the 0 return period exposure (this creates negative exposure that is tackled after
        # interp.) (mind the 0 rp when computing probabilities)
        if len(res.columns) == 1:
            res[0] = res.squeeze()
        else:
            res[0] = (
                # exposure of smallest return period
                res.iloc[:, 0] -
                # smallest return period * (exposure of second-smallest rp - exposure of smallest rp) /
                res.columns[0] * (res.iloc[:, 1] - res.iloc[:, 0]) /
                # (second-smallest rp - smallest rp)
                (res.columns[1] - res.columns[0])
            )

        # add new, interpolated values for fa_ratios, assuming constant exposure on the right
        x = res.columns.values
        y = res.values
        res = pd.DataFrame(
            data=interp1d(x, y, bounds_error=False)(all_rps),
            index=res.index,
            columns=all_rps
        ).sort_index(axis=1).clip(lower=0).ffill(axis=1)
        res.columns.name = "rp"

    if flag_stack:
        res = res.stack("rp").reset_index().set_index(original_index_names).sort_index()
    return res