"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model. It is based on
  previous work by Adrien Vogt-Schilb, Jinqiang Chen, Brian Walsh,
  and Jun Rentschler. See https://github.com/walshb1/gRIMM.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

import copy
import numpy as np
import pandas as pd
from unbreakable.misc.helpers import concat_categories, average_over_rp
from unbreakable.model.recovery_optimizer import optimize_data, recompute_data_with_tax


def reshape_input(macro, cat_info, hazard_ratios, event_level):
    """
    Reshapes macroeconomic and category-level data to the event level.

    Args:
        macro (pd.DataFrame): Macroeconomic data.
        cat_info (pd.DataFrame): Category-level data.
        hazard_ratios (pd.DataFrame): Hazard ratios data.
        event_level (list): List of index levels for events.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Reshaped macroeconomic data.
            - pd.DataFrame: Reshaped category-level data.
    """

    # Broadcast macro to event level
    macro_event = pd.merge(macro, hazard_ratios.iloc[:, 0].unstack('income_cat'), left_index=True, right_index=True)[macro.columns]
    macro_event = macro_event.reorder_levels(event_level).sort_index()

    cat_info_event = pd.merge(hazard_ratios[['fa', 'v_ew']], cat_info, left_index=True, right_index=True)
    cat_info_event = cat_info_event.reorder_levels(event_level + ["income_cat"]).sort_index()

    return macro_event, cat_info_event


def compute_dk(macro_event, cat_info_event, event_level, affected_cats):
    """
    Calculates potential damage to capital.

    Args:
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        cat_info_event (pd.DataFrame): Category-level data at the event level.
        event_level (list): List of index levels for events.
        affected_cats (pd.Index): Categories for social protection.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated macroeconomic data with damage calculations.
            - pd.DataFrame: Updated category-level data with damage calculations.
    """
    macro_event_ = copy.deepcopy(macro_event)
    cat_info_event_ = copy.deepcopy(cat_info_event)
    cat_info_event_ia = concat_categories(cat_info_event_, cat_info_event_, index=affected_cats)

    # counts affected and non affected
    n_affected = cat_info_event_["n"] * cat_info_event_.fa
    n_not_affected = cat_info_event_["n"] * (1 - cat_info_event_.fa)
    cat_info_event_ia["n"] = concat_categories(n_affected, n_not_affected, index=affected_cats)

    # capital losses and total capital losses
    cat_info_event_ia["dk"] = cat_info_event_ia[["k", "v_ew"]].prod(axis=1, skipna=False)  # capital potentially be damaged
    cat_info_event_ia.loc[pd.IndexSlice[:, :, :, :, 'na'], "dk"] = 0

    # compute reconstruction capital share
    if 'k_household_share' in macro_event_.columns:
        cat_info_event_ia["dk_reco"] = cat_info_event_ia["dk"] * macro_event["k_household_share"]
    elif 'k_household_share' in cat_info_event_ia.columns:
        cat_info_event_ia["dk_reco"] = cat_info_event_ia[["dk", "k_household_share"]].prod(axis=1)

    # "national" losses
    macro_event_["dk_ctry"] = agg_to_event_level(cat_info_event_ia, "dk", event_level)

    return macro_event_, cat_info_event_ia


def compute_response(macro_event, cat_info_event_ia, event_level, scope, lending_rate=0.05, targeting="data", variant="unif_poor", borrowing_ability="data",
                     covered_loss_share=.2, loss_measure="dk_reco"):
    """
    Calculates the post-disaster response.

    Args:
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        cat_info_event_ia (pd.DataFrame): Category-level data with initial damage assessments.
        event_level (list): List of index levels for events.
        scope (pd.Index): Population scope indices.
        targeting (str): Targeting strategy for social protection.
        lending_rate (float): Lending rate for post-disaster support.
        variant (str): Variant of the post-disaster response.
        borrowing_ability (float): Borrowing ability of affected populations.
        loss_measure (str): Measure of loss to be used.
        covered_loss_share (float): Share of losses covered by social protection.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated macroeconomic data with response calculations.
            - pd.DataFrame: Updated category-level data with response calculations.
    """

    macro_event_ = macro_event.copy()
    macro_event_["fa"] = agg_to_event_level(cat_info_event_ia, "fa", event_level)

    # targeting errors
    if targeting == "perfect":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 0
    elif targeting == "data":
        macro_event_["error_incl"] = ((1 - macro_event_["prepare_scaleup"]) / 2 * macro_event_["fa"] / (1 - macro_event_["fa"])).clip(upper=1, lower=0)  # as in equation 16 of the paper
        macro_event_["error_excl"] = ((1 - macro_event_["prepare_scaleup"]) / 2).clip(upper=1, lower=0)  # as in equation 16 of the paper
    else:
        raise ValueError(f"Unknown targeting option {targeting}")

    # add helped_cat level and apply inclusion and exlusion errors
    # first, n adds to 2 due to duplication of helped_cat
    cat_info_event_iah_ = concat_categories(cat_info_event_ia, cat_info_event_ia, index=pd.Index(["helped", "not_helped"], name="helped_cat"))
    cat_info_event_iah_[["help_needed", "help_received"]] = 0.0

    # apply targeting errors to income categories within the scope of the policy
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, 'a', 'helped'], 'n'] *= (1 - macro_event_["error_excl"])
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, 'a', 'not_helped'], 'n'] *= (macro_event_["error_excl"])
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, 'na', 'helped'], 'n'] *= (macro_event_["error_incl"])
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, 'na', 'not_helped'], 'n'] *= (1 - macro_event_["error_incl"])
    # now, n adds to 1 again

    # income categories outside the scope of the policy are not helped
    not_helped_quintiles = np.setdiff1d(cat_info_event_iah_.index.get_level_values('income_cat').unique(), scope)
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, not_helped_quintiles, :, 'helped'], 'n'] = 0

    # calculate help_needed depending on the PDS variant and scope of the policy
    if variant == "no":
        reference_loss = 0
    elif "uniform" in variant:
        try:
            threshold = float(variant.split(':')[1])
        except ValueError:
            raise ValueError(f"Could not parse threshold value from variant {variant}. For uniform PDS, please provide"
                             f"pds_variant parameter of the form 'uniform:<threshold>' with threshold in [0, 1].")
        min_income_cat = cat_info_event_iah_.index.get_level_values('income_cat').min()
        if threshold < min_income_cat:
            print(f"Warning. pds_variant={variant} was selected, but minimum income category is {min_income_cat}. "
                  f"Setting threshold to {min_income_cat} instead.")
            threshold = min_income_cat
        if threshold > max(scope):
            print(f"Warning. pds_variant={variant} was selected, but maximum covered income category is {max(scope)}. "
                  f"Setting threshold to {max(scope)} instead.")
            threshold = max(scope)
        poor_cat_slice = [ic for ic in cat_info_event_iah_.index.get_level_values('income_cat').unique() if ic <= threshold]
        poor_cat_slice = pd.IndexSlice[:, :, :, poor_cat_slice, "a", :]
        reference_loss = cat_info_event_iah_.loc[poor_cat_slice, [loss_measure, 'n']]
        reference_loss = reference_loss.prod(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() / reference_loss['n'].groupby(['iso3', 'hazard', 'rp']).sum()
    elif variant == "proportional":
        reference_loss = cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, 'a', :], loss_measure]
    else:
        raise ValueError(f"Unknown PDS variant {variant}")
    cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, scope, :, 'helped'], "help_needed"] = covered_loss_share * reference_loss

    # total need (cost) for all helped hh = sum over help_needed for helped hh
    macro_event_["need"] = agg_to_event_level(cat_info_event_iah_, "help_needed", event_level)

    # actual aid is limited by borrowing capacity
    max_aid = lending_rate * macro_event_["gdp_pc_pp"]
    if borrowing_ability == "data":
        # clip by max_aid * borrowing ability
        macro_event_["aid"] = macro_event_["need"].clip(upper=max_aid * macro_event_["borrowing_ability"])
    elif borrowing_ability == "lending_rate":
        # clip by max_aid only
        macro_event_["aid"] = macro_event_["need"].clip(upper=max_aid)
    elif borrowing_ability == "unlimited":
        # no clipping
        macro_event_["aid"] = macro_event_["need"]
    else:
        raise ValueError(f"Unknown borrowing ability option {borrowing_ability}")

    # scale help_needed by the ratio of actual aid to need to get help_received
    cat_info_event_iah_["help_received"] = (macro_event_["aid"] / macro_event_["need"] * cat_info_event_iah_["help_needed"]).fillna(0) # fillna(0) to avoid NaNs in case of zero losses and zero need
    return macro_event_, cat_info_event_iah_


def optimize_recovery(macro_event, cat_info_event_iah, capital_t=50, num_cores=None):
    """
    Prepares data for recovery rate optimization of affected households after a disaster and uses recovery optimizer module to obtain recovery rates.

    Args:
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        cat_info_event_iah (pd.DataFrame): Category-level data with post-disaster response.
        capital_t (int): Time horizon for capital recovery. Defaults to 50.
        num_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to None.

    Returns:
        pd.DataFrame: Updated category-level data with optimized recovery rates.

    Notes:
        - The function merges macroeconomic and category-level data, reformats it for optimization,
          and computes recovery rates using a numerical optimization process.
        - Only households with capital losses are included in the optimization.
        - The recovery rates are merged back into the category-level data.
    """
    opt_data = pd.merge(
        macro_event.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta',
                                    'k_household_share': 'sigma_h'}),
        cat_info_event_iah.rename(columns={'k': 'k_h_eff', 'dk': 'delta_k_h_eff', 'liquidity': 'savings_s_h',
                                           'help_received': 'delta_i_h_pds', 'avg_prod_k': 'productivity_pi',
                                           'k_household_share': 'sigma_h'}),
        left_index=True, right_index=True
    )[['productivity_pi', 'discount_rate_rho', 'eta', 'k_h_eff', 'delta_k_h_eff', 'savings_s_h',
       'sigma_h', 'delta_i_h_pds', 'delta_tax_sp', 'diversified_share']]
    opt_data['capital_t'] = capital_t
    opt_data_filtered = opt_data[(opt_data.delta_k_h_eff > 0) & (opt_data.sigma_h > 0)].round(10)
    recovery_rates_lambda = optimize_data(
        df_in=opt_data_filtered,
        tolerance=1e-2,
        min_lambda=0.,
        max_lambda=1e2,
        num_cores=num_cores
    )
    res = pd.merge(cat_info_event_iah, recovery_rates_lambda, left_index=True, right_index=True, how='left')
    if (opt_data.sigma_h <= 0).any():
        fill_values = res.xs('a', level='affected_cat', drop_level=False).sort_values(by=['iso3', 'region', 'c'] if 'region' in res.columns else ['iso3', 'c'])
        missing_index = fill_values[fill_values.lambda_h.isna()].index
        print(f"Filling missing lambda_h values with linear interpolation for {len(missing_index)} households.")
        fill_index = fill_values.index
        fill_values = (
            fill_values.reset_index()
            .groupby('iso3', group_keys=False)
            .apply(lambda g: g.interpolate(method='nearest').ffill().bfill())
        ).set_index(fill_index.names).loc[fill_index]
        fill_values = fill_values.loc[missing_index].lambda_h.values
        res.loc[missing_index, 'lambda_h'] = fill_values
    return res


def compute_dw_reco_and_used_savings(cat_info_event_iah, macro_event, event_level_, capital_t=50,
                                     num_cores=None):
    """
    Computes well-being losses from reconstruction and savings usage after a disaster using the recovery optimizer module.
    Now, also includes transfers and tax.

    Args:
        cat_info_event_iah (pd.DataFrame): Category-level data with post-disaster response.
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        event_level_ (list): List of index levels for events.
        capital_t (int): Time horizon for capital recovery. Defaults to 50.
        num_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to None.

    Returns:
        pd.DataFrame: Updated category-level data with well-being losses and savings usage.

    Notes:
        - Combines recovery parameters (e.g., capital losses, recovery rates) for affected households.
        - Recomputes data with tax adjustments and integrates recovery parameters.
        - Calculates the amount of savings and social protection used for reconstruction.
    """
    recovery_parameters = cat_info_event_iah.xs('a', level='affected_cat')[['n', 'dk', 'lambda_h']]
    recovery_parameters['dk_pc'] = recovery_parameters['dk'] * recovery_parameters['n']
    recovery_parameters.drop(['n', 'dk'], axis=1, inplace=True)
    recovery_parameters['recovery_params'] = recovery_parameters[['dk_pc', 'lambda_h']].apply(tuple, axis=1)
    recovery_parameters = recovery_parameters.groupby(level=event_level_).agg(list).recovery_params
    recompute_data = pd.merge(cat_info_event_iah, macro_event, left_index=True, right_index=True, how='left')
    recompute_data = recompute_data.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta',
                                                    'k_household_share': 'sigma_h', 'k': 'k_h_eff',
                                                    'dk': 'delta_k_h_eff', 'liquidity': 'savings_s_h',
                                                    'help_received': 'delta_i_h_pds'})
    recompute_data['capital_t'] = capital_t
    recompute_data['social_protection_share_gamma_h'] = recompute_data['gamma_SP']  # * recompute_data['n']
    pov_line_cols = [c for c in recompute_data.columns if 'pov_line' in c]

    recompute_data = recompute_data[['capital_t', 'social_protection_share_gamma_h', 'productivity_pi',
                                     'discount_rate_rho', 'eta', 'k_h_eff', 'delta_k_h_eff',
                                     'savings_s_h', 'sigma_h', 'delta_i_h_pds', 'delta_tax_sp', 'lambda_h',
                                     'diversified_share'] + pov_line_cols]
    recompute_data[pov_line_cols] *= 365  # convert from daily to annual poverty lines

    recompute_data = pd.merge(recompute_data, recovery_parameters, left_index=True, right_index=True, how='left')

    recompute_result = recompute_data_with_tax(recompute_data, num_cores)

    cat_info_event_iah = pd.merge(cat_info_event_iah, recompute_result, left_index=True, right_index=True, how='left')
    cat_info_event_iah = pd.merge(cat_info_event_iah, recovery_parameters, left_index=True, right_index=True, how='left')

    return cat_info_event_iah


def compute_dw_long_term(cat_info_event_iah, macro_event, event_level):#, long_term_horizon_=None):
    """
    Computes long-term well-being losses from public asset reconstruction costs, PDS costs, and used savings.

    Args:
        cat_info_event_iah (pd.DataFrame): Category-level data with post-disaster response.
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        event_level (list): List of index levels for events.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated category-level data with long-term well-being loss calculations.
            - pd.DataFrame: Updated macroeconomic data.

    Notes:
        - Calculates public asset reconstruction costs and distributes them across households.
        - Includes the impact of savings and social protection on long-term consumption losses.
        - Computes long-term well-being losses based on income elasticity and consumption changes.
    """
    if 'k_household_share' in macro_event.columns:
        cat_info_event_iah['dk_pub'] = cat_info_event_iah['dk'] * (1 - macro_event['k_household_share'])
    else:
        cat_info_event_iah['dk_pub'] = cat_info_event_iah['dk'] * (1 - cat_info_event_iah['k_household_share'])
    hh_fee_share = cat_info_event_iah["c"] / agg_to_event_level(cat_info_event_iah, "c", event_level)
    cat_info_event_iah['help_fee'] = hh_fee_share * agg_to_event_level(cat_info_event_iah, 'help_received', event_level)
    cat_info_event_iah['reco_fee'] = hh_fee_share * agg_to_event_level(cat_info_event_iah, 'dk_pub', event_level)
    cat_info_event_iah['dc_long_term'] = (
            cat_info_event_iah['help_fee'] + cat_info_event_iah['reco_fee'] +
            (cat_info_event_iah['dS_reco_PDS'] - cat_info_event_iah['help_received'])
            # 'dS_reco_PDS' is the amount used from both PDS and savings, therefore deducting 'help_received' to get the
            # amount used from savings only. In case 'help_received' is higher than 'dS_reco_PDS', the household
            # has made a net gain, no savings were used and long-term consumption losses decrease.
    )
    cat_info_event_iah['dW_long_term'] = (cat_info_event_iah['c'] ** (-macro_event['income_elasticity_eta']) *
                                          cat_info_event_iah['dc_long_term'])
    return cat_info_event_iah, macro_event


def compute_dw(cat_info_event_iah, macro_event, event_level_, capital_t=50, num_cores=None):
    """
    Calculates dynamic consumption and well-being losses due to disasters.

    Args:
        cat_info_event_iah (pd.DataFrame): Category-level data with post-disaster response.
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        event_level_ (list): List of index levels for events.
        capital_t (int): Time horizon for capital recovery.
        num_cores (int): Number of CPU cores to use for parallel processing.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated category-level data.
            - pd.DataFrame: Updated macroeconomic data.
    """

    # compute the optimal recovery rates
    cat_info_event_iah_ = optimize_recovery(
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        capital_t=capital_t,
        num_cores=num_cores,
    )
    if cat_info_event_iah_.xs('a', level='affected_cat').lambda_h.isna().any():
        failed_optimizations = cat_info_event_iah_[cat_info_event_iah_['lambda_h'].isna()].xs('a', level='affected_cat').index.droplevel(['rp', 'helped_cat', 'income_cat']).unique()
        print(f"Failed to optimize recovery rates for {failed_optimizations}. Dropping {len(failed_optimizations)} entries.")
        cat_info_event_iah_ = cat_info_event_iah_.drop(failed_optimizations)

    # compute the welfare losses from destroyed assets, decreased transfers, and reconstruction
    cat_info_event_iah_ = compute_dw_reco_and_used_savings(cat_info_event_iah_, macro_event, event_level_, capital_t,
                                                           num_cores)

    # compute the long-term welfare losses from public asset reconstruction costs, PDS costs, and used savings
    cat_info_event_iah_, macro_event_ = compute_dw_long_term(cat_info_event_iah_, macro_event, event_level_)#, long_term_horizon_)

    # sum the welfare losses from reconstruction and long-term welfare losses
    cat_info_event_iah_['dc'] = cat_info_event_iah_['dc_long_term'] + cat_info_event_iah_['dc_short_term']
    cat_info_event_iah_['dw'] = cat_info_event_iah_['dW_reco'] + cat_info_event_iah_['dW_long_term']
    return cat_info_event_iah_, macro_event_


def prepare_output(macro, macro_event, cat_info_event_iah, event_level, hazard_protection_,
                   is_local_welfare=True):
    """
    Aggregates data to the event level and prepares results.

    Args:
        macro (pd.DataFrame): Original macroeconomic data.
        macro_event (pd.DataFrame): Macroeconomic data at the event level.
        cat_info_event_iah (pd.DataFrame): Category-level data with welfare loss calculations.
        event_level (list): List of index levels for events.
        hazard_protection_ (pd.DataFrame): Hazard protection data.
        is_local_welfare (bool): Whether to include local welfare calculations.

    Returns:
        pd.DataFrame: Final results aggregated to the event level.
    """

    # generate output df
    out = pd.DataFrame(index=macro_event.index)

    # pull 'aid' and 'dk' from macro_event
    out["average_aid_cost_pc"] = macro_event["aid"] if 'aid' in macro_event else 0
    out["dk"] = macro_event["dk_ctry"]

    # aggregate delta_W at event-level
    out["dw"] = agg_to_event_level(cat_info_event_iah, "dw", event_level)
    out["dc"] = agg_to_event_level(cat_info_event_iah, "dc", event_level)

    out["dk_tot"] = out["dk"] * macro_event["pop"]
    out["dc_tot"] = out["dc"] * macro_event["pop"]
    out["dw_tot"] = out["dw"] * macro_event["pop"]

    # aggregate losses
    # Averages over return periods to get dk_{hazard} and dW_{hazard}
    if out.index.get_level_values('rp').nunique() > 1:
        out = average_over_rp(out, hazard_protection_)
    else:
        out = out.reset_index('rp', drop=True)

    # Sums over hazard dk, dW (gets one line per economy)
    out = out.groupby(level="iso3").sum()

    # adds dk and dw-like columns to macro
    out = pd.concat((macro, out), axis=1)

    # computes socio-economic capacity and risk at economy level
    out = calc_risk_and_resilience_from_k_w(df=out, is_local_welfare=is_local_welfare)#, long_term_horizon_=long_term_horizon_)

    return out


def agg_to_event_level(df, seriesname, event_level):
    """
    Aggregates a specified series in a DataFrame to the event level using weights.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be aggregated.
        seriesname (str or list of str): Name(s) of the series/column(s) to aggregate.
        event_level (list): List of index levels defining the event level (e.g., country, hazard, return period).

    Returns:
        pd.Series: Aggregated series at the event level.

    Notes:
        - The aggregation is weighted by the population share (column 'n') in the DataFrame.
        - Weights are not normalized to sum to 1.
    """
    return (df[seriesname].T * df["n"]).T.groupby(level=event_level).sum()


def calc_risk_and_resilience_from_k_w(df, is_local_welfare=True):#, long_term_horizon_=None):
    """
    Computes risk and resilience from welfare losses and capital losses.

    Args:
        df (pd.DataFrame): DataFrame containing macroeconomic and welfare loss data.
        is_local_welfare (bool): Whether to use local welfare. Defaults to True.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated risk and resilience metrics.

    Notes:
        - Expresses welfare losses in monetary terms using the derivative of welfare with respect to
          the net present value of future consumption.
        - Calculates expected welfare loss per household and total.
        - Computes risk to welfare as a percentage of local GDP.
        - Derives socio-economic resilience and risk to assets.
    """
    df = df.copy()

    # derivative of welfare with respect to NPV of future consumption
    if is_local_welfare:
        w_prime = df["gdp_pc_pp"]**(-df["income_elasticity_eta"])
    else:
        w_prime = df["gdp_pc_pp_nat"]**(-df["income_elasticity_eta"])

    d_w_ref = w_prime * df["dk"]

    # expected welfare loss (per household and total)
    df["dWpc_currency"] = df["dw"] / w_prime
    df["dWtot_currency"] = df["dWpc_currency"] * df["pop"]

    # Risk to welfare as percentage of local GDP
    df["risk_to_wellbeing"] = df["dWpc_currency"] / df["gdp_pc_pp"]
    df["risk_to_consumption"] = df["dc"] / df["gdp_pc_pp"]

    # socio-economic resilience
    df["resilience"] = d_w_ref / df["dw"]

    # risk to assets
    df["risk_to_assets"] = df.resilience * df.risk_to_wellbeing

    return df
