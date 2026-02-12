"""
  Copyright (c) 2023-2025 Robin Middelanis <rmiddelanis@worldbank.org>

  This file is part of the global Unbreakable model.

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


import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import tqdm
from scipy import integrate, optimize
from scipy.optimize import root_scalar


def delta_capital_k_eff_of_t(t_, recovery_params_):
    """
    Compute the dynamic effective capital loss at the national level.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).

    Returns:
        float or np.ndarray: Effective capital loss at the national level.
    """
    dk_national = 0
    for delta_k_h_eff_, lambda_h_ in recovery_params_:
        dk_h = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)
        if dk_national is None:
            dk_national = dk_h
        else:
            dk_national += dk_h
    return dk_national


def delta_k_h_eff_of_t(t_, delta_k_h_eff_, lambda_h_):
    """
    Compute the dynamic effective capital loss in the exponential recovery regime.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        delta_k_h_eff_ (float): Initial effective household-level capital loss.
        lambda_h_ (float): Recovery rate.

    Returns:
        float or np.ndarray: Effective household-level capital loss at time t_.
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return delta_k_h_eff_ * np.exp(-lambda_h_ * t_)


def delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_):
    """
    Compute the dynamic labor income loss.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        productivity_pi_ (float): Productivity parameter.
        delta_tax_sp_ (float): Tax adjustment factor.
        delta_k_h_eff_ (float): Initial effective household-level capital loss.
        lambda_h_ (float): Recovery rate.

    Returns:
        float or np.ndarray: Household-level labor income loss at time t_.
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    dk_eff = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)
    return dk_eff * productivity_pi_ * (1 - delta_tax_sp_)


def cum_delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_):
    """
    Compute the cumulative household-level labor income loss at time t_.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        productivity_pi_ (float): Productivity parameter.
        delta_tax_sp_ (float): Tax adjustment factor.
        delta_k_h_eff_ (float): Initial effective capital loss.
        lambda_h_ (float): Recovery rate.

    Returns:
        float or np.ndarray: Cumulative labor household-level income loss at time t_.
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return (1 - delta_tax_sp_) * productivity_pi_ / lambda_h_ * delta_k_h_eff_ * (1 - np.exp(-lambda_h_ * t_))


def delta_i_h_div_of_t(t_, recovery_params_=None, productivity_pi_=None, social_protection_share_gamma_h_=None,
                       delta_tax_sp_=None, verbose=True):
    """
    Compute the diversified income loss.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        productivity_pi_ (float): Capital productivity.
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
        delta_tax_sp_ (float): Tax rate.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        float or np.ndarray: Social protection income loss at time t_.
    """
    if delta_tax_sp_ != 0:
        if (recovery_params_ is None or productivity_pi_ is None or social_protection_share_gamma_h_ is None or
                delta_tax_sp_ is None) and verbose:
            raise ValueError("To include tax, recovery parameters, productivity and social protection share must "
                             "be provided to compute social protection income loss.")
        delta_capital_k_eff_national = delta_capital_k_eff_of_t(t_, recovery_params_)
        return delta_capital_k_eff_national * productivity_pi_ * delta_tax_sp_ * social_protection_share_gamma_h_
    return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)


def cum_delta_i_h_sp_of_t(t_, recovery_parameters_, productivity_pi_, social_protection_share_gamma_h_,
                          delta_tax_sp_, verbose=True):
    """
    Compute the cumulative dynamic social protection income loss.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        recovery_parameters_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        productivity_pi_ (float): Capital productivity.
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
        delta_tax_sp_ (float): Tax rate.
        verbose (bool): Whether to print warnings. Defaults to True.

    Returns:
        float or np.ndarray: Cumulative social protection income loss at time t_.
    """
    if delta_tax_sp_ != 0:
        if recovery_parameters_ is not None:
            cum_d_i_h_sp_of_t = None
            for delta_k_h_eff_other_, lambda_h_other_ in recovery_parameters_:
                delta_k_h_eff_other_of_t = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_other_,
                                                                   lambda_h_=lambda_h_other_)
                cum_d_i_h_sp_from_h_other = (social_protection_share_gamma_h_ * productivity_pi_ * delta_tax_sp_ / lambda_h_other_ *
                                             (delta_k_h_eff_other_ - delta_k_h_eff_other_of_t))
                if cum_d_i_h_sp_of_t is None:
                    cum_d_i_h_sp_of_t = cum_d_i_h_sp_from_h_other
                else:
                    cum_d_i_h_sp_of_t += cum_d_i_h_sp_from_h_other
            return cum_d_i_h_sp_of_t
        elif verbose:
            print("Warning: function cum_delta_i_h_sp_of_t received no recovery_parameters. Returning 0.")
    return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)


def cum_delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                       recovery_params_, social_protection_share_gamma_h_):
    """
    Compute the cumulative dynamic income loss.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        productivity_pi_ (float): Capital productivity.
        delta_tax_sp_ (float): Tax rate.
        delta_k_h_eff_ (float): Initial effective capital loss.
        lambda_h_ (float): Recovery rate.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.

    Returns:
        float or np.ndarray: Cumulative income loss at time t_.
    """
    cum_d_i_h_lab_of_t = cum_delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                                delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)

    cum_d_i_h_sp_of_t = cum_delta_i_h_sp_of_t(t_, recovery_parameters_=recovery_params_, productivity_pi_=productivity_pi_,
                                              social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                                              delta_tax_sp_=delta_tax_sp_)
    return cum_d_i_h_lab_of_t + cum_d_i_h_sp_of_t


def delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                   recovery_params_, social_protection_share_gamma_h_, return_elements=False):
    """
    Compute the dynamic income loss.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        productivity_pi_ (float): Capital productivity.
        delta_tax_sp_ (float): Tax rate.
        delta_k_h_eff_ (float): Initial effective capital loss.
        lambda_h_ (float): Recovery rate.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
        return_elements (bool): Whether to return individual components. Defaults to False.

    Returns:
        float or np.ndarray: Total income loss at time t_.
        If return_elements is True, returns a tuple of individual components.
    """
    d_i_h_lab_of_t = delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                        delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)
    d_i_h_sp_of_t = delta_i_h_div_of_t(t_, recovery_params_=recovery_params_, productivity_pi_=productivity_pi_,
                                       social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                                       delta_tax_sp_=delta_tax_sp_)
    # income from post disaster support (a one-time payment) is similar to savings and not considered here
    if return_elements:
        return d_i_h_lab_of_t, d_i_h_sp_of_t
    return d_i_h_lab_of_t + d_i_h_sp_of_t


def delta_c_h_reco_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_):
    """
    Compute the dynamic recovery consumption loss in the exponential recovery regime.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        delta_k_h_eff_ (float): Initial effective capital loss.
        lambda_h_ (float): Recovery rate.
        sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.

    Returns:
        float or np.ndarray: Recovery consumption loss at time t_.
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return lambda_h_ * sigma_h_ * delta_k_h_eff_ * np.exp(-lambda_h_ * t_)


def cum_delta_c_h_reco_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_):
    """
    Compute the cumulative dynamic recovery consumption loss in the exponential recovery regime.

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        delta_k_h_eff_ (float): Initial effective capital loss.
        lambda_h_ (float): Recovery rate.
        sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.

    Returns:
        float or np.ndarray: Cumulative recovery consumption loss at time t_.
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return sigma_h_ * delta_k_h_eff_ * (1 - np.exp(-lambda_h_ * t_))


def calc_t_hat(lambda_h_=None, consumption_floor_xi_=None, productivity_pi_=None, delta_tax_sp_=None,
               delta_k_h_eff_=None, savings_s_h_=None, delta_i_h_pds_=None,
               recovery_params_=None, social_protection_share_gamma_h_=None, sigma_h_=None):
    """
    Compute the time at which the consumption floor is reached.

    Args:
        lambda_h_ (float): Recovery rate.
        consumption_floor_xi_ (float): Consumption floor parameter.
        productivity_pi_ (float): Productivity parameter.
        delta_tax_sp_ (float): Tax rate.
        delta_k_h_eff_ (float): Initial effective capital loss.
        savings_s_h_ (float): Savings available for recovery.
        delta_i_h_pds_ (float): Post-disaster support income.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
        sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.

    Returns:
        float: Time at which the consumption floor is reached.
    """
    if delta_tax_sp_ == 0:
        if consumption_floor_xi_ is None or lambda_h_ is None:
            raise ValueError("Consumption floor and lambda must be provided to compute t_hat without tax.")
        return -1 / lambda_h_ * np.log(consumption_floor_xi_)
    else:
        # check whether consumption losses can be fully offset with savings
        def cum_d_c(t__):
            cum_d_i_h = cum_delta_i_h_of_t(t__, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, recovery_params_, social_protection_share_gamma_h_)
            cum_d_c_h_reco = cum_delta_c_h_reco_of_t(t__, delta_k_h_eff_, lambda_h_, sigma_h_)
            return cum_d_i_h + cum_d_c_h_reco
        if cum_d_c(np.inf) <= savings_s_h_ + delta_i_h_pds_:
            return np.inf

        # if not, find the time at which savings are used up
        def search_func(t__):
            cum_d_c_of_t__ = cum_d_c(t__)
            delta_c_h_floor = delta_c_h_of_t(
                t__, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, 0, 0,
                recovery_params_, social_protection_share_gamma_h_
            )
            return cum_d_c_of_t__ - delta_c_h_floor * t__ - (savings_s_h_ + delta_i_h_pds_)
        return optimize.root_scalar(search_func, method='brentq', bracket=[0, 1e4]).root


def delta_c_h_savings_pds_of_t(t_, lambda_h_, sigma_h_, productivity_pi_, consumption_floor_xi_, t_hat_, delta_k_h_eff_,
                               savings_s_h_, delta_i_h_pds_, delta_i_h, delta_c_h_reco, delta_tax_sp_,
                               recovery_params_, social_protection_share_gamma_h_, consumption_offset_):
    """
    Compute the dynamic consumption gain from savings and post-disaster support (PDS).

    Args:
        t_ (float or np.ndarray): Time or array of time points.
        lambda_h_ (float): Recovery rate.
        sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
        productivity_pi_ (float): Capital productivity.
        consumption_floor_xi_ (float): Consumption floor parameter.
        t_hat_ (float): Time at which the consumption floor is reached.
        delta_k_h_eff_ (float): Initial effective capital loss.
        savings_s_h_ (float): Savings available for recovery.
        delta_i_h_pds_ (float): Post-disaster support income.
        delta_i_h (float or np.ndarray): Total income loss at time t_.
        delta_c_h_reco (float or np.ndarray): Recovery consumption loss at time t_.
        delta_tax_sp_ (float): Tax rate.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
        consumption_offset_ (float or None): Offset for recomputation of consumption losses.

    Returns:
        float or np.ndarray: Dynamic consumption gain from savings and PDS at time t_.

    Raises:
        ValueError: If both `consumption_offset_` and `consumption_floor_xi_` are provided.
    """
    if savings_s_h_ + delta_i_h_pds_ <= 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)

    if consumption_offset_ is not None and not np.isnan(consumption_floor_xi_):
        raise ValueError("Cannot provide both consumption_offset_ and consumption_floor_xi_ parameters. "
                         "The former is used for recomputation, the latter during optimization.")
    # for optimization, neglecting tax, social protection, and transfers
    if not np.isnan(consumption_floor_xi_):
        alpha = calc_alpha(
            lambda_h_=lambda_h_,
            sigma_h_=sigma_h_,
            delta_k_h_eff_=delta_k_h_eff_,
            productivity_pi_=productivity_pi_
        )
        d_c_h_savings_pds_of_t = alpha * (np.exp(-lambda_h_ * t_) - consumption_floor_xi_)
        if consumption_floor_xi_ == 0:
            # consumption losses can be fully offset with savings
            return d_c_h_savings_pds_of_t

    # for evaluation, including tax, social protection, and transfers
    else:
        if t_hat_ < np.inf:
            # consumption losses cannot be fully offset with savings
            if consumption_offset_ is None:
                consumption_offset_ = delta_c_h_of_t(t_hat_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                                                     lambda_h_, sigma_h_, 0, 0,
                                                     recovery_params_, social_protection_share_gamma_h_,
                                                     consumption_floor_xi_, t_hat_,
                                                     return_elements=False)
            d_c_h_savings_pds_of_t = (delta_i_h + delta_c_h_reco) - consumption_offset_
        else:
            # consumption losses can be fully offset with savings
            d_c_h_savings_pds_of_t = delta_i_h + delta_c_h_reco
    if isinstance(t_, (float, int)):
        if not (0 <= t_ < t_hat_):
            d_c_h_savings_pds_of_t = 0
    else:
        d_c_h_savings_pds_of_t[t_ >= t_hat_] = 0
        d_c_h_savings_pds_of_t[t_ < 0] = 0
    return d_c_h_savings_pds_of_t


def delta_c_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                   savings_s_h_, delta_i_h_pds_, recovery_params_, social_protection_share_gamma_h_,
                   consumption_floor_xi_=None, t_hat=None,
                   consumption_offset=None, return_elements=False):
    """
        Compute the dynamic consumption loss.

        Args:
            t_ (float or np.ndarray): Time or array of time points.
            productivity_pi_ (float): Capital productivity.
            delta_tax_sp_ (float): Tax rate.
            delta_k_h_eff_ (float): Initial effective capital loss.
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
                delta_k_h_eff_ (float) and lambda_h_ (float).
            social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
            consumption_floor_xi_ (float, optional): Consumption floor parameter. Defaults to None.
            t_hat (float, optional): Time at which the consumption floor is reached. Defaults to None.
            consumption_offset (float, optional): Offset for recomputation of consumption losses. Defaults to None.
            return_elements (bool, optional): Whether to return individual components of the consumption loss.
                Defaults to False.

        Returns:
            float or np.ndarray: Total dynamic consumption loss at time t_.
            If `return_elements` is True, returns a tuple containing:
                - d_i_h_lab_of_t: Labor income loss.
                - d_i_h_sp_of_t: Social protection income loss.
                - delta_c_h_reco: Recovery consumption loss.
                - delta_c_h_savings_pds: Consumption gain from savings and PDS.

        Raises:
            ValueError: If `consumption_floor_xi_` and `t_hat` are not both provided or both are None.
        """

    # no income loss from labor or social protection and transfers --> baseline case
    if delta_k_h_eff_ == 0 and (recovery_params_ is None or np.all([rp[0] == 0 for rp in recovery_params_])):
        if return_elements:
            return t_ * 0, t_ * 0, t_ * 0, t_ * 0
        return t_ * 0

    if consumption_floor_xi_ == t_hat is None:
        consumption_floor_xi_, t_hat = solve_consumption_floor_xi(
            lambda_h_=lambda_h_,
            sigma_h_=sigma_h_,
            delta_k_h_eff_=delta_k_h_eff_,
            productivity_pi_=productivity_pi_,
            savings_s_h_=savings_s_h_,
            delta_i_h_pds_=delta_i_h_pds_,
            delta_tax_sp_=delta_tax_sp_,
            recovery_params_=recovery_params_,
            social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        )
    elif np.sum([v is None for v in [consumption_floor_xi_, t_hat]]) > 0:
        raise ValueError("Must pass all or none of consumption floor xi and t_hat")
    d_i_h_lab_of_t, d_i_h_sp_of_t = delta_i_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        return_elements=True,
    )
    delta_i_h = d_i_h_lab_of_t + d_i_h_sp_of_t
    delta_c_h_reco = delta_c_h_reco_of_t(
        t_=t_,
        sigma_h_=sigma_h_,
        lambda_h_=lambda_h_,
        delta_k_h_eff_=delta_k_h_eff_,
    )
    delta_c_h_savings_pds = delta_c_h_savings_pds_of_t(t_=t_, lambda_h_=lambda_h_, sigma_h_=sigma_h_,
                                                       productivity_pi_=productivity_pi_,
                                                       consumption_floor_xi_=consumption_floor_xi_, t_hat_=t_hat,
                                                       delta_k_h_eff_=delta_k_h_eff_, savings_s_h_=savings_s_h_,
                                                       delta_i_h_pds_=delta_i_h_pds_, delta_i_h=delta_i_h,
                                                       delta_c_h_reco=delta_c_h_reco,
                                                       delta_tax_sp_=delta_tax_sp_, recovery_params_=recovery_params_,
                                                       social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                                                       consumption_offset_=consumption_offset)
    if return_elements:
        return d_i_h_lab_of_t, d_i_h_sp_of_t, delta_c_h_reco, delta_c_h_savings_pds
    return delta_i_h + delta_c_h_reco - delta_c_h_savings_pds


def baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share):
    """
    Compute the baseline consumption level for a household.

    Args:
        productivity_pi_ (float): Capital productivity.
        k_h_eff_ (float): Effective capital stock of the household.
        delta_tax_sp_ (float): Tax adjustment factor.
        diversified_share (float): Share of income that is diversified.

    Returns:
        float: Baseline consumption level of the household.
    """
    return productivity_pi_ * k_h_eff_ * (1 - delta_tax_sp_) / (1 - diversified_share)


def consumption_c_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_,
                       delta_i_h_pds_, k_h_eff_, recovery_params_, social_protection_share_gamma_h_,
                       diversified_share_, consumption_floor_xi_=None, t_hat=None, consumption_offset=None, include_tax=False):
    """
        Compute the dynamic consumption level.

        Args:
            t_ (float or np.ndarray): Time or array of time points.
            productivity_pi_ (float): Capital productivity.
            delta_tax_sp_ (float): Tax rate.
            delta_k_h_eff_ (float): Initial effective capital loss.
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            k_h_eff_ (float): Effective capital stock of the household.
            recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
                delta_k_h_eff_ (float) and lambda_h_ (float).
            social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
            diversified_share_ (float): Share of income that is diversified.
            consumption_floor_xi_ (float, optional): Consumption floor parameter. Defaults to None.
            t_hat (float, optional): Time at which the consumption floor is reached. Defaults to None.
            consumption_offset (float, optional): Offset for recomputation of consumption losses. Defaults to None.
            include_tax (bool, optional): Whether to include tax in the computation. Defaults to False.

        Returns:
            float or np.ndarray: Dynamic consumption level at time t_.
        """
    consumption_loss = delta_c_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_ if include_tax else 0,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        recovery_params_=recovery_params_ if include_tax else [(0, 0)],
        social_protection_share_gamma_h_=social_protection_share_gamma_h_ if include_tax else 0,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        consumption_offset=consumption_offset,
    )
    baseline_consumption = baseline_consumption_c_h(
        productivity_pi_=productivity_pi_,
        k_h_eff_=k_h_eff_,
        delta_tax_sp_=delta_tax_sp_,
        diversified_share=diversified_share_,
    )
    return baseline_consumption - consumption_loss


def calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_):
    """
        Compute the alpha parameter.

        Args:
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            delta_k_h_eff_ (float): Initial effective capital loss.
            productivity_pi_ (float): Capital productivity.

        Returns:
            float: The computed alpha parameter.
        """
    return (productivity_pi_ + lambda_h_ * sigma_h_) * delta_k_h_eff_


def solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                               delta_i_h_pds_, delta_tax_sp_, recovery_params_,
                               social_protection_share_gamma_h_):
    """
    Solve for the consumption floor parameter (xi).

    Args:
        lambda_h_ (float): Recovery rate.
        sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
        delta_k_h_eff_ (float): Initial effective capital loss.
        productivity_pi_ (float): Capital productivity.
        savings_s_h_ (float): Savings available for recovery.
        delta_i_h_pds_ (float): Post-disaster support income.
        delta_tax_sp_ (float): Tax rate.
        recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
            delta_k_h_eff_ (float) and lambda_h_ (float).
        social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.

    Returns:
        tuple: A tuple containing:
            - float: The solved consumption floor parameter (xi).
            - float: Time at which the consumption floor is reached (t_hat).
    """

    # for the optimization problem, neglecting tax, social protection, and transfers
    if delta_tax_sp_ == 0:
        alpha = calc_alpha(
            lambda_h_=lambda_h_,
            sigma_h_=sigma_h_,
            delta_k_h_eff_=delta_k_h_eff_,
            productivity_pi_=productivity_pi_
        )
        if lambda_h_ <= 0 or savings_s_h_ + delta_i_h_pds_ <= 0:
            xi_res, t_hat = np.nan, 0  # No savings available to offset consumption losses
        elif savings_s_h_ + delta_i_h_pds_ >= alpha / lambda_h_:
            xi_res, t_hat = 0, np.inf  # All capital losses can be repaired at time 0, and consumption loss can be reduced to 0
        else:
            # Solve numerically for xi
            def xi_func(xi_):
                rhs = 1 - lambda_h_ / alpha * (savings_s_h_ + delta_i_h_pds_)
                lhs = xi_ * (1 - np.log(xi_)) if xi_ > 0 else 0
                return lhs - rhs
            xi_res = optimize.brentq(xi_func, 0, 1)
            t_hat = calc_t_hat(lambda_h_=lambda_h_, consumption_floor_xi_=xi_res, delta_tax_sp_=delta_tax_sp_)

    # for the recomputation, including tax, social protection, and transfers
    else:
        if savings_s_h_ + delta_i_h_pds_ <= 0:
            # no savings to offset losses
            xi_res, t_hat = np.nan, 0
        else:
            # find t_hat (with tax, social protection, and transfers), at which savings are used up
            t_hat = calc_t_hat(
                lambda_h_=lambda_h_,
                consumption_floor_xi_=None,
                productivity_pi_=productivity_pi_,
                delta_tax_sp_=delta_tax_sp_,
                delta_k_h_eff_=delta_k_h_eff_,
                savings_s_h_=savings_s_h_,
                delta_i_h_pds_=delta_i_h_pds_,
                recovery_params_=recovery_params_,
                social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                sigma_h_=sigma_h_,
            )
            # with tax, xi_res has no meaning (as alpha would also depend on tax and income losses from transfers)
            xi_res = np.nan
    return xi_res, t_hat


def calc_leftover_savings(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_):
    """
        Compute the leftover savings after accounting for recovery costs.

        Args:
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            delta_k_h_eff_ (float): Initial effective capital loss.
            productivity_pi_ (float): Capital productivity.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.

        Returns:
            float: The leftover savings after recovery costs are accounted for.
        """
    alpha = calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_)
    return max(0., savings_s_h_ + delta_i_h_pds_ - alpha / lambda_h_)


def objective_func(lambda_h_, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                   delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_, delta_tax_sp_, diversified_share_):
    """
        Objective function to be minimized for optimizing the recovery rate (lambda_h).

        Args:
            lambda_h_ (float): Recovery rate to be optimized.
            capital_t_ (float): Time horizon for the welfare computation.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            delta_k_h_eff_ (float): Initial effective capital loss.
            productivity_pi_ (float): Capital productivity.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            discount_rate_rho_ (float): Discount rate for future welfare.
            k_h_eff_ (float): Effective capital stock of the household.
            delta_tax_sp_ (float): Tax rate.
            diversified_share_ (float): Share of income that is diversified.

        Returns:
            float: Negative of the aggregate welfare value, as the function is designed for minimization.
        """
    # print all inputs
    if isinstance(lambda_h_, np.ndarray):
        lambda_h_ = lambda_h_[0]

    objective = aggregate_welfare_w_of_c_of_capital_t(
        capital_t_=capital_t_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_,
        eta_=eta_,
        delta_tax_sp_=delta_tax_sp_,
        discount_rate_rho_=discount_rate_rho_,
        k_h_eff_=k_h_eff_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        recovery_params_=[(0, 0)],  # for optimization, no recovery parameters are needed
        social_protection_share_gamma_h_=0,  # for optimization, no social protection share is needed
        diversified_share_=diversified_share_,
        include_tax=False
    )

    # for the optimization, include leftover savings, s.th. not the first but the fastest recovery path is chosen where
    # consumption losses can be fully offset
    leftover = calc_leftover_savings(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                     delta_i_h_pds_)
    objective += leftover
    return -objective  # negative to transform into a minimization problem



def welfare_of_c(c_, eta_):
    """
    Welfare function.

    Args:
        c_ (float or np.ndarray): Consumption level(s). Must be positive.
        eta_ (float): Elasticity of marginal utility of consumption.

    Returns:
        float or np.ndarray: Welfare value(s) corresponding to the given consumption level(s).

    Raises:
        ValueError: If any value in `c_` is non-positive.
    """
    if (np.array(c_) <= 0).any():
        return -np.inf
    return (c_ ** (1 - eta_) - 1) / (1 - eta_)


def welfare_w_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                   sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_,
                   recovery_params_, social_protection_share_gamma_h_, diversified_share_,
                   consumption_floor_xi_=None, t_hat=None,
                   consumption_offset=None, discount=True, include_tax=False):
    """
        Compute the discounted time-dependent welfare.

        Args:
            t_ (float or np.ndarray): Time or array of time points.
            discount_rate_rho_ (float): Discount rate for future welfare.
            productivity_pi_ (float): Capital productivity.
            delta_tax_sp_ (float): Tax rate.
            delta_k_h_eff_ (float): Initial effective capital loss.
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            k_h_eff_ (float): Effective capital stock of the household.
            recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
                delta_k_h_eff_ (float) and lambda_h_ (float).
            social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
            diversified_share_ (float): Share of income that is diversified.
            consumption_floor_xi_ (float, optional): Consumption floor parameter. Defaults to None.
            t_hat (float, optional): Time at which the consumption floor is reached. Defaults to None.
            consumption_offset (float, optional): Offset for recomputation of consumption losses. Defaults to None.
            discount (bool, optional): Whether to apply discounting to welfare. Defaults to True.
            include_tax (bool, optional): Whether to include tax in the computation. Defaults to False.

        Returns:
            float or np.ndarray: Discounted welfare value(s) at time t_.
        """
    c_of_t = consumption_c_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        k_h_eff_=k_h_eff_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        consumption_offset=consumption_offset,
        include_tax=include_tax,
    )
    assert (np.array(c_of_t) > 0).all()
    res = welfare_of_c(c_of_t, eta_=eta_)
    if discount:
        res = res * np.exp(-discount_rate_rho_ * t_)
    return res


def aggregate_welfare_w_of_c_of_capital_t(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_,
                                          delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_,
                                          eta_, k_h_eff_, recovery_params_,
                                          social_protection_share_gamma_h_, diversified_share_,
                                          consumption_floor_xi_=None, t_hat=None, consumption_offset=None,
                                          include_tax=False):
    """
        Compute the time-aggregate welfare function.

        Args:
            capital_t_ (float): Time horizon for the welfare computation.
            discount_rate_rho_ (float): Discount rate for future welfare.
            productivity_pi_ (float): Capital productivity.
            delta_tax_sp_ (float): Tax rate.
            delta_k_h_eff_ (float): Initial effective capital loss.
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            k_h_eff_ (float): Effective capital stock of the household.
            recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
                delta_k_h_eff_ (float) and lambda_h_ (float).
            social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.
            diversified_share_ (float): Share of income that is diversified.
            consumption_floor_xi_ (float, optional): Consumption floor parameter. Defaults to None.
            t_hat (float, optional): Time at which the consumption floor is reached. Defaults to None.
            consumption_offset (float, optional): Offset for recomputation of consumption losses. Defaults to None.
            include_tax (bool, optional): Whether to include tax in the computation. Defaults to False.

        Returns:
            float: Time-aggregated welfare value over the specified time horizon.
        """
    t_ = np.array([0] + list(np.geomspace(1e-6, capital_t_, int(np.ceil(5000/np.log(50) * np.log(capital_t_))))))
    w_agg = integrate.trapezoid(welfare_w_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, recovery_params_, social_protection_share_gamma_h_, diversified_share_, consumption_floor_xi_, t_hat, consumption_offset, True, include_tax), t_)
    return w_agg


def recompute_with_tax(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                       lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_,
                       diversified_share_, recovery_params_, social_protection_share_gamma_h_, consumption_lines_=None):
    """
        Recompute the final change in welfare, including tax, social protection, and transfers.

        Args:
            capital_t_ (float): Time horizon for the welfare computation.
            discount_rate_rho_ (float): Discount rate for future welfare.
            productivity_pi_ (float): Capital productivity.
            delta_tax_sp_ (float): Tax rate.
            delta_k_h_eff_ (float): Initial effective capital loss.
            lambda_h_ (float): Recovery rate.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            k_h_eff_ (float): Effective capital stock of the household.
            diversified_share_ (float): Share of income that is diversified.
            recovery_params_ (list of tuples): Recovery parameters, where each tuple contains
                delta_k_h_eff_ (float) and lambda_h_ (float).
            social_protection_share_gamma_h_ (float): Share of total diversified income that goes to household h.

        Returns:
            tuple: A tuple containing:
                - float: Welfare loss due to the disaster.
                - float: Used liquidity (savings and post-disaster support).
                - float: Short-term consumption loss.
                - float: Maximum consumption loss.
        """
    consumption_floor_xi_, t_hat = solve_consumption_floor_xi(
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        delta_tax_sp_=delta_tax_sp_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
    )

    if t_hat < np.inf:
        consumption_offset = delta_c_h_of_t(t_hat, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                                             lambda_h_, sigma_h_, 0, 0,
                                             recovery_params_, social_protection_share_gamma_h_, consumption_floor_xi_,
                                             t_hat, return_elements=False)
    else:
        consumption_offset = np.nan

    # determine the used savings
    def used_liquidity_func(t_):
        return delta_c_h_of_t(
            t_=t_,
            productivity_pi_=productivity_pi_,
            delta_tax_sp_=delta_tax_sp_,
            delta_k_h_eff_=delta_k_h_eff_,
            lambda_h_=lambda_h_,
            sigma_h_=sigma_h_,
            savings_s_h_=savings_s_h_,
            delta_i_h_pds_=delta_i_h_pds_,
            recovery_params_=recovery_params_,
            social_protection_share_gamma_h_=social_protection_share_gamma_h_,
            return_elements=True,
            consumption_floor_xi_=consumption_floor_xi_,
            t_hat=t_hat,
            consumption_offset=consumption_offset,
        )[3]
    t_ = np.array([0] + list(np.geomspace(1e-6, capital_t_, int(np.ceil(5000/np.log(50) * np.log(capital_t_))))))
    used_liquidity = integrate.trapezoid(used_liquidity_func(t_), t_)

    delta_c_h_of_t_partial = partial(
        delta_c_h_of_t,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        return_elements=False,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        consumption_offset=consumption_offset
    )
    delta_c_h_over_t = delta_c_h_of_t_partial(t_)
    dc_short_term = integrate.trapezoid(delta_c_h_over_t, t_)
    delta_c_h_max = delta_c_h_of_t_partial(0)

    c_baseline = baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share_)
    def calc_time_below_consumption_level(c_level):
        if c_baseline < c_level:
            return np.inf
        elif c_baseline - delta_c_h_max > c_level:
            return 0
        else:
            def opt_fun(t_):
                return c_baseline - delta_c_h_of_t_partial(t_) - c_level
            return root_scalar(opt_fun, bracket=[0, capital_t_], method='brentq').root
    if consumption_lines_ is not None:
        c_level_durations = {cl: calc_time_below_consumption_level(c_level) for cl, c_level in consumption_lines_.items()}
    else:
        c_level_durations = {}

    aggregate_w_partial = partial(
        aggregate_welfare_w_of_c_of_capital_t,
        capital_t_=capital_t_,
        discount_rate_rho_=discount_rate_rho_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        eta_=eta_,
        k_h_eff_=k_h_eff_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        consumption_offset=consumption_offset,
        include_tax=True
    )
    w_baseline = aggregate_w_partial(delta_k_h_eff_ = 0, recovery_params_=[(0, 0)])
    w_disaster = aggregate_w_partial(delta_k_h_eff_=delta_k_h_eff_, recovery_params_=recovery_params_)

    return w_baseline - w_disaster, used_liquidity, dc_short_term, delta_c_h_max, c_level_durations


def recompute_with_tax_wrapper(recompute_args):
    """
        Wrapper function for `recompute_with_tax`.

        Args:
            recompute_args (tuple): A tuple containing:
                - index (int): The index of the row being processed.
                - row (pd.Series): A pandas Series containing the input parameters for the `recompute_with_tax` function.

        Returns:
            tuple: A tuple containing:
                - int: The index of the row being processed.
                - tuple: The result of the `recompute_with_tax` function, which includes:
                    - float: Welfare loss due to the disaster.
                    - float: Used liquidity (savings and post-disaster support).
                    - float: Short-term consumption loss.
                    - float: Maximum consumption loss.

        Raises:
            Exception: If an error occurs during the execution of `recompute_with_tax`.
        """
    index, row = recompute_args
    consumption_lines = [i for i in row.index if 'pov_line' in i]
    try:
        return recompute_with_tax(
            capital_t_=row['capital_t'],
            discount_rate_rho_=row['discount_rate_rho'],
            productivity_pi_=row['productivity_pi'],
            delta_tax_sp_=row['delta_tax_sp'],
            delta_k_h_eff_=row['delta_k_h_eff'],
            lambda_h_=row['lambda_h'],
            sigma_h_=row['sigma_h'],
            savings_s_h_=row['savings_s_h'],
            delta_i_h_pds_=row['delta_i_h_pds'],
            eta_=row['eta'],
            k_h_eff_=row['k_h_eff'],
            diversified_share_=row['diversified_share'],
            recovery_params_=row['recovery_params'],
            social_protection_share_gamma_h_=row['social_protection_share_gamma_h'],
            consumption_lines_={cl: row[cl] for cl in consumption_lines},
        )
    except Exception as e:
        print(f"Error in row {index}: {e}")
        raise e


def recompute_data_with_tax(df_in, num_cores=None):
    """
        Recompute the final change in welfare, savings, and consumption.
        Calculation includes tax, social protection, and transfers.

        Args:
            df_in (pd.DataFrame): Input DataFrame containing the parameters for the welfare computation.
            num_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to None,
                which uses all available cores.

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - dW_reco: Welfare loss due to the disaster.
                - dS_reco_PDS: Used liquidity (savings and post-disaster support).
                - dc_short_term: Short-term consumption loss.
                - dC_max: Maximum consumption loss.
        """
    with multiprocessing.Pool(processes=num_cores) as pool:
        res = list(tqdm.tqdm(pool.imap(recompute_with_tax_wrapper, df_in.iterrows()), total=len(df_in),
                             desc='Recomputing actual welfare loss and used liquidity'))
    res = pd.DataFrame(res, columns=['dW_reco', 'dS_reco_PDS', 'dc_short_term', 'dC_max', 'consumption_level_times'], index=df_in.index)
    res = (
        res
        .drop(columns=["consumption_level_times"])
        .join(res["consumption_level_times"].apply(pd.Series).add_prefix("time_below_"))
    )

    return res


def calc_lambda_bounds_for_optimization(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                        delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_,
                                        delta_tax_sp_, diversified_share_, min_lambda_, max_lambda_):
    """
        Compute the bounds for the recovery rate (lambda_h) optimization.

        Args:
            capital_t_ (float): Time horizon for the welfare computation.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            delta_k_h_eff_ (float): Initial effective capital loss.
            productivity_pi_ (float): Capital productivity.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            discount_rate_rho_ (float): Discount rate for future welfare.
            k_h_eff_ (float): Effective capital stock of the household.
            delta_tax_sp_ (float): Tax rate.
            diversified_share_ (float): Share of income that is diversified.
            min_lambda_ (float): Minimum bound for the recovery rate.
            max_lambda_ (float): Maximum bound for the recovery rate.

        Returns:
            tuple: A tuple containing:
                - float: Minimum bound for the recovery rate (lambda_h).
                - float: Maximum bound for the recovery rate (lambda_h).
                - float: Initial guess for the recovery rate (lambda_h) for optimization.

        Raises:
            ValueError: If no valid bounds for lambda_h exist that maintain positive consumption.
        """

    assert min_lambda_ >= 0

    # if no maximum consumption loss is defined find the maximal allowable lambda that maintains positive consumption
    c_baseline = baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share_)
    if savings_s_h_ + delta_i_h_pds_ <= 0:
        if c_baseline < calc_alpha(0, sigma_h_, delta_k_h_eff_, productivity_pi_):
            # there is no lambda that maintains positive consumption
            print(f"Optimization not possible for capital_t={capital_t_}, sigma_h={sigma_h_}, "
                  f"delta_k_h_eff={delta_k_h_eff_}, productivity_pi={productivity_pi_}, savings_s_h={savings_s_h_}, "
                  f"delta_i_h_pds={delta_i_h_pds_}, eta={eta_}, discount_rate_rho={discount_rate_rho_}, "
                  f"k_h_eff={k_h_eff_}, delta_tax_sp={delta_tax_sp_}, "
                  f"diversified_share={diversified_share_}. No lambda exists that maintains positive consumption.")
            return np.nan, np.nan, np.nan
        max_lambda_ = min(max_lambda_, 1 / sigma_h_ * (c_baseline / delta_k_h_eff_ - productivity_pi_))
    elif savings_s_h_ + delta_i_h_pds_ > 0:
        def lambda_func(lambda_h_):
            alpha_ = calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_)
            xi_, _ = solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_,
                                                      savings_s_h_, delta_i_h_pds_, 0, None, 0)
            return alpha_ * xi_ - (c_baseline - 1e-5)  # alpha * xi must be smaller than the baseline consumption

        if np.sign(lambda_func(min_lambda_)) != np.sign(lambda_func(max_lambda_)):
            opt_lambda = optimize.brentq(lambda_func, min_lambda_, max_lambda_, full_output=True)[0]
            if lambda_func(opt_lambda - 1e-5) < lambda_func(opt_lambda + 1e-5):
                max_lambda_ = min(max_lambda_, opt_lambda)
            else:
                min_lambda_ = max(min_lambda_, opt_lambda)
        elif lambda_func(1e-5) > 0:
            # there is no lambda that maintains positive consumption
            print(f"Optimization not possible for capital_t={capital_t_}, sigma_h={sigma_h_}, "
                  f"delta_k_h_eff={delta_k_h_eff_}, productivity_pi={productivity_pi_}, savings_s_h={savings_s_h_}, "
                  f"delta_i_h_pds={delta_i_h_pds_}, eta={eta_}, discount_rate_rho={discount_rate_rho_}, "
                  f"k_h_eff={k_h_eff_}, delta_tax_sp={delta_tax_sp_}, "
                  f"diversified_share={diversified_share_}. No lambda exists that maintains positive consumption.")
            return np.nan, np.nan, np.nan

    # check whether some lambda exists, such that consumption losses can be fully offset
    if (savings_s_h_ + delta_i_h_pds_) > sigma_h_ * delta_k_h_eff_:
        lambda_full_offset = 1 / ((savings_s_h_ + delta_i_h_pds_) / delta_k_h_eff_ - sigma_h_) * productivity_pi_
        if lambda_full_offset < max_lambda_:
            min_lambda_ = max(min_lambda_, lambda_full_offset)

    init_candidates = np.linspace(min_lambda_ + .2 * (max_lambda_ - min_lambda_),
                                  max_lambda_ - .2 * (max_lambda_ - min_lambda_), 10)
    best_lambda_init = None
    best_candicate_objective = None
    for ic in init_candidates:
        objective = objective_func(ic, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                   delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_, delta_tax_sp_,
                                   diversified_share_)
        if best_lambda_init is None or objective < best_candicate_objective:
            best_lambda_init = ic
            best_candicate_objective = objective

    return min_lambda_, max_lambda_, best_lambda_init


def optimize_lambda(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
                    discount_rate_rho_, k_h_eff_, delta_tax_sp_, diversified_share_,
                    tolerance=1e-10, min_lambda=0.05, max_lambda=100):
    """
        Optimize the recovery rate (lambda_h) for a given set of parameters.

        Args:
            capital_t_ (float): Time horizon for the welfare computation.
            sigma_h_ (float): Share of asset loss that households have to reconstruct at their own cost.
            delta_k_h_eff_ (float): Initial effective capital loss.
            productivity_pi_ (float): Capital productivity.
            savings_s_h_ (float): Savings available for recovery.
            delta_i_h_pds_ (float): Post-disaster support income.
            eta_ (float): Elasticity of marginal utility of consumption.
            discount_rate_rho_ (float): Discount rate for future welfare.
            k_h_eff_ (float): Effective capital stock of the household.
            delta_tax_sp_ (float): Tax rate.
            diversified_share_ (float): Share of income that is diversified.
            tolerance (float, optional): Tolerance for the optimization algorithm. Defaults to 1e-10.
            min_lambda (float, optional): Minimum bound for the recovery rate. Defaults to 0.05.
            max_lambda (float, optional): Maximum bound for the recovery rate. Defaults to 100.

        Returns:
            float: The optimized recovery rate (lambda_h).

        Raises:
            ValueError: If no valid bounds for lambda_h exist that maintain positive consumption.
        """

    min_lambda, max_lambda, lambda_h_init = calc_lambda_bounds_for_optimization(
        capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_, discount_rate_rho_,
        k_h_eff_, delta_tax_sp_, diversified_share_, min_lambda, max_lambda
    )

    if np.isnan(min_lambda) or np.isnan(max_lambda) or np.isnan(lambda_h_init):
        return [np.nan]

    res = optimize.minimize(
        fun=objective_func,
        x0=np.array(lambda_h_init),
        bounds=[(min_lambda, max_lambda)],
        args=(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
              discount_rate_rho_, k_h_eff_, delta_tax_sp_, diversified_share_),
        method='Nelder-Mead',
        tol=tolerance,
    )

    return res.x


def optimize_lambda_wrapper(opt_args, min_lambda, max_lambda):
    """
        Wrapper function for `optimize_lambda`.

        Args:
            opt_args (tuple): A tuple containing:
                - index (int): The index of the row being processed.
                - row (pd.Series): A pandas Series containing the input parameters for the `optimize_lambda` function.
            min_lambda (float): Minimum bound for the recovery rate (lambda_h).
            max_lambda (float): Maximum bound for the recovery rate (lambda_h).

        Returns:
            tuple: A tuple containing:
                - int: The index of the row being processed.
                - float: The optimized recovery rate (lambda_h).

        Raises:
            Exception: If an error occurs during the execution of `optimize_lambda`.
        """
    index, row = opt_args
    try:
        res = optimize_lambda(
            capital_t_=row['capital_t'],
            sigma_h_=row['sigma_h'],
            delta_k_h_eff_=row['delta_k_h_eff'],
            productivity_pi_=row['productivity_pi'],
            savings_s_h_=row['savings_s_h'],
            delta_i_h_pds_=row['delta_i_h_pds'],
            eta_=row['eta'],
            discount_rate_rho_=row['discount_rate_rho'],
            k_h_eff_=row['k_h_eff'],
            tolerance=row['tolerance'],
            delta_tax_sp_=row['delta_tax_sp'],
            diversified_share_=row['diversified_share'],
            min_lambda=min_lambda,
            max_lambda=max_lambda,
        )
    except Exception as e:
        print(f"Error in row {index}: {e}")
        raise e
    return index, res[0]


def optimize_data(df_in, tolerance=1e-2, min_lambda=.05, max_lambda=6, num_cores=None):
    """
        Optimize the recovery rate (lambda_h) for each row in the input DataFrame.

        Args:
            df_in (pd.DataFrame): Input DataFrame containing the parameters for the optimization process.
            tolerance (float, optional): Tolerance for the optimization algorithm. Defaults to 1e-2.
            min_lambda (float, optional): Minimum bound for the recovery rate (lambda_h). Defaults to 0.05.
            max_lambda (float, optional): Maximum bound for the recovery rate (lambda_h). Defaults to 6.
            num_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to None,
                which uses all available cores.

        Returns:
            pd.Series: A pandas Series containing the optimized recovery rate (lambda_h) for each row in the input DataFrame.

        Raises:
            Exception: If an error occurs during the optimization process for any row.
        """
    df = df_in.copy()
    # map index to unique optimization values to improve performance
    df['mapping'] = df.fillna(0).groupby(df.columns.tolist()).ngroup()
    opt_data = df.set_index('mapping', drop=True).drop_duplicates()
    opt_data['tolerance'] = tolerance
    with multiprocessing.Pool(processes=num_cores) as pool:
        res = list(tqdm.tqdm(pool.imap(partial(optimize_lambda_wrapper, min_lambda=min_lambda, max_lambda=max_lambda),
                                       opt_data.iterrows()), total=len(opt_data), desc='Optimizing recovery'))
    res = pd.Series(dict(res), name='lambda_h')
    # map back to original index
    lambda_h_results = pd.merge(df, res, left_on='mapping', right_index=True, how='left').lambda_h
    return lambda_h_results
