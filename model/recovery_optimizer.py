import argparse
import multiprocessing
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import tqdm
from scipy import integrate, optimize


def delta_capital_k_eff_of_t(t_, recovery_parameters_, t_tilde_parameters_=None):
    """
    Compute the dynamic effective capital loss at country level
    """
    if t_tilde_parameters_ is not None:
        raise Warning("t_tilde_parameters_ is not yet implemented. Ignoring.")
    dk_national = 0
    for delta_k_h_eff_, lambda_h_ in recovery_parameters_:
        dk_h = delta_k_h_eff_of_t_exp_regime(t_=t_, delta_tilde_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=0)
        if dk_national is None:
            dk_national = dk_h
        else:
            dk_national += dk_h
    return dk_national


def delta_k_h_eff_of_t(t_, t_tilde_, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic effective capital loss
    """
    to_float = False
    if isinstance(t_, (float, int)):
        to_float = True
        t_ = np.array([t_])
    if t_tilde_ != 0:
        limit_regime = delta_k_h_eff_of_t_limit_regime(t_[t_ < t_tilde_], delta_k_h_eff_, sigma_h_, delta_c_h_max_,
                                                       productivity_pi_)
        delta_tilde_k_h_eff = delta_k_h_eff_of_t_limit_regime(t_tilde_, delta_k_h_eff_, sigma_h_, delta_c_h_max_,
                                                              productivity_pi_)
        exp_regime = delta_k_h_eff_of_t_exp_regime(t_[t_ >= t_tilde_], delta_tilde_k_h_eff, lambda_h_, t_tilde_)
        res = np.concatenate([limit_regime, exp_regime])
    else:
        res = delta_k_h_eff_of_t_exp_regime(t_, delta_k_h_eff_, lambda_h_, t_tilde_)
    if to_float:
        return res.item()
    else:
        return res


def delta_k_h_eff_of_t_exp_regime(t_, delta_tilde_k_h_eff_, lambda_h_, t_tilde_):
    """
    Compute the dynamic effective capital loss in the normal exponential recovery regime
    """
    if delta_tilde_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return delta_tilde_k_h_eff_ * np.exp(-lambda_h_ * (t_ - t_tilde_))


def delta_k_h_eff_of_t_limit_regime(t_, delta_k_h_eff_, sigma_h_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic effective capital loss in the consumption floor regime
    """

    # simplified this with constant sigma (and, thus, limit regime also for non-private reconstruction)
    # delta_k_nonprv = delta_k_h_nonprv_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_)
    # delta_k_prv = delta_k_h_prv_of_t_limit_regime(t_, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_,
    #                                               productivity_pi_)
    # return delta_k_nonprv + delta_k_prv
    factor = (delta_c_h_max_ - productivity_pi_ * delta_k_h_eff_) / productivity_pi_
    return delta_k_h_eff_ - factor * (np.exp(productivity_pi_ / sigma_h_ * t_) - 1)


def delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_,
                       delta_c_h_max_):
    """
    Compute dynamic the labor income loss
    """
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    dk_eff = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=t_tilde_,
                                sigma_h_=sigma_h_, delta_c_h_max_=delta_c_h_max_, productivity_pi_=productivity_pi_)
    return dk_eff * productivity_pi_ * (1 - delta_tax_sp_)


def cum_delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_=0):
    """
    Compute the cumulative dynamic labor income loss in the exponential recovery regime.
    """
    if t_tilde_ != 0:
        raise NotImplementedError("Cumulative labor income loss for reconstruction in the limit regime is not yet "
                                  "implemented.")
    if delta_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return (1 - delta_tax_sp_) * productivity_pi_ / lambda_h_ * delta_k_h_eff_ * (1 - np.exp(-lambda_h_ * (t_ - t_tilde_)))


def delta_i_h_sp_of_t(t_, recovery_params_=None, productivity_pi_=None, social_protection_share_gamma_h_=None,
                      delta_tax_sp_=None, verbose=True):
    if delta_tax_sp_ != 0:
        if (recovery_params_ is None or productivity_pi_ is None or social_protection_share_gamma_h_ is None or
                delta_tax_sp_ is None) and verbose:
            print(recovery_params_, productivity_pi_, social_protection_share_gamma_h_, delta_tax_sp_)
            raise ValueError("To include tax, recovery parameters, productivity and social protection share must "
                             "be provided to compute social protection income loss.")
        else:
            delta_capital_k_eff_national = delta_capital_k_eff_of_t(t_, recovery_params_)
            return delta_capital_k_eff_national * productivity_pi_ * delta_tax_sp_ * social_protection_share_gamma_h_
    return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)


def cum_delta_i_h_sp_of_t(t_, recovery_parameters_, productivity_pi_, social_protection_share_gamma_h_,
                          delta_tax_sp_, t_tilde_parameters_=None, verbose=True):
    if delta_tax_sp_ != 0:
        if t_tilde_parameters_ is not None:
            raise Warning("t_tilde_parameters_ is not yet implemented. Ignoring.")
        if recovery_parameters_ is not None:
            cum_d_i_h_sp_of_t = None
            for delta_k_h_eff_, lambda_h_ in recovery_parameters_:
                delta_k_h_eff_of_h_other_of_t = delta_k_h_eff_of_t_exp_regime(t_=t_, delta_tilde_k_h_eff_=delta_k_h_eff_,
                                                                              lambda_h_=lambda_h_, t_tilde_=0)
                cum_d_i_h_sp_from_h_other = (social_protection_share_gamma_h_ * productivity_pi_ * delta_tax_sp_ / lambda_h_ *
                                             (delta_k_h_eff_ - delta_k_h_eff_of_h_other_of_t))
                if cum_d_i_h_sp_of_t is None:
                    cum_d_i_h_sp_of_t = cum_d_i_h_sp_from_h_other
                else:
                    cum_d_i_h_sp_of_t += cum_d_i_h_sp_from_h_other
            return cum_d_i_h_sp_of_t
        elif verbose:
            print("Warning: function cum_delta_i_h_sp_of_t received no recovery_parameters. Returning 0.")
    return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)


def cum_delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_, delta_c_h_max_,
                       recovery_params_, social_protection_share_gamma_h_):
    """
    Compute the cumulative dynamic income loss
    """
    cum_d_i_h_lab_of_t = cum_delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                                delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=t_tilde_)

    cum_d_i_h_sp_of_t = cum_delta_i_h_sp_of_t(t_, recovery_parameters_=recovery_params_, productivity_pi_=productivity_pi_,
                                              social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                                              delta_tax_sp_=delta_tax_sp_)
    return cum_d_i_h_lab_of_t + cum_d_i_h_sp_of_t


def delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_, delta_c_h_max_,
                   recovery_params_, social_protection_share_gamma_h_, return_elements=False):
    """
    Compute the dynamic income loss
    """
    d_i_h_lab_of_t = delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                        delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=t_tilde_,
                                        sigma_h_=sigma_h_, delta_c_h_max_=delta_c_h_max_)
    d_i_h_sp_of_t = delta_i_h_sp_of_t(t_, recovery_params_=recovery_params_, productivity_pi_=productivity_pi_,
                                      social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                                      delta_tax_sp_=delta_tax_sp_)
    # income from post disaster support (a one-time payment) is similar to savings and not considered here
    if return_elements:
        return d_i_h_lab_of_t, d_i_h_sp_of_t
    return d_i_h_lab_of_t + d_i_h_sp_of_t


def delta_c_h_reco_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_, t_tilde_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic recovery consumption loss in the exponential recovery regime
    """
    to_float = False
    if isinstance(t_, (float, int)):
        to_float = True
        t_ = np.array([t_])
    if t_tilde_ != 0:
        limit_regime = delta_c_h_reco_of_t_limit_regime(t_[t_ < t_tilde_], delta_c_h_max_, sigma_h_, productivity_pi_,
                                                        delta_k_h_eff_, lambda_h_)
        delta_tilde_k_h_eff = delta_k_h_eff_of_t_limit_regime(t_tilde_, delta_k_h_eff_, sigma_h_, delta_c_h_max_, productivity_pi_)
        exp_regime = delta_c_h_reco_of_t_exp_regime(t_[t_ >= t_tilde_], delta_tilde_k_h_eff, lambda_h_, t_tilde_, sigma_h_)
        res = np.concatenate([limit_regime, exp_regime])
    else:
        res = delta_c_h_reco_of_t_exp_regime(t_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_)
    if to_float:
        return res.item()
    return res


def delta_c_h_reco_of_t_limit_regime(t_, delta_c_h_max_, sigma_h_, productivity_pi_, delta_k_h_eff_, lambda_h_):
    """
    Compute the dynamic recovery consumption loss in the consumption floor regime
    """

    # simplified this with constant sigma (and, thus, limit regime also for non-private reconstruction)
    # factor1 = productivity_pi_ * delta_k_h_eff_
    # factor2 = lambda_h_ * (1 - sigma_h_) / (productivity_pi_ + lambda_h_)
    # beta1 = delta_c_h_max_ - factor1 * (1 - factor2)
    # return beta1 * np.exp(productivity_pi_ * t_) - factor1 * factor2 * np.exp(-lambda_h_ * t_)

    beta = delta_c_h_max_ - productivity_pi_ * delta_k_h_eff_
    return beta * np.exp(productivity_pi_ / sigma_h_ * t_)


def delta_c_h_reco_of_t_exp_regime(t_, delta_tilde_k_h_eff_, lambda_h_, t_tilde_, sigma_h_):
    """
    Compute the dynamic recovery consumption loss in the normal exponential recovery regime
    """
    if delta_tilde_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return lambda_h_ * sigma_h_ * delta_tilde_k_h_eff_ * np.exp(-lambda_h_ * (t_ - t_tilde_))


def cum_delta_c_h_reco_of_t(t_, delta_tilde_k_h_eff_, lambda_h_, sigma_h_, t_tilde_=0):
    """
    Compute the cumulative dynamic recovery consumption loss in the normal exponential recovery regime.
    This is the integral of delta_c_h_reco_of_t_exp_regime from 0 to t
    """
    if t_tilde_ != 0:
        raise NotImplementedError("Cumulative consumption loss for reconstruction in the limit regime is not yet "
                                  "implemented.")
    if delta_tilde_k_h_eff_ == 0:
        return 0 if isinstance(t_, (float, int)) else np.zeros(t_.shape)
    return sigma_h_ * delta_tilde_k_h_eff_ * (1 - np.exp(-lambda_h_ * (t_ - t_tilde_)))


def calc_t_tilde(productivity_pi_, delta_k_h_eff_, delta_c_h_max_, sigma_h_, delta_tilde_k_h_eff_):
    """
    Compute the time at which the normal exponential recovery regime begins
    """
    assert not np.isnan(delta_c_h_max_)
    inner = productivity_pi_ * (delta_k_h_eff_ - delta_tilde_k_h_eff_) / (delta_c_h_max_ - productivity_pi_ * delta_k_h_eff_) + 1
    if inner <= 0:
        return -np.inf
    return sigma_h_ / productivity_pi_ * np.log(inner)


def calc_t_hat(lambda_h_=None, consumption_floor_xi_=None, productivity_pi_=None, delta_tax_sp_=None,
               delta_k_h_eff_=None, savings_s_h_=None, delta_i_h_pds_=None, delta_c_h_max_=None,
               recovery_params_=None, social_protection_share_gamma_h_=None, sigma_h_=None, t_tilde_=0):
    """
    Compute the time at which the consumption floor is reached
    """
    if delta_tax_sp_ == 0:
        if consumption_floor_xi_ is None or lambda_h_ is None:
            raise ValueError("Consumption floor and lambda must be provided to compute t_hat without tax.")
        return -1 / lambda_h_ * np.log(consumption_floor_xi_)
    else:
        # check whether consumption losses can be fully offset with savings
        def cum_d_c(t__):
            cum_d_i_h = cum_delta_i_h_of_t(t__, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_,
                                           sigma_h_, delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_)
            cum_d_c_h_reco = cum_delta_c_h_reco_of_t(t__, delta_k_h_eff_, lambda_h_, sigma_h_, t_tilde_)
            return cum_d_i_h + cum_d_c_h_reco
        if cum_d_c(np.inf) < savings_s_h_ + delta_i_h_pds_:
            return np.inf

        # if not, find the time at which savings are used up
        def search_func(t__):
            cum_d_c_of_t__ = cum_d_c(t__)
            delta_c_h_floor = delta_c_h_of_t(
                t__, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, 0, 0,
                delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_
            )
            return cum_d_c_of_t__ - delta_c_h_floor * t__ - (savings_s_h_ + delta_i_h_pds_)
        return optimize.root_scalar(search_func, method='brentq', bracket=[0, 100]).root


def delta_c_h_savings_pds_of_t(t_, lambda_h_, sigma_h_, productivity_pi_, consumption_floor_xi_, t_hat_,
                               delta_tilde_k_h_eff_, savings_s_h_, delta_i_h_pds_, t_tilde_, delta_i_h, delta_c_h_reco,
                               delta_c_h_max_, delta_tax_sp_, recovery_params_, social_protection_share_gamma_h_,
                               consumption_offset_):
    """
    Compute the dynamic consumption gain from savings and PDS
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
            delta_k_h_eff_=delta_tilde_k_h_eff_,
            productivity_pi_=productivity_pi_
        )
        d_c_h_savings_pds_of_t = alpha * (np.exp(-lambda_h_ * (t_ - t_tilde_)) - consumption_floor_xi_)
        if consumption_floor_xi_ == 0:
            # consumption losses can be fully offset with savings
            return d_c_h_savings_pds_of_t

    # for evaluation, including tax, social protection, and transfers
    else:
        if t_tilde_ != 0:
            raise NotImplementedError("Minimum consumption level is not yet implemented for the case with tax"
                                      "(requires a limit regime for the tax cases).")
        if t_hat_ < np.inf:
            # consumption losses cannot be fully offset with savings
            if consumption_offset_ is None:
                consumption_offset_ = delta_c_h_of_t(t_hat_, productivity_pi_, delta_tax_sp_, delta_tilde_k_h_eff_,
                                                     lambda_h_, sigma_h_, 0, 0, delta_c_h_max_,
                                                     recovery_params_, social_protection_share_gamma_h_,
                                                     consumption_floor_xi_, t_hat_, t_tilde_, delta_tilde_k_h_eff_,
                                                     return_elements=False)
            d_c_h_savings_pds_of_t = (delta_i_h + delta_c_h_reco) - consumption_offset_
        else:
            # consumption losses can be fully offset with savings
            d_c_h_savings_pds_of_t = delta_i_h + delta_c_h_reco
    if isinstance(t_, (float, int)):
        if not (t_tilde_ < t_ < t_tilde_ + t_hat_):
            d_c_h_savings_pds_of_t = 0
    else:
        d_c_h_savings_pds_of_t[t_ >= t_tilde_ + t_hat_] = 0
        d_c_h_savings_pds_of_t[t_ < t_tilde_] = 0
    return d_c_h_savings_pds_of_t


def delta_c_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                   savings_s_h_, delta_i_h_pds_, delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_,
                   consumption_floor_xi_=None, t_hat=None, t_tilde=None, delta_tilde_k_h_eff=None,
                   consumption_offset=None, return_elements=False):
    """
    Compute the dynamic consumption loss
    """
    # no income loss from labor or social protection and transfers --> baseline case
    if delta_k_h_eff_ == 0 and (recovery_params_ is None or np.all([rp[0] == 0 for rp in recovery_params_])):
        if return_elements:
            return t_ * 0, t_ * 0, t_ * 0, t_ * 0
        return t_ * 0

    if consumption_floor_xi_ == t_hat == t_tilde == delta_tilde_k_h_eff is None:
        consumption_floor_xi_, t_hat, t_tilde, delta_tilde_k_h_eff = solve_consumption_floor_xi(
            lambda_h_=lambda_h_,
            sigma_h_=sigma_h_,
            delta_k_h_eff_=delta_k_h_eff_,
            productivity_pi_=productivity_pi_,
            savings_s_h_=savings_s_h_,
            delta_i_h_pds_=delta_i_h_pds_,
            delta_c_h_max_=delta_c_h_max_,
            delta_tax_sp_=delta_tax_sp_,
            recovery_params_=recovery_params_,
            social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        )
    elif np.sum([v is None for v in [consumption_floor_xi_, t_hat, t_tilde, delta_tilde_k_h_eff]]) > 0:
        raise ValueError("Must pass all or none of consumption floor xi, t_hat, t_tilde, and delta_tilde_k_h_eff.")
    d_i_h_lab_of_t, d_i_h_sp_of_t = delta_i_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        t_tilde_=t_tilde,
        sigma_h_=sigma_h_,
        delta_c_h_max_=delta_c_h_max_,
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
        t_tilde_=t_tilde,
        delta_c_h_max_=delta_c_h_max_,
        productivity_pi_=productivity_pi_
    )
    delta_c_h_savings_pds = delta_c_h_savings_pds_of_t(
        t_=t_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        productivity_pi_=productivity_pi_,
        delta_tilde_k_h_eff_=delta_tilde_k_h_eff,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        t_hat_=t_hat,
        consumption_floor_xi_=consumption_floor_xi_,
        t_tilde_=t_tilde,
        delta_i_h=delta_i_h,
        delta_c_h_reco=delta_c_h_reco,
        delta_c_h_max_=delta_c_h_max_,
        delta_tax_sp_=delta_tax_sp_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        consumption_offset_=consumption_offset,
    )
    if return_elements:
        return d_i_h_lab_of_t, d_i_h_sp_of_t, delta_c_h_reco, delta_c_h_savings_pds
    return delta_i_h + delta_c_h_reco - delta_c_h_savings_pds


def baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share):
    """
    Compute the baseline consumption level
    """
    return productivity_pi_ * k_h_eff_ * (1 - delta_tax_sp_) / (1 - diversified_share)


def consumption_c_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_,
                       delta_i_h_pds_, k_h_eff_, delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_,
                       diversified_share_, consumption_floor_xi_=None, t_hat=None, t_tilde=None,
                       delta_tilde_k_h_eff=None, consumption_offset=None, include_tax=False):
    """
    Compute the dynamic consumption level
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
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=recovery_params_ if include_tax else [(0, 0)],
        social_protection_share_gamma_h_=social_protection_share_gamma_h_ if include_tax else 0,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        t_tilde=t_tilde,
        delta_tilde_k_h_eff=delta_tilde_k_h_eff,
        consumption_offset=consumption_offset,
    )
    baseline_consumption = baseline_consumption_c_h(
        productivity_pi_=productivity_pi_,
        k_h_eff_=k_h_eff_,
        delta_tax_sp_=delta_tax_sp_,
        diversified_share=diversified_share_,
    )
    return baseline_consumption - consumption_loss


def welfare_of_c(c_, eta_):
    """
    The welfare function
    """
    if (np.array(c_) <= 0).any():
        return -np.inf
    return (c_ ** (1 - eta_) - 1) / (1 - eta_)


def welfare_w_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                   sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_,
                   recovery_params_, social_protection_share_gamma_h_, diversified_share_,
                   consumption_floor_xi_=None, t_hat=None, t_tilde=None, delta_tilde_k_h_eff=None,
                   consumption_offset=None, discount=True, include_tax=False):
    """
    Compute the discounted time-dependent welfare
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
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        t_tilde=t_tilde,
        delta_tilde_k_h_eff=delta_tilde_k_h_eff,
        consumption_offset=consumption_offset,
        include_tax=include_tax,
    )
    assert (np.array(c_of_t) > 0).all()
    res = welfare_of_c(c_of_t, eta_=eta_)
    if discount:
        res = res * np.exp(-discount_rate_rho_ * t_)
    return res


def discounted_consumption_c_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                                  sigma_h_, savings_s_h_, delta_i_h_pds_, _, k_h_eff_, delta_c_h_max_, recovery_params_,
                                  social_protection_share_gamma_h_, diversified_share_, consumption_floor_xi_=None,
                                  t_hat=None, t_tilde=None, delta_tilde_k_h_eff=None, consumption_offset=None,
                                  include_tax=False):
    """ Compute the discounted consumption level """
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
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        t_tilde=t_tilde,
        delta_tilde_k_h_eff=delta_tilde_k_h_eff,
        consumption_offset=consumption_offset,
        include_tax=include_tax,
    )
    return c_of_t * np.exp(-discount_rate_rho_ * t_)


def aggregate_welfare_w_of_c_of_capital_t(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_,
                                          delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_,
                                          eta_, k_h_eff_, delta_c_h_max_, recovery_params_,
                                          social_protection_share_gamma_h_, diversified_share_,
                                          consumption_floor_xi_=None, t_hat=None, t_tilde=None,
                                          delta_tilde_k_h_eff=None, consumption_offset=None, agg_approach='PHL_paper',
                                          include_tax=False):
    """
    Compute the time-aggregate welfare function
    """
    args = (discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
            sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_, recovery_params_,
            social_protection_share_gamma_h_, diversified_share_, consumption_floor_xi_, t_hat, t_tilde,
            delta_tilde_k_h_eff, consumption_offset, True, include_tax)
    if agg_approach == 'PHL_paper':
        # aggregate welfare approach taken in the Philippines paper
        w_agg = integrate.quad(welfare_w_of_t, 0, capital_t_, args=args, limit=50)[0]
    elif agg_approach == 'UB_technical_paper':
        # aggregate welfare approach taken in the original Unbreakable technical paper
        w_agg = welfare_of_c(integrate.quad(discounted_consumption_c_of_t, 0, capital_t_, args=args, limit=50)[0], eta_)
    else:
        raise ValueError(f"Unknown aggregation approach: {agg_approach}")
    return w_agg


def recompute_with_tax(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                       lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_,
                       diversified_share_, recovery_params_, social_protection_share_gamma_h_):
    consumption_floor_xi_, t_hat, t_tilde, delta_tilde_k_h_eff = solve_consumption_floor_xi(
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        delta_c_h_max_=delta_c_h_max_,
        delta_tax_sp_=delta_tax_sp_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
    )

    if t_hat < np.inf:
        consumption_offset = delta_c_h_of_t(t_hat, productivity_pi_, delta_tax_sp_, delta_tilde_k_h_eff,
                                             lambda_h_, sigma_h_, 0, 0, delta_c_h_max_,
                                             recovery_params_, social_protection_share_gamma_h_, consumption_floor_xi_,
                                             t_hat, t_tilde, delta_tilde_k_h_eff, return_elements=False)
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
            delta_c_h_max_=delta_c_h_max_,
            recovery_params_=recovery_params_,
            social_protection_share_gamma_h_=social_protection_share_gamma_h_,
            return_elements=True,
            consumption_floor_xi_=consumption_floor_xi_,
            t_hat=t_hat,
            t_tilde=t_tilde,
            delta_tilde_k_h_eff=delta_tilde_k_h_eff,
            consumption_offset=consumption_offset,
        )[3]
    used_liquidity = integrate.quad(used_liquidity_func, 0, capital_t_, limit=50)[0]

    w_baseline = aggregate_welfare_w_of_c_of_capital_t(
        capital_t_=capital_t_,
        discount_rate_rho_=discount_rate_rho_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=0,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        eta_=eta_,
        k_h_eff_=k_h_eff_,
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=[(0, 0)],
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        agg_approach='PHL_paper',
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        t_tilde=t_tilde,
        delta_tilde_k_h_eff=delta_tilde_k_h_eff,
        consumption_offset=consumption_offset,
        include_tax=True
    )
    w_disaster = aggregate_welfare_w_of_c_of_capital_t(
        capital_t_=capital_t_,
        discount_rate_rho_=discount_rate_rho_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        eta_=eta_,
        k_h_eff_=k_h_eff_,
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        agg_approach='PHL_paper',
        diversified_share_=diversified_share_,
        consumption_floor_xi_=consumption_floor_xi_,
        t_hat=t_hat,
        t_tilde=t_tilde,
        delta_tilde_k_h_eff=delta_tilde_k_h_eff,
        consumption_offset=consumption_offset,
        include_tax=True
    )

    return w_baseline - w_disaster, used_liquidity


def recompute_with_tax_wrapper(recompute_args):
    """
    Wrapper for recompute_with_tax
    """
    index, row = recompute_args
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
            delta_c_h_max_=row['delta_c_h_max'],
            diversified_share_=row['diversified_share'],
            recovery_params_=row['recovery_params'],
            social_protection_share_gamma_h_=row['social_protection_share_gamma_h'],
        )
    except Exception as e:
        print(f"Error in row {index}: {e}")
        raise e


def recompute_data_with_tax(df_in, num_cores=None):
    """
    Recompute the final change in welfare, including tax, social protection, and transfers.
    """
    with multiprocessing.Pool(processes=num_cores) as pool:
        res = list(tqdm.tqdm(pool.imap(recompute_with_tax_wrapper, df_in.iterrows()), total=len(df_in),
                             desc='Recomputing actual welfare loss and used liquidity'))
    res = pd.DataFrame(res, columns=['dW_reco', 'dS_reco_PDS'], index=df_in.index)
    return res


def compute_delta_welfare_dw_reco(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                                  lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_,
                                  diversified_share_, recovery_params_, social_protection_share_gamma_h_):
    """
    Recompute the final change in welfare, including tax, social protection, and transfers.
    """
    w_baseline = aggregate_welfare_w_of_c_of_capital_t(
        capital_t_=capital_t_,
        discount_rate_rho_=discount_rate_rho_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=0,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        eta_=eta_,
        k_h_eff_=k_h_eff_,
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=[(0, 0)],
        social_protection_share_gamma_h_=None,
        agg_approach='PHL_paper',
        diversified_share_=diversified_share_,
        include_tax=True
    )
    w_disaster = aggregate_welfare_w_of_c_of_capital_t(
        capital_t_=capital_t_,
        discount_rate_rho_=discount_rate_rho_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        eta_=eta_,
        k_h_eff_=k_h_eff_,
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=recovery_params_,
        social_protection_share_gamma_h_=social_protection_share_gamma_h_,
        agg_approach='PHL_paper',
        diversified_share_=diversified_share_,
        include_tax=True
    )

    return w_baseline - w_disaster


def calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_):
    """
    Compute the alpha parameter
    """
    return (productivity_pi_ + lambda_h_ * sigma_h_) * delta_k_h_eff_


def solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                               delta_i_h_pds_, delta_c_h_max_, delta_tax_sp_, recovery_params_,
                               social_protection_share_gamma_h_):
    """
    Solve for the consumption floor xi
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
        if not np.isnan(xi_res):
            resulting_max_delta_c_h = alpha * xi_res
        else:
            resulting_max_delta_c_h = alpha

        if not np.isnan(delta_c_h_max_):
            if resulting_max_delta_c_h > delta_c_h_max_ and savings_s_h_ + delta_i_h_pds_ > 0:
                # consumption losses exceed max cons losses at t=0 and savings are available to offset consumption losses
                # need to recompute xi_res
                def xi_func(xi_):
                    rhs = 1
                    lhs = xi_ * (1 + lambda_h_ * (savings_s_h_ + delta_i_h_pds_) / delta_c_h_max_ - np.log(xi_)) if xi_ > 0 else 0
                    return lhs - rhs
                xi_res = optimize.brentq(xi_func, 0, 1)
                t_hat = calc_t_hat(lambda_h_=lambda_h_, consumption_floor_xi_=xi_res, delta_tax_sp_=delta_tax_sp_)
                delta_tilde_k_h_eff = delta_c_h_max_ / ((productivity_pi_ + lambda_h_ * sigma_h_) * xi_res)
            elif resulting_max_delta_c_h > delta_c_h_max_ and savings_s_h_ + delta_i_h_pds_ <= 0:
                # consumption losses exceed max cons losses at t=0 and no savings are available to offset consumption losses
                delta_tilde_k_h_eff = delta_c_h_max_ / (productivity_pi_ + lambda_h_ * sigma_h_)
            else:
                # consumption losses are below max cons losses at t=0
                delta_tilde_k_h_eff = delta_k_h_eff_
            t_tilde = calc_t_tilde(productivity_pi_, delta_k_h_eff_, delta_c_h_max_, sigma_h_, delta_tilde_k_h_eff)
        else:
            t_tilde, delta_tilde_k_h_eff = 0, delta_k_h_eff_

    # for the recomputation, including tax, social protection, and transfers
    else:
        if not np.isnan(delta_c_h_max_):
            raise ValueError("Maximum consumption loss is not yet supported for the case without tax.")
        else:
            t_tilde, delta_tilde_k_h_eff = 0, delta_k_h_eff_
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
                    delta_c_h_max_=delta_c_h_max_,
                    recovery_params_=recovery_params_,
                    social_protection_share_gamma_h_=social_protection_share_gamma_h_,
                    sigma_h_=sigma_h_,
                    t_tilde_=t_tilde,
                )
                # with tax, xi_res has no meaning (as alpha would also depend on tax and income losses from transfers)
                xi_res = np.nan
    return xi_res, t_hat, t_tilde, delta_tilde_k_h_eff


def calc_leftover_savings(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_):
    """
    Compute the leftover savings
    """
    alpha = calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_)
    return max(0, savings_s_h_ + delta_i_h_pds_ - alpha / lambda_h_)


def objective_func(lambda_h_, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                   delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_, delta_c_h_max_, delta_tax_sp_, diversified_share_):
    """
    Objective function to be minimized
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
        delta_c_h_max_=delta_c_h_max_,
        recovery_params_=[(0, 0)],  # for optimization, no recovery parameters are needed
        social_protection_share_gamma_h_=0,  # for optimization, no social protection share is needed
        diversified_share_=diversified_share_,
        include_tax=False
    )

    # for the optimization, include leftover savings, s.th. not the first but the fastest recovery path is chosen where
    # consumption losses can be fully offset
    objective += calc_leftover_savings(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                       delta_i_h_pds_)
    return -objective  # negative to transform into a minimization problem


def calc_lambda_bounds_for_optimization(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                        delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_, delta_c_h_max_,
                                        delta_tax_sp_, diversified_share_, min_lambda_, max_lambda_):
    """
    Compute the bounds for the lambda parameter
    """

    assert min_lambda_ >= 0

    # if no maximum consumption loss is defined (and thus, the option to wait until t_tilde when capital loss has
    # decreased to delta_tilde_k_h_eff), find the maximal allowable lambda that maintains positive consumption
    if np.isnan(delta_c_h_max_):
        c_baseline = baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share_)
        if savings_s_h_ + delta_i_h_pds_ <= 0:
            if c_baseline < calc_alpha(0, sigma_h_, delta_k_h_eff_, productivity_pi_):
                # there is no lambda that maintains positive consumption
                print(f"Optimization not possible for capital_t={capital_t_}, sigma_h={sigma_h_}, "
                      f"delta_k_h_eff={delta_k_h_eff_}, productivity_pi={productivity_pi_}, savings_s_h={savings_s_h_}, "
                      f"delta_i_h_pds={delta_i_h_pds_}, eta={eta_}, discount_rate_rho={discount_rate_rho_}, "
                      f"k_h_eff={k_h_eff_}, delta_c_h_max={delta_c_h_max_}, delta_tax_sp={delta_tax_sp_}, "
                      f"diversified_share={diversified_share_}. No lambda exists that maintains positive consumption.")
                return np.nan, np.nan, np.nan
            # max_lambda_ = min(max_lambda_, productivity_pi_ / sigma_h_ * (k_h_eff_ / delta_k_h_eff_ - 1))
            max_lambda_ = min(max_lambda_, 1 / sigma_h_ * (c_baseline / delta_k_h_eff_ - productivity_pi_))
        elif savings_s_h_ + delta_i_h_pds_ > 0:
            def lambda_func(lambda_h_):
                alpha_ = calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_)
                xi_, _, _, _ = solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_,
                                                          savings_s_h_, delta_i_h_pds_, delta_c_h_max_, 0, None, 0)
                return alpha_ * xi_ - (c_baseline - 1e-5)  # alpha * xi must be smaller than the baseline consumption

            if np.sign(lambda_func(min_lambda_)) != np.sign(lambda_func(max_lambda_)):
                opt_lambda = optimize.brentq(lambda_func, min_lambda_, max_lambda_)
                if lambda_func(opt_lambda - 1e-5) < lambda_func(opt_lambda + 1e-5):
                    max_lambda_ = min(max_lambda_, opt_lambda)
                else:
                    min_lambda_ = max(min_lambda_, opt_lambda)
            elif lambda_func(1e-5) > 0:
                # there is no lambda that maintains positive consumption
                print(f"Optimization not possible for capital_t={capital_t_}, sigma_h={sigma_h_}, "
                      f"delta_k_h_eff={delta_k_h_eff_}, productivity_pi={productivity_pi_}, savings_s_h={savings_s_h_}, "
                      f"delta_i_h_pds={delta_i_h_pds_}, eta={eta_}, discount_rate_rho={discount_rate_rho_}, "
                      f"k_h_eff={k_h_eff_}, delta_c_h_max={delta_c_h_max_}, delta_tax_sp={delta_tax_sp_}, "
                      f"diversified_share={diversified_share_}. No lambda exists that maintains positive consumption.")
                return np.nan, np.nan, np.nan

    # check whether some lambda exists, such that consumption losses can be fully offset
    if (savings_s_h_ + delta_i_h_pds_) > sigma_h_ * delta_k_h_eff_:
        lambda_full_offset = 1 / ((savings_s_h_ + delta_i_h_pds_) / delta_k_h_eff_ - sigma_h_) * productivity_pi_
        if lambda_full_offset < max_lambda_:
            min_lambda_ = max(min_lambda_, lambda_full_offset)
            # print("min_lambda_ adjusted to:", min_lambda_)

    init_candidates = np.linspace(min_lambda_ + .2 * (max_lambda_ - min_lambda_),
                                  max_lambda_ - .2 * (max_lambda_ - min_lambda_), 10)
    best_lambda_init = None
    best_candicate_objective = None
    for ic in init_candidates:
        objective = objective_func(ic, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                                   delta_i_h_pds_, eta_, discount_rate_rho_, k_h_eff_, delta_c_h_max_, delta_tax_sp_,
                                   diversified_share_)
        if best_lambda_init is None or objective < best_candicate_objective:
            best_lambda_init = ic
            best_candicate_objective = objective

    return min_lambda_, max_lambda_, best_lambda_init


def optimize_lambda(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
                    discount_rate_rho_, k_h_eff_, delta_c_h_max_, delta_tax_sp_, diversified_share_,
                    tolerance=1e-10, min_lambda=0.05, max_lambda=100):
    """
    Optimize the lambda parameter
    """

    min_lambda, max_lambda, lambda_h_init = calc_lambda_bounds_for_optimization(
        capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_, discount_rate_rho_,
        k_h_eff_, delta_c_h_max_, delta_tax_sp_, diversified_share_, min_lambda, max_lambda
    )

    if np.isnan(min_lambda) or np.isnan(max_lambda) or np.isnan(lambda_h_init):
        return [np.nan]

    res = optimize.minimize(
        fun=objective_func,
        x0=np.array(lambda_h_init),
        bounds=[(min_lambda, max_lambda)],
        args=(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
              discount_rate_rho_, k_h_eff_, delta_c_h_max_, delta_tax_sp_, diversified_share_),
        method='Nelder-Mead',
        tol=tolerance,
    )

    return res.x


def optimize_lambda_wrapper(opt_args, min_lambda, max_lambda):
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
            delta_c_h_max_=row['delta_c_h_max'],
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
    Optimize the lambda parameter for each row in the dataframe
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


def run_experiment(sigma_h, delta_c_h_max, tolerance=1e-2, index=None,
                   capital_t=50, min_lambda=1e-10, max_lambda=6, liquidity_='data:average', ew_vul_reduction=0.2):
    vulnerability = pd.read_csv("./intermediate/scenarios/baseline/scenario__vulnerability_unadjusted.csv",
                                index_col=['iso3', 'hazard', 'income_cat'])
    cat_info = pd.read_csv("./intermediate/scenarios/baseline/scenario__cat_info.csv",
                           index_col=['iso3', 'income_cat'])
    macro = pd.read_csv("./intermediate/scenarios/baseline/scenario__macro.csv", index_col=['iso3'])
    data = pd.merge(vulnerability, cat_info, left_index=True, right_index=True)
    data = pd.merge(data, macro, left_index=True, right_index=True)
    data.rename(columns={'avg_prod_k': 'productivity_pi', 'income_elasticity_eta': 'eta', 'rho': 'discount_rate_rho',
                         'k': 'k_h_eff', 'recovery_share_sigma': 'sigma_h'}, inplace=True)

    data['capital_t'] = capital_t
    data['delta_tax_sp'] = 0  # neglect social protection and tax for optimization
    data['v_ew'] = data["v"] * (1 - ew_vul_reduction * data["ew"])
    data['delta_k_h_eff'] = data['v_ew'] * data['k_h_eff']
    data['delta_i_h_pds'] = 0

    if delta_c_h_max == 'None':
        data['delta_c_h_max'] = np.nan
    elif 'factor' in delta_c_h_max:
        data['delta_c_h_max'] = float(delta_c_h_max.replace('factor:', '')) * data['k_h_eff'] * data['productivity_pi']
    else:
        raise ValueError("Invalid delta_c_h_max parameter. Choose from 'None' or 'factor{value}'.")

    if sigma_h == 'data':
        data['sigma_h'] = data['k_household_share']
    elif 'factor' in sigma_h:
        data['sigma_h'] = float(sigma_h.replace('factor:', '')) * data['delta_k_h_eff']
    else:
        raise ValueError("Invalid sigma_h parameter. Choose from 'data' or 'factor{value}'.")

    if liquidity_ == 'data':
        data['savings_s_h'] = data.liquidity
    elif 'factor' in liquidity_:
        data['savings_s_h'] = float(liquidity_.replace('factor:', '')) * data['sigma_h'] * data['delta_k_h_eff']
    else:
        raise ValueError("Invalid liquidity inclusion option. Choose from 'liquid', 'illiquid', 'average'.")

    opt_data = data[['capital_t', 'sigma_h', 'delta_k_h_eff', 'productivity_pi', 'savings_s_h', 'delta_i_h_pds', 'eta',
                     'delta_tax_sp', 'discount_rate_rho', 'k_h_eff', 'delta_c_h_max']]
    if index:
        opt_data = opt_data.loc[[index]]
    recovery_rates = optimize_data(opt_data, tolerance=tolerance, min_lambda=min_lambda, max_lambda=max_lambda)
    result = pd.merge(opt_data, recovery_rates, left_index=True, right_index=True)
    result[['consumption_floor_xi', 't_hat', 't_tilde', 'delta_tilde_k_h_eff']] = result.apply(
        lambda x: pd.Series(solve_consumption_floor_xi(
            lambda_h_=x['lambda_h'],
            sigma_h_=x['sigma_h'],
            delta_k_h_eff_=x['delta_k_h_eff'],
            productivity_pi_=x['productivity_pi'],
            savings_s_h_=x['savings_s_h'],
            delta_i_h_pds_=x['delta_i_h_pds'],
            delta_c_h_max_=x['delta_c_h_max_adjusted'] if 'delta_c_h_max_adjusted' in x.index else x['delta_c_h_max']
        )), axis=1)
    return result


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--run', action='store_true')
    args.add_argument('--sigma_h', type=str, default='data:prv')
    args.add_argument('--delta_c_h_max', type=str, default='None')
    args.add_argument('--tolerance', type=float, default=1e-2)
    args.add_argument('--idx', type=str, default=None)
    args.add_argument('--capital_t', type=int, default=50)
    args.add_argument('--no_output', action='store_true')
    args.add_argument('--liquidity', type=str, default='data:average')
    args = args.parse_args()
    idx = args.idx
    if idx is not None:
        idx = tuple(args.idx.split('_'))

    if args.run:
        results = run_experiment(args.sigma_h, args.delta_c_h_max, args.tolerance, idx,
                                 args.capital_t, liquidity_=args.liquidity)

        if not args.no_output and idx is None:
            results.to_csv(f"./optimization_experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M')}__"
                           f"sigma_h_{args.sigma_h.replace(':', '-')}__delta_c_h_max{args.delta_c_h_max.replace(':', '-')}"
                           f"__tolerance_{args.tolerance}__capital_t_{args.capital_t}"
                           f"__liquidity_{args.liquidity.replace(':', '-')}.csv")
        elif not args.no_output:
            print("No output file created. Please remove the index selection to save the results.")
