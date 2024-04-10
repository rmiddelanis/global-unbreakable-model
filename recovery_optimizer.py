import argparse
import multiprocessing
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from scipy import integrate, optimize

import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('error')

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


def delta_k_h_nonprv_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_):
    """
    Compute the dynamic non-private capital loss
    """
    return (1 - sigma_h_) * delta_k_h_eff_ * np.exp(-lambda_h_ * t_)


def delta_k_h_prv_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_, t_tilde_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic private capital loss
    """
    to_float = False
    if isinstance(t_, (float, int)):
        to_float = True
        t_ = np.array([t_])
    if t_tilde_ != 0:
        limit_regime = delta_k_h_prv_of_t_limit_regime(t_[t_ < t_tilde_], delta_k_h_eff_, lambda_h_, sigma_h_,
                                                       delta_c_h_max_, productivity_pi_)
        exp_regime = delta_k_h_prv_of_t_exp_regime(t_[t_ >= t_tilde_], delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_,
                                                   delta_c_h_max_, productivity_pi_)
        res = np.concatenate([limit_regime, exp_regime])
    else:
        res = delta_k_h_prv_of_t_exp_regime(t_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_, delta_c_h_max_,
                                            productivity_pi_)
    if to_float:
        return res.item()
    else:
        return res


def delta_k_h_prv_of_t_limit_regime(t_, delta_k_h_eff_, lambda_h_, sigma_h_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic private capital loss
    """
    return sigma_h_ * delta_k_h_eff_ - cum_delta_c_h_reco_of_t_limit_regime(t_, delta_c_h_max_, sigma_h_,
                                                                            productivity_pi_, delta_k_h_eff_, lambda_h_)


def delta_k_h_prv_of_t_exp_regime(t_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_, delta_c_h_max_, productivity_pi_):
    """
    Compute the dynamic private capital loss in the normal exponential recovery regime
    """
    delta_tilde_k_h_prv = calc_delta_tilde_k_h_prv(delta_k_h_eff_, sigma_h_, t_tilde_, delta_c_h_max_, productivity_pi_,
                                                   lambda_h_)
    return delta_tilde_k_h_prv * np.exp(-lambda_h_ * (t_ - t_tilde_))



# TODO: delta_tax_sp
def delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_,
                       delta_c_h_max_):
    """
    Compute dynamic the labor income loss
    """
    dk_eff_of_t = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=t_tilde_,
                                     sigma_h_=sigma_h_, delta_c_h_max_=delta_c_h_max_, productivity_pi_=productivity_pi_)
    return dk_eff_of_t * productivity_pi_ * (1 - delta_tax_sp_)


# TODO: for now, neglect social protection in optimization
def delta_i_h_sp_of_t(t_):
    return 0


def delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_, delta_c_h_max_):
    """
    Compute the dynamic income loss
    """
    d_i_h_lab_of_t = delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                        delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_, t_tilde_=t_tilde_,
                                        sigma_h_=sigma_h_, delta_c_h_max_=delta_c_h_max_)
    d_i_h_sp_of_t = delta_i_h_sp_of_t(t_)
    # income from post disaster support (considered a one-time payment) is considered savings
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
        # delta_tilde_k_h_prv = calc_delta_tilde_k_h_prv(delta_k_h_eff_, sigma_h_, t_tilde_, delta_c_h_max_,
        #                                                productivity_pi_, lambda_h_)
        delta_tilde_k_h_eff = delta_k_h_eff_of_t_limit_regime(t_tilde_, delta_k_h_eff_, sigma_h_, delta_c_h_max_, productivity_pi_)
        exp_regime = delta_c_h_reco_of_t_exp_regime(t_[t_ >= t_tilde_], delta_tilde_k_h_eff, lambda_h_, t_tilde_, sigma_h_)
        res = np.concatenate([limit_regime, exp_regime])
    else:
        res = delta_c_h_reco_of_t_exp_regime(t_, delta_k_h_eff_, lambda_h_, t_tilde_, sigma_h_)
    if to_float:
        return res.item()
    else:
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


def cum_delta_c_h_reco_of_t_limit_regime(t_, delta_c_h_max_, sigma_h_, productivity_pi_, delta_k_h_eff_, lambda_h_):
    """
    Compute the cumulative dynamic recovery consumption loss in the consumption floor regime.
    This is the integral of delta_c_h_reco_of_t_limit_regime from 0 to t
    """

    # simplified this with constant sigma (and, thus, limit regime also for non-private reconstruction)
    # factor1 = productivity_pi_ * delta_k_h_eff_
    # factor2 = (1 - sigma_h_) / (productivity_pi_ + lambda_h_)
    # beta1 = delta_c_h_max_ - factor1 * (1 - lambda_h_ * factor2)
    # beta2 = 0
    # part1 = beta1 / productivity_pi_ * (np.exp(productivity_pi_ * t_) - 1)
    # part2 = factor1 * factor2 * (np.exp(-lambda_h_ * t_) - 1) + beta2
    # return part1 + part2




def calc_delta_tilde_k_h_prv(delta_k_h_eff_, sigma_h_, t_tilde_, delta_c_h_max_, productivity_pi_, lambda_h_):
    if t_tilde_ == 0:
        return delta_k_h_eff_ * sigma_h_
    return delta_k_h_eff_ * sigma_h_ - cum_delta_c_h_reco_of_t_limit_regime(t_tilde_, delta_c_h_max_, sigma_h_,
                                                                            productivity_pi_, delta_k_h_eff_, lambda_h_)


def delta_c_h_reco_of_t_exp_regime(t_, delta_tilde_k_h_eff_, lambda_h_, t_tilde_, sigma_h_):
    """
    Compute the dynamic recovery consumption loss in the normal exponential recovery regime
    """
    # return lambda_h_ * delta_tilde_k_h_prv_ * np.exp(-lambda_h_ * (t_ - t_tilde_))
    return lambda_h_ * sigma_h_ * delta_tilde_k_h_eff_ * np.exp(-lambda_h_ * (t_ - t_tilde_))


def calc_t_tilde(productivity_pi_, delta_k_h_eff_, delta_c_h_max_, sigma_h_, delta_tilde_k_h_eff_):
    """
    Compute the time at which the normal exponential recovery regime begins
    """
    assert not np.isnan(delta_c_h_max_)
    inner = productivity_pi_ * (delta_k_h_eff_ - delta_tilde_k_h_eff_) / (delta_c_h_max_ - productivity_pi_ * delta_k_h_eff_) + 1
    if inner <= 0:
        return -np.inf
    return sigma_h_ / productivity_pi_ * np.log(inner)


def calc_t_hat(lambda_h_, consumption_floor_xi_):
    """
    Compute the time at which the consumption floor is reached
    """
    return - 1 / lambda_h_ * np.log(consumption_floor_xi_)



def delta_c_h_savings_pds_of_t(t_, lambda_h_, sigma_h_, productivity_pi_, consumption_floor_xi_, t_hat_,
                               delta_tilde_k_h_eff_, savings_s_h_, delta_i_h_pds_, t_tilde_):
    """
    Compute the dynamic consumption gain from savings and PDS
    """
    if savings_s_h_ + delta_i_h_pds_ <= 0:
        res = np.zeros_like(t_)
        if res.shape == ():
            return 0
        return res
    if consumption_floor_xi_ == np.nan:
        print("Warning. Xi is nan.")
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

    if isinstance(t_, (float, int)):
        if not (t_tilde_ < t_ < t_tilde_ + t_hat_):
            d_c_h_savings_pds_of_t = 0
    else:
        d_c_h_savings_pds_of_t[t_ >= t_tilde_ + t_hat_] = 0
        d_c_h_savings_pds_of_t[t_ < t_tilde_] = 0
    return d_c_h_savings_pds_of_t


# TODO: delta_tax_sp
def delta_c_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                   savings_s_h_, delta_i_h_pds_, delta_c_h_max_, return_elements=False):
    """
    Compute the dynamic consumption loss
    """
    consumption_floor_xi_, t_hat, t_tilde, delta_tilde_k_h_eff = solve_consumption_floor_xi(
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        delta_c_h_max_=delta_c_h_max_
    )
    delta_i_h = delta_i_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        t_tilde_=t_tilde,
        sigma_h_=sigma_h_,
        delta_c_h_max_=delta_c_h_max_
    )
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
        t_tilde_=t_tilde
    )
    if return_elements:
        return delta_i_h, delta_c_h_reco, delta_c_h_savings_pds
    return delta_i_h + delta_c_h_reco - delta_c_h_savings_pds


def baseline_consumption_c_h(productivity_pi_, k_h_eff_):
    """
    Compute the baseline consumption level
    """
    return productivity_pi_ * k_h_eff_


def consumption_c_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                       savings_s_h_, delta_i_h_pds_, k_h_eff_, delta_c_h_max_):
    """
    Compute the dynamic consumption level
    """
    consumption_loss = delta_c_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_,
        delta_c_h_max_=delta_c_h_max_
    )
    baseline_consumption = baseline_consumption_c_h(
        productivity_pi_=productivity_pi_,
        k_h_eff_=k_h_eff_
    )
    return baseline_consumption - consumption_loss


def welfare_of_c(c_, eta_):
    """
    The welfare function
    """
    if (np.array(c_) <= 0).any():
        return -np.inf
    return (c_ ** (1 - eta_) - 1) / (1 - eta_)


def discounted_welfare_w_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                              sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_):
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
        delta_c_h_max_=delta_c_h_max_
    )
    assert (np.array(c_of_t) >= 0).all()
    return welfare_of_c(c_of_t, eta_=eta_) * np.exp(-discount_rate_rho_ * t_)


def aggregate_welfare_w_of_c_of_capital_t(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_,
                                          delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_,
                                          k_h_eff_, delta_c_h_max_):
    """
    Compute the time-aggregate welfare function
    """
    args = (discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
            sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_)
    integral = integrate.quad(discounted_welfare_w_of_t, 0, capital_t_, args=args, limit=50)[0]
    return integral


def calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_):
    """
    Compute the alpha parameter
    """
    return (productivity_pi_ + lambda_h_ * sigma_h_) * delta_k_h_eff_


def solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                               delta_i_h_pds_, delta_c_h_max_):
    """
    Solve for the consumption floor xi
    @param delta_c_h_max_:
    @param lambda_h_:
    @param sigma_h_:
    @param delta_k_h_eff_:
    @param productivity_pi_:
    @param savings_s_h_:
    @param delta_i_h_pds_:
    @return:
    """
    alpha = calc_alpha(
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_
    )

    if lambda_h_ <= 0 or savings_s_h_ + delta_i_h_pds_ <= 0:
        xi_res, t_hat = np.nan, 0  # No savings available to offset consumption losses
    elif savings_s_h_ + delta_i_h_pds_ >= alpha / lambda_h_:
        xi_res, t_hat = 0, 0  # All capital losses can be repaired at time 0, and consumption loss can be fduced to 0
    else:
        # Solve numerically for xi
        def xi_func(xi_):
            rhs = 1 - lambda_h_ / alpha * (savings_s_h_ + delta_i_h_pds_)
            lhs = xi_ * (1 - np.log(xi_)) if xi_ > 0 else 0
            return lhs - rhs
        xi_res = optimize.brentq(xi_func, 0, 1)
        t_hat = calc_t_hat(lambda_h_, xi_res)
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
            t_hat = calc_t_hat(lambda_h_, xi_res)
            delta_tilde_k_h_eff = delta_c_h_max_ / ((productivity_pi_ + lambda_h_ * sigma_h_) * xi_res)
        elif resulting_max_delta_c_h > delta_c_h_max_ and savings_s_h_ + delta_i_h_pds_ <= 0:
            # consumption losses exceed max cons losses at t=0 and no savings are available to offset consumption losses
            delta_tilde_k_h_eff = delta_c_h_max_ / (productivity_pi_ + lambda_h_ * sigma_h_)
        else:
            # consumption losses are below max cons losses at t=0
            delta_tilde_k_h_eff = delta_k_h_eff_
        t_tilde = calc_t_tilde(productivity_pi_, delta_k_h_eff_, delta_c_h_max_, sigma_h_, delta_tilde_k_h_eff)
    else:
        t_tilde = 0
        delta_tilde_k_h_eff = delta_k_h_eff_
    return xi_res, t_hat, t_tilde, delta_tilde_k_h_eff


def objective_func(lambda_h_, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                   delta_i_h_pds_, eta_, delta_tax_sp_, discount_rate_rho_, k_h_eff_, delta_c_h_max_):
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
        delta_c_h_max_=delta_c_h_max_
    )
    return -objective  # negative to transform into a minimization problem


def optimize_lambda(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
                    discount_rate_rho_, k_h_eff_, delta_c_h_max_, lambda_h_init, delta_tax_sp_=0,
                    obj_func=objective_func, tolerance=1e-10, min_lambda=0.05, max_lambda=100):
    """
    Optimize the lambda parameter
    """
    # if no maximum consumption loss is defined (and thus, the option to wait until t_tilde when capital loss has
    # decreased to delta_tilde_k_h_eff, find the maximal allowable lambda that maintains positive consumption
    if np.isnan(delta_c_h_max_):
        if savings_s_h_ + delta_i_h_pds_ <= 0:
            max_lambda = productivity_pi_ / sigma_h_ * (k_h_eff_ / delta_k_h_eff_ - 1)
        elif savings_s_h_ + delta_i_h_pds_ > 0:
            def lambda_func(lambda_h_):
                alpha_ = calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_)
                xi_, _, _, _ = solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_,
                                                          savings_s_h_, delta_i_h_pds_, delta_c_h_max_)
                return alpha_ * xi_ - (productivity_pi_ * k_h_eff_ - 1e-5)
            if np.sign(lambda_func(min_lambda)) == np.sign(lambda_func(max_lambda)):
                max_lambda = max_lambda
            else:
                max_lambda = optimize.brentq(lambda_func, min_lambda, max_lambda)

    if not min_lambda < lambda_h_init < max_lambda:
        lambda_h_init = (min_lambda + max_lambda) / 2

    res = optimize.minimize(
        fun=obj_func,
        x0=np.array(lambda_h_init),
        bounds=[(min_lambda, max_lambda)],
        args=(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
              delta_tax_sp_, discount_rate_rho_, k_h_eff_, delta_c_h_max_),
        method='SLSQP',
        tol=tolerance,
    )
    return res.x


def optimize_lambda_old(productivity_pi_, capital_t_, eta_, discount_rate_rho_, v_):
    def welfare_func(t_, lambda_h_, eta_, discount_rate_rho_, v_):
        factor = productivity_pi_ + lambda_h_
        part1 = (productivity_pi_ - factor * v_ * np.e ** (-lambda_h_ * t_)) ** (-eta_)
        part2 = t_ * factor - 1
        part3 = np.e ** (-t_ * (discount_rate_rho_ + lambda_h_))
        return part1 * part2 * part3

    def agg_welfare_func(lambda_h_):
        return integrate.quad(welfare_func, 0, capital_t_, args=(lambda_h_, eta_, discount_rate_rho_, v_))[0]

    def agg_welfare_func_(lambda_h_, years_to_recover=20, tot_weeks=1040):
        integral = 0
        for t__ in np.linspace(0, years_to_recover, tot_weeks):
            integral += welfare_func(t__, lambda_h_, eta_, discount_rate_rho_, v_) * 10 / 100
        return integral

    def objective_func_(lambda_h_):
        return -agg_welfare_func(lambda_h_)

    res = optimize.minimize(fun=objective_func_, x0=np.array(0), bounds=[(0, None)], method='SLSQP', tol=1e-10)
    return res.x


def optimize_lambda_wrapper(args, min_lambda, max_lambda):
    index, row = args
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
        lambda_h_init=row['lambda_h_init'],
        delta_tax_sp_=row['delta_tax_sp'],
        delta_c_h_max_=row['delta_c_h_max'],
        tolerance=row['tolerance'],
        min_lambda=min_lambda,
        max_lambda=max_lambda,
    )
    return index, res[0]


def optimize_data(df, tolerance=1e-10, min_lambda=.05, max_lambda=100):
    """
    Optimize the lambda parameter for each row in the dataframe
    """
    df = df.copy(deep=True)
    df['tolerance'] = tolerance
    with multiprocessing.Pool() as pool:
        res = list(tqdm.tqdm(pool.imap(partial(optimize_lambda_wrapper, min_lambda=min_lambda, max_lambda=max_lambda),
                                       df.iterrows()), total=len(df)))
    lambda_h_results = pd.Series(dict(res), name='lambda_h')
    lambda_h_results.index.names = df.index.names
    return lambda_h_results


def make_plot(df, idx, capital_t):
    """
    Make a plot of the consumption and capital losses over time
    """

    (sigma_h, delta_k_h_eff, productivity_pi, savings_s_h, delta_i_h_pds, eta,
     delta_tax_sp, discount_rate_rho, k_h_eff, delta_c_h_max, lambda_h, t_tilde, t_hat) = df.loc[idx, (
        'sigma_h', 'delta_k_h_eff', 'productivity_pi', 'savings_s_h', 'delta_i_h_pds', 'eta', 'delta_tax_sp',
        'discount_rate_rho', 'k_h_eff', 'delta_c_h_max', 'lambda_h', 't_tilde', 't_hat')].values

    if 'delta_c_h_max_adjusted' in df.columns:
        delta_c_h_max = df.loc[idx, 'delta_c_h_max_adjusted']

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7, 5))

    t_ = np.linspace(0, capital_t, 1000)
    if t_tilde not in t_:
        t_ = np.array(sorted(list(t_) + [t_tilde]))
    if t_hat not in t_:
        t_ = np.array(sorted(list(t_) + [t_hat]))
    if t_hat + t_tilde not in t_:
        t_ = np.array(sorted(list(t_) + [t_hat + t_tilde]))
    c_baseline = baseline_consumption_c_h(productivity_pi_=productivity_pi, k_h_eff_=k_h_eff)
    di_h, dc_reco, dc_savings_pds = delta_c_h_of_t(t_, productivity_pi, delta_tax_sp,delta_k_h_eff, lambda_h,
                                                   sigma_h, savings_s_h, delta_i_h_pds, delta_c_h_max,
                                                   return_elements=True)
    axs[0].fill_between(t_, c_baseline, c_baseline - di_h, color='red', alpha=0.5, label='Income loss', lw=0)
    axs[0].fill_between(t_, c_baseline - di_h, c_baseline - (di_h + dc_reco), color='red', alpha=0.25,
                       label='Reconstruction loss', lw=0)
    axs[0].fill_between(t_[dc_savings_pds != 0], (c_baseline - (di_h + dc_reco) + dc_savings_pds)[dc_savings_pds != 0],
                       (c_baseline - (di_h + dc_reco))[dc_savings_pds != 0], facecolor='none', lw=0, hatch='XXX',
                        edgecolor='grey', label='Savings and PDS')
    axs[0].plot(t_, consumption_c_of_t(t_, productivity_pi, delta_tax_sp, delta_k_h_eff, lambda_h, sigma_h,
                                      savings_s_h, delta_i_h_pds, k_h_eff, delta_c_h_max), color='black',
                label='Consumption')

    dk_eff = delta_k_h_eff_of_t(t_, t_tilde, delta_k_h_eff, lambda_h, sigma_h, delta_c_h_max, productivity_pi)
    axs[1].fill_between(t_, 0, dk_eff, color='red', alpha=0.5, label='Effective capital loss')
    axs[1].plot(t_, dk_eff, color='black', label='Effective capital loss')

    if t_tilde != 0:
        axs[0].axvline(t_tilde, color='black', linestyle='dotted', lw=1, label=r'$\tilde{t}$')
        axs[1].axvline(t_tilde, color='black', linestyle='dotted', lw=1)
    if t_hat != 0:
        axs[0].axvline(t_hat + t_tilde, color='black', linestyle='--', lw=1, label=r'$\hat{t}$')
        axs[1].axvline(t_hat + t_tilde, color='black', linestyle='--', lw=1)
    axs[1].set_xlabel('Time [y]')
    axs[0].set_ylabel(r'Consumption $c(t)$')
    axs[1].set_ylabel(r'Capital loss $\Delta k(t)$')
    fig.suptitle(f"{idx[0]} - {idx[1]} - {idx[2]}")
    for ax in axs:
        ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()


def run_experiment(sigma_h, delta_c_h_max, tolerance=1e-6, lambda_h_init=1e-5, index=None,
                   capital_t=50, min_lambda=1e-10, max_lambda=6, liquidity_='data:average', ew_vul_reduction=0.2):
    vulnerability = pd.read_csv("./intermediate/scenarios/baseline/scenario__vulnerability_unadjusted.csv",
                                index_col=['iso3', 'hazard', 'income_cat'])
    cat_info = pd.read_csv("./intermediate/scenarios/baseline/scenario__cat_info.csv",
                           index_col=['iso3', 'income_cat'])
    macro = pd.read_csv("./intermediate/scenarios/baseline/scenario__macro.csv", index_col=['iso3'])
    data = pd.merge(vulnerability, cat_info, left_index=True, right_index=True)
    data = pd.merge(data, macro, left_index=True, right_index=True)
    data.rename(columns={'avg_prod_k': 'productivity_pi', 'income_elasticity': 'eta', 'rho': 'discount_rate_rho',
                         'k': 'k_h_eff', 'recovery_share_sigma': 'sigma_h'}, inplace=True)

    data['capital_t'] = capital_t
    data['delta_tax_sp'] = 0  # neglect social protection and tax for optimization
    data['lambda_h_init'] = lambda_h_init
    data['v_ew'] = data["v"] * (1 - ew_vul_reduction * data["ew"])
    data['delta_k_h_eff'] = data['v_ew'] * data['k_h_eff']
    data['delta_i_h_pds'] = 0

    if delta_c_h_max == 'None':
        data['delta_c_h_max'] = np.nan
    elif 'factor' in delta_c_h_max:
        data['delta_c_h_max'] = float(delta_c_h_max.replace('factor:', '')) * data['k_h_eff'] * data['productivity_pi']
    else:
        raise ValueError("Invalid delta_c_h_max parameter. Choose from 'None' or 'factor{value}'.")

    if sigma_h == 'data:prv':
        data['sigma_h'] = data['recovery_share_sigma:prv']
    elif sigma_h == 'data:prv_oth':
        data['sigma_h'] = data['recovery_share_sigma:prv_oth']
    elif 'factor' in sigma_h:
        data['sigma_h'] = float(sigma_h.replace('factor:', '')) * data['delta_k_h_eff']
    else:
        raise ValueError("Invalid sigma_h parameter. Choose from 'data' or 'factor{value}'.")

    if liquidity_ == 'data:max':
        data['savings_s_h'] = data.liquidity
    elif liquidity_ == 'data:average':
        data['savings_s_h'] = data.liquidity * data.liquidity_share
    elif 'factor' in liquidity_:
        data['savings_s_h'] = float(liquidity_.replace('factor:', '')) * data['sigma_h'] * data['delta_k_h_eff']
    else:
        raise ValueError("Invalid liquidity inclusion option. Choose from 'liquid', 'illiquid', 'average'.")

    opt_data = data[['capital_t', 'sigma_h', 'delta_k_h_eff', 'productivity_pi', 'savings_s_h', 'delta_i_h_pds', 'eta',
                     'delta_tax_sp', 'discount_rate_rho', 'k_h_eff', 'delta_c_h_max', 'lambda_h_init']]
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
    args.add_argument('--tolerance', type=float, default=1e-6)
    args.add_argument('--lambda_h_init', type=float, default=0.2)
    args.add_argument('--idx', type=str, default=None)
    args.add_argument('--plot', action='store_true')
    args.add_argument('--capital_t', type=int, default=50)
    args.add_argument('--no_output', action='store_true')
    args.add_argument('--liquidity', type=str, default='data:average')
    args = args.parse_args()
    idx = args.idx
    if idx is not None:
        idx = tuple(args.idx.split('_'))

    if args.run:
        results = run_experiment(args.sigma_h, args.delta_c_h_max, args.tolerance, args.lambda_h_init, idx,
                                 args.capital_t, liquidity_=args.liquidity)
        if args.plot:
            if idx is None:
                raise ValueError("Please provide an index to plot.")
            make_plot(results, idx, 15)
            plt.show()

        if not args.no_output and idx is None:
            results.to_csv(f"./optimization_experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M')}__"
                           f"sigma_h_{args.sigma_h.replace(':', '-')}__delta_c_h_max{args.delta_c_h_max.replace(':', '-')}"
                           f"__tolerance_{args.tolerance}__lambda_h_init_{args.lambda_h_init}__capital_t_{args.capital_t}"
                           f"__liquidity_{args.liquidity.replace(':', '-')}.csv")
        elif not args.no_output:
            print("No output file created. Please remove the index selection to save the results.")
