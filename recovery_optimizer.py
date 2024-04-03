import time

import numpy as np
import multiprocessing as mp

import pandas as pd
import tqdm
from scipy import integrate, optimize


def delta_k_h_eff_of_t(t_, delta_k_h_eff_, lambda_h_):
    """
    Compute the dynamic effective capital loss
    """
    return delta_k_h_eff_ * np.exp(-lambda_h_ * t_)


# TODO: delta_tax_sp
def delta_i_h_lab_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_):
    """
    Compute dynamic the labor income loss
    """
    dk_eff_of_t = delta_k_h_eff_of_t(t_=t_, delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)
    return dk_eff_of_t * productivity_pi_ * (1 - delta_tax_sp_)


# TODO: for now, neglect social protection in optimization
def delta_i_h_sp_of_t(t_):
    return 0


def delta_i_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_):
    """
    Compute the dynamic income loss
    """
    d_i_h_lab_of_t = delta_i_h_lab_of_t(t_, productivity_pi_=productivity_pi_, delta_tax_sp_=delta_tax_sp_,
                                        delta_k_h_eff_=delta_k_h_eff_, lambda_h_=lambda_h_)
    d_i_h_sp_of_t = delta_i_h_sp_of_t(t_)
    # income from post disaster support (considered a one-time payment) is considered savings
    return d_i_h_lab_of_t + d_i_h_sp_of_t


def delta_c_h_reco_of_t(t_, delta_k_h_eff_, lambda_h_, sigma_h_):
    """
    Compute the dynamic recovery consumption loss
    """
    return lambda_h_ * sigma_h_ * delta_k_h_eff_ * np.exp(-lambda_h_ * t_)


def calc_t_head(lambda_h_, consumption_floor_xi_):
    """
    Compute the time at which the consumption floor is reached
    """
    return - 1 / lambda_h_ * np.log(consumption_floor_xi_)


def delta_c_h_savings_pds_of_t(t_, lambda_h_, sigma_h_, productivity_pi_,
                               delta_k_h_eff_, savings_s_h_, delta_i_h_pds_):
    """
    Compute the dynamic consumption gain from savings and PDS
    """
    if savings_s_h_ + delta_i_h_pds_ <= 0:
        res = np.zeros_like(t_)
        if res.shape == ():
            return 0
        return res
    consumption_floor_xi_, t_head = solve_consumption_floor_xi(
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        delta_k_h_eff_=delta_k_h_eff_,
        productivity_pi_=productivity_pi_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_
    )
    if consumption_floor_xi_ == np.nan:
        print("Warning. Xi is nan.")
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

    if isinstance(t_, (float, int)):
        if t_ > t_head:
            d_c_h_savings_pds_of_t = 0
    else:
        d_c_h_savings_pds_of_t[t_ > t_head] = 0
    return d_c_h_savings_pds_of_t


# TODO: delta_tax_sp
def delta_c_h_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                   savings_s_h_, delta_i_h_pds_):
    """
    Compute the dynamic consumption loss
    """
    delta_i_h = delta_i_h_of_t(
        t_=t_,
        productivity_pi_=productivity_pi_,
        delta_tax_sp_=delta_tax_sp_,
        delta_k_h_eff_=delta_k_h_eff_,
        lambda_h_=lambda_h_,
    )
    delta_c_h_reco = delta_c_h_reco_of_t(
        t_=t_,
        sigma_h_=sigma_h_,
        lambda_h_=lambda_h_,
        delta_k_h_eff_=delta_k_h_eff_,
    )
    delta_c_h_savings_pds = delta_c_h_savings_pds_of_t(
        t_=t_,
        lambda_h_=lambda_h_,
        sigma_h_=sigma_h_,
        productivity_pi_=productivity_pi_,
        delta_k_h_eff_=delta_k_h_eff_,
        savings_s_h_=savings_s_h_,
        delta_i_h_pds_=delta_i_h_pds_
    )
    return delta_i_h + delta_c_h_reco - delta_c_h_savings_pds


def baseline_consumption_c_h(productivity_pi_, k_h_eff_):
    """
    Compute the baseline consumption level
    """
    return productivity_pi_ * k_h_eff_


def consumption_c_of_t(t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_,
                       savings_s_h_, delta_i_h_pds_, k_h_eff_):
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
        delta_i_h_pds_=delta_i_h_pds_
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
    return (c_ ** (1 - eta_) - 1) / (1 - eta_)


def discounted_welfare_w_of_t(t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
                              sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_):
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
        delta_i_h_pds_=delta_i_h_pds_
    )
    w_of_c_of_t = welfare_of_c(c_of_t, eta_=eta_) * np.exp(-discount_rate_rho_ * t_)
    return w_of_c_of_t


def aggregate_welfare_w_of_c_of_capital_t(capital_t_, discount_rate_rho_, productivity_pi_, delta_tax_sp_,
                                          delta_k_h_eff_, lambda_h_, sigma_h_, savings_s_h_, delta_i_h_pds_, eta_,
                                          k_h_eff_):
    """
    Compute the time-aggregate welfare function
    """
    args = (discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_,
            sigma_h_, savings_s_h_, delta_i_h_pds_, eta_, k_h_eff_)
    integral = integrate.quad(discounted_welfare_w_of_t, 0, capital_t_, args=args)[0]
    return integral


def calc_alpha(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_):
    """
    Compute the alpha parameter
    """
    return (productivity_pi_ + lambda_h_ * sigma_h_) * delta_k_h_eff_


def solve_consumption_floor_xi(lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                               delta_i_h_pds_):
    """
    Solve for the consumption floor xi
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

    if savings_s_h_ + delta_i_h_pds_ >= alpha / lambda_h_:
        return 0, 0  # All capital losses can be repaired at time 0, and consumption can be reduced to 0%
    elif savings_s_h_ + delta_i_h_pds_ <= 0:
        return np.nan, 0  # No savings available to offset consumption losses

    # Solve numerically for xi
    def xi_func(xi_):
        rhs = 1 - lambda_h_ / alpha * (savings_s_h_ + delta_i_h_pds_)
        lhs = xi_ * (1 - np.log(xi_)) if xi_ > 0 else 0
        return lhs - rhs
    xi_res = optimize.brentq(xi_func, 0, 1)
    t_head = calc_t_head(lambda_h_, xi_res)
    return xi_res, t_head


def objective_func(lambda_h_, capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_,
                   delta_i_h_pds_, eta_, delta_tax_sp_, discount_rate_rho_, k_h_eff_):
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
        delta_i_h_pds_=delta_i_h_pds_
    )
    return -objective  # negative to transform into a minimization problem


def optimize_lambda(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
                    discount_rate_rho_, k_h_eff_, lambda_h_init=0, delta_tax_sp_=0, verbose=False,
                    obj_func=objective_func):
    """
    Optimize the lambda parameter
    """
    res = optimize.minimize(
        fun=obj_func,
        x0=np.array(lambda_h_init),
        bounds=[(0, None)],
        args=(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
              delta_tax_sp_, discount_rate_rho_, k_h_eff_),
        method='SLSQP',
        tol=1e-10,
    )
    if verbose:
        print(res)
        return res
    return res.x


def optimize_data(df, capital_t_col: str, sigma_h_col: str, delta_k_h_eff_col: str, productivity_pi_col: str,
                  savings_s_h_col: str, delta_i_h_pds_col: str, eta_col: str, delta_tax_sp_col: str,
                  discount_rate_rho_col: str, k_h_eff_col: str, lambda_h_init_=0, verbose_=False):
    """
    Optimize the lambda parameter for each row in the dataframe
    """
    # with mp.Pool(mp.cpu_count()) as pool:
    #     results = pool.starmap(optimize_lambda, [df[capital_t_col], df[sigma_h_col], df[delta_k_h_eff_col],
    #                                              df[productivity_pi_col], df[savings_s_h_col], df[delta_i_h_pds_col],
    #                                              df[eta_col], df[discount_rate_rho_col], df[k_h_eff_col],
    #                                              df[lambda_h_init_col], df[delta_tax_sp_col], df[verbose_col]])
    results = pd.Series(index=df.index)
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Optimizing lambda parameter"):
        capital_t_ = row[capital_t_col]
        sigma_h_ = row[sigma_h_col]
        delta_k_h_eff_ = row[delta_k_h_eff_col]
        productivity_pi_ = row[productivity_pi_col]
        savings_s_h_ = row[savings_s_h_col]
        delta_i_h_pds_ = row[delta_i_h_pds_col]
        eta_ = row[eta_col]
        discount_rate_rho_ = row[discount_rate_rho_col]
        k_h_eff_ = row[k_h_eff_col]
        delta_tax_sp_ = row[delta_tax_sp_col]
        verbose_ = verbose_
        lambda_h_init_ = lambda_h_init_

        res = optimize_lambda(
            capital_t_=capital_t_,
            sigma_h_=sigma_h_,
            delta_k_h_eff_=delta_k_h_eff_,
            productivity_pi_=productivity_pi_,
            savings_s_h_=savings_s_h_,
            delta_i_h_pds_=delta_i_h_pds_,
            eta_=eta_,
            discount_rate_rho_=discount_rate_rho_,
            k_h_eff_=k_h_eff_,
            lambda_h_init=lambda_h_init_,
            delta_tax_sp_=delta_tax_sp_,
            verbose=verbose_
        )
        results[idx] = res
    return results


if __name__ == '__main__':
    vulnerability = pd.read_csv("./intermediate/scenarios/baseline/scenario__vulnerability_unadjusted.csv",
                                index_col=['iso3', 'hazard', 'income_cat'])
    cat_info = pd.read_csv("./intermediate/scenarios/baseline/scenario__cat_info.csv",
                           index_col=['iso3', 'income_cat'])
    macro = pd.read_csv("./intermediate/scenarios/baseline/scenario__macro.csv", index_col=['iso3'])

    data = pd.merge(vulnerability, cat_info, left_index=True, right_index=True)
    data = pd.merge(data, macro, left_index=True, right_index=True)

    # data['capital_t'] = np.inf
    data['capital_t'] = 10  # TODO: old recovery optimization also used 10
    # data['sigma_h'] = 0.5
    data['sigma_h'] = 1
    data['savings_s_h'] = 0
    data['delta_i_h_pds'] = 0
    data['delta_tax_sp'] = 0
    data['verbose'] = True
    data['lambda_h_init'] = 0

    data.rename(columns={'avg_prod_k': 'productivity_pi', 'income_elasticity': 'eta', 'rho': 'discount_rate_rho',
                         'k': 'k_h_eff'},
                inplace=True)

    # data['v_ew'] = data["v"] * (1 - 0.2 * data["ew"])
    data['v_ew'] = data["v"]  # TODO: old recovery optimization also used v, not v_ew
    data['delta_k_h_eff'] = data['v_ew'] * data['k_h_eff']

    opt_data = data[['capital_t', 'sigma_h', 'delta_k_h_eff', 'productivity_pi', 'savings_s_h', 'delta_i_h_pds', 'eta',
                     'delta_tax_sp', 'discount_rate_rho', 'k_h_eff']]

    (capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
     delta_tax_sp_, discount_rate_rho_, k_h_eff_) = opt_data.loc[('ALB', 'q1', 'Earthquake')].values

    recovery_durations = optimize_data(opt_data, *opt_data.columns)
    #
    # # # Dummy Parameters
    # # vulnerability = 0.3
    # # k_h_eff = 5
    # # delta_k_h_eff = vulnerability * k_h_eff
    # # sigma_h = 0.5
    # # delta_k_h_reco = sigma_h * delta_k_h_eff
    # #
    # # savings_s_h = delta_k_h_reco * 0.25 / 2
    # # delta_i_h_pds = delta_k_h_reco * 0.25 / 2
    # #
    # # productivity_pi = .3
    # # delta_tax_sp = 0
    # # discount_rate_rho = 0.06
    # # eta = 1.5
    # #
    # # # Time horizon
    # # capital_t = np.inf
    # # # capital_t = 10
    #
    # # Solve
    # time_pre = time.time()
    # result = optimize.minimize(
    #     fun=objective_func,
    #     x0=np.array(0),
    #     bounds=[(0, None)],
    #     args=(capital_t_, sigma_h_, delta_k_h_eff_, productivity_pi_, savings_s_h_, delta_i_h_pds_, eta_,
    #           delta_tax_sp_, discount_rate_rho_, k_h_eff_),
    #     method='SLSQP',
    #     tol=1e-10,
    # )
    # print(f"Optimization took {time.time() - time_pre:.2f} seconds")
    #
    # lambda_h = result.x[0]
    # consumption_floor_xi, t_head = solve_consumption_floor_xi(
    #     lambda_h_=lambda_h,
    #     sigma_h_=sigma_h_,
    #     delta_k_h_eff_=delta_k_h_eff_,
    #     productivity_pi_=productivity_pi_,
    #     savings_s_h_=savings_s_h_,
    #     delta_i_h_pds_=delta_i_h_pds_
    # )
    #
    # print(result)
