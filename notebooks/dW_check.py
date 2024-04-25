import sys
sys.path.append('../')
from recovery_optimizer import *
from matplotlib import pyplot as plt
from lib_compute_resilience_and_risk import *
import os
import pandas as pd

idx = ('NOR', 'Earthquake', 10.0, 'q1', 'na', 'not_helped')
capital_t = 20

if __name__ == '__main__':
    experiment_countries = ['NOR']

    # define directory
    model = '/Users/robin/sync/git/global-unbreakable-model/'  # get current directory
    input_dir = model + '/inputs/'  # get inputs data directory
    intermediate_dir = model + '/intermediate/'  # get outputs data directory

    scenarios = os.listdir(os.path.join(intermediate_dir, 'scenarios'))
    scenarios = [s for s in scenarios if os.path.isdir(os.path.join(intermediate_dir, 'scenarios', s))]

    # TODO: remove this later; only for testing purposes
    scenario = 'baseline'

    option_fee = "tax"
    option_pds = "unif_poor"

    if option_fee == "insurance_premium":
        option_b = 'unlimited'
        option_t = 'perfect'
    else:
        option_b = 'data'
        option_t = 'data'

    print(f'optionFee ={option_fee}, optionPDS ={option_pds}, optionB ={option_b}, optionT ={option_t}')

    # Options and parameters
    econ_scope = "iso3"  # province, deparmtent
    event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens
    default_rp = "default_rp"  # return period to use when no rp is provided (mind that this works with protection)
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")
    poor_cat = 'q1'

    # read data

    # macro-economic country economic data
    macro = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__macro.csv"),
                        index_col=econ_scope)

    # consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
    cat_info = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__cat_info.csv"),
                           index_col=[econ_scope, "income_cat"])

    # exposure, vulnerability, and access to early warning by country, hazard, return period, income category
    hazard_ratios = pd.read_csv(
        os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_ratios.csv"),
        index_col=event_level + ["income_cat"])

    hazard_protection = pd.read_csv(
        os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_protection.csv"),
        index_col=[econ_scope, "hazard"])

    # compute
    # reshape macro and cat_info to event level, move hazard_ratios data to cat_info_event
    macro_event, cat_info_event = reshape_input(
        macro=macro,
        cat_info=cat_info,
        hazard_ratios=hazard_ratios,
        event_level=event_level,
    )

    macro_event = macro_event.loc[experiment_countries]
    cat_info_event = cat_info_event.loc[experiment_countries]

    # calculate the potential damage to capital, and consumption
    # adds 'dk', 'dc', 'dc_npv_pre' to cat_info_event_ia, also adds aggregated 'dk' to macro_event
    macro_event, cat_info_event_ia = compute_dK(
        macro_event=macro_event,
        cat_info_event=cat_info_event,
        event_level=event_level,
        affected_cats=affected_cats,
    )

    # calculate the post-disaster response
    # adds 'error_incl', 'error_excl', 'max_aid', 'need', 'aid', 'unif_aid' to macro_event
    # adds 'help_received', 'help_fee', 'help_needed' to cat_info_event_ia(h)
    macro_event, cat_info_event_iah = calculate_response(
        macro_event=macro_event,
        cat_info_event_ia=cat_info_event_ia,
        event_level=event_level,
        poor_cat=poor_cat,
        helped_cats=helped_cats,
        option_fee=option_fee,
        option_t=option_t,
        option_pds=option_pds,
        option_b=option_b,
        loss_measure="dk",
        fraction_inside=1,
        share_insured=.25,
    )
    cat_info_event_iah, macro_event = compute_dw_new(
        cat_info_event_iah=cat_info_event_iah,
        macro_event=macro_event,
        event_level_=event_level,
        capital_t=capital_t,
        delta_c_h_max=np.nan,
    )

    # # compute welfare losses
    # # adds 'dc_npv_post', 'dw' to cat_info_event_iah
    # cat_info_event_iah = compute_dW(
    #     macro_event=macro_event,
    #     cat_info_event_iah_=cat_info_event_iah,
    # )

    # aggregate to event-level (ie no more income_cat, helped_cat, affected_cat, n)
    # Computes results
    # results consists of all variables from macro, as well as 'aid' and 'dk' from macro_event
    results = prepare_output(
        macro=macro,
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        event_level=event_level,
        hazard_protection_=hazard_protection,
        econ_scope=econ_scope,
        default_rp=default_rp,
        is_local_welfare=True,
        return_stats=True,
    )

    data = pd.merge(cat_info_event_iah, macro_event, left_index=True, right_index=True, how='left').loc[idx]
    capital_t_ = capital_t
    discount_rate_rho_ = data.rho
    productivity_pi_ = data.avg_prod_k
    delta_tax_sp_ = data.tau_tax
    delta_k_h_eff_ = data.dk
    lambda_h_ = data.lambda_h
    sigma_h_ = data.reconstruction_share_sigma_h
    savings_s_h_ = data.liquidity
    delta_i_h_pds_ = data.help_received
    eta_ = data.income_elasticity_eta
    k_h_eff_ = data.k
    delta_c_h_max_ = np.nan
    diversified_share_ = data.diversified_share
    recovery_params_ = data.recovery_params
    social_protection_share_gamma_h_ = data.gamma_SP * data.n
    help_fee = data.help_fee
    help_received = data.help_received
    delta_tax_pub = data.delta_tax_pub

    t_ = np.linspace(0, capital_t_, 5000)
    discounting = np.exp(-discount_rate_rho_ * t_)

    c_baseline = baseline_consumption_c_h(productivity_pi_, k_h_eff_, delta_tax_sp_, diversified_share_)
    w_baseline = integrate.trapz(welfare_of_c(c_baseline, eta_) * discounting, t_)

    assert np.isclose(c_baseline, data.c)

    # with liquidity

    for l in [0, savings_s_h_]:
        consumption_floor_xi_, t_hat, t_tilde, delta_tilde_k_h_eff = solve_consumption_floor_xi(
            lambda_h_, sigma_h_, delta_k_h_eff_, productivity_pi_, l, delta_i_h_pds_, delta_c_h_max_,
            delta_tax_sp_, recovery_params_, social_protection_share_gamma_h_
        )
        di_h_lab, di_h_sp, dc_reco, dc_savings_pds = delta_c_h_of_t(
            t_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_, lambda_h_, sigma_h_, l, delta_i_h_pds_,
            delta_c_h_max_, recovery_params_, social_protection_share_gamma_h_, None, None, None,
            None, None, True
        )

        # run checks:

        # check that the savings_s_h_ and delta_i_h_pds_ are completely used up at time t_hat and that
        # dc_savings_pds is 0 after t_hat
        used_liquidity = integrate.trapz(dc_savings_pds, t_)
        print("The difference of used liquidity to available liqudity is ", l + delta_i_h_pds_ - used_liquidity)
        assert t_hat == np.inf or np.all(dc_savings_pds[t_ > t_hat] == 0)

        # compute welfare loss, with and without savings (here, including tax) TODO: check whether this has an impact

        w_disaster = integrate.trapz(welfare_of_c(c_baseline - (di_h_lab + di_h_sp + dc_reco - dc_savings_pds), eta_) * discounting, t_)
        dw_reco = w_baseline - w_disaster
        dw_long_term = c_baseline ** (-eta_) * used_liquidity

        dw_total = dw_reco + dw_long_term

        print(f'liquidity: {l}, dw_reco: {dw_reco}, dw_long_term: {dw_long_term}, dw_total: {dw_total}')

        dw_reco_, used_liquidity_ = recompute_with_tax(capital_t, discount_rate_rho_, productivity_pi_, delta_tax_sp_, delta_k_h_eff_,
                           lambda_h_, sigma_h_, l, delta_i_h_pds_, eta_, k_h_eff_, delta_c_h_max_, diversified_share_,
                           recovery_params_, social_protection_share_gamma_h_)

        assert np.isclose(dw_reco, dw_reco_)
        assert np.isclose(used_liquidity, used_liquidity_)
