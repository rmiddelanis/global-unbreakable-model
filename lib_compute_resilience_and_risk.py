import copy

import numpy as np
import pandas as pd
from scipy import integrate

from pandas_helper import concat_categories
from lib_prepare_scenario import average_over_rp

from recovery_optimizer import optimize_data, recompute_data_with_tax

pd.set_option('display.width', 220)

def reshape_input(macro, cat_info, hazard_ratios, event_level):
    # Broadcast macro to event level
    macro_event = pd.merge(macro, hazard_ratios.iloc[:, 0].unstack('income_cat'), left_index=True, right_index=True)[macro.columns]
    macro_event = macro_event.reorder_levels(event_level).sort_index()

    cat_info_event = pd.merge(hazard_ratios[['fa', 'v_ew']], cat_info, left_index=True,
                              right_index=True)
    cat_info_event = cat_info_event.reorder_levels(event_level + ["income_cat"]).sort_index()

    return macro_event, cat_info_event


def compute_dK(macro_event, cat_info_event, event_level, affected_cats):
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
    # cat_info_event_ia.loc[(cat_info_event_ia.affected_cat == 'na'), "dk"] = 0

    # compute reconstruction capital share
    cat_info_event_ia["dk_reco"] = cat_info_event_ia[["dk", "k_household_share"]].prod(axis=1)

    # "national" losses
    macro_event_["dk_ctry"] = agg_to_event_level(cat_info_event_ia, "dk", event_level)

    return macro_event_, cat_info_event_ia


def calculate_response(macro_event, cat_info_event_ia, event_level, helped_cats, poor_cat, pds_targeting="data",
                       pds_variant="unif_poor", pds_borrowing_ability="data", loss_measure="dk_reco", pds_shareable=.2):
    cat_info_event_iah = concat_categories(cat_info_event_ia, cat_info_event_ia, index=helped_cats)
    cat_info_event_iah["help_received"] = 0.0

    macro_event, cat_info_event_iah = compute_response(
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        event_level=event_level,
        poor_cat=poor_cat,
        pds_targeting=pds_targeting,
        pds_variant=pds_variant,
        pds_borrowing_ability=pds_borrowing_ability,
        pds_shareable=pds_shareable,
        loss_measure=loss_measure,
    )

    return macro_event, cat_info_event_iah


def compute_response(macro_event, cat_info_event_iah, event_level, poor_cat, pds_targeting="data", pds_variant="unif_poor", pds_borrowing_ability="data",
                     pds_shareable=.2, loss_measure="dk_reco"):

    """Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing,
    and dimensioning of the help. Returns copies of macro_event and cats_event_iah updated with stuff.
    @param pds_shareable:
    @param macro_event:
    @param cat_info_event_iah:
    @param event_level:
    @param poor_cat: list of income categories to be considered poor
    @param pds_targeting: Targeting error option. Changes how inclusion and exclusion errors are calculated. Values:
        "perfect": no targeting errors, "prop_nonpoor_lms": , "data": , "x33": , "incl": , "excl":
    @param pds_variant: Post disaster support options. Values: "unif_poor", "no", "prop", "prop_nonpoor", "unif_poor_only"
    @param pds_borrowing_ability: Post disaster support budget option. Values: "data", "unif_poor", "max01", "max05", "unlimited",
        "one_per_affected", "one_per_helped", "one", "no"
    @param option_fee: Values: "insurance_premium", "tax"
    @param fraction_inside:
    @param loss_measure:
    @return: macro_event, cats_event_iah """

    macro_event_ = macro_event.copy()
    cat_info_event_iah_ = cat_info_event_iah.copy()
    cat_info_event_iah_[['income_cat', 'helped_cat', 'affected_cat']] = cat_info_event_iah_.reset_index()[['income_cat', 'helped_cat', 'affected_cat']].values

    # because cats_event_ia is duplicated in cat_info_event_iah_, cat_info_event_iah_.n.groupby(level=event_level).sum() is
    # 2 instead of 1, here /2 is to correct it. macro_event_["fa"] =  agg_to_event_level(cats_event_ia,"fa") would work
    # but needs to pass a new variable cats_event_ia.
    macro_event_["fa"] = agg_to_event_level(cat_info_event_iah_, "fa", event_level) / 2

    # targeting errors
    if pds_targeting == "perfect":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 0
    elif pds_targeting == "prop_nonpoor_lms":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 1 - 25 / 80  # 25% of pop chosen among top 80 DO receive the aid
    elif pds_targeting == "data":
        macro_event_["error_incl"] = ((1 - macro_event_["prepare_scaleup"]) / 2 * macro_event_["fa"] / (1 - macro_event_["fa"])).clip(upper=1, lower=0)  # as in equation 16 of the paper
        macro_event_["error_excl"] = ((1 - macro_event_["prepare_scaleup"]) / 2).clip(upper=1, lower=0)  # as in equation 16 of the paper
    elif pds_targeting == "x33":
        macro_event_["error_incl"] = .33 * macro_event_["fa"] / (1 - macro_event_["fa"])
        macro_event_["error_excl"] = .33
    elif pds_targeting == "incl":
        macro_event_["error_incl"] = .33 * macro_event_["fa"] / (1 - macro_event_["fa"])
        macro_event_["error_excl"] = 0
    elif pds_targeting == "excl":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 0.33
    else:
        print("unrecognized targeting error option " + pds_targeting)
        return None

    # counting (mind self multiplication of n)
    # compute inclusion and exclusion error impact
    cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped') & (cat_info_event_iah_.affected_cat == 'a'), "n"] *= (
            1 - macro_event_["error_excl"])
    cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped') & (cat_info_event_iah_.affected_cat == 'a'), "n"] *= (
        macro_event_["error_excl"])
    cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped') & (cat_info_event_iah_.affected_cat == 'na'), "n"] *= (
        macro_event_["error_incl"])
    cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped') & (cat_info_event_iah_.affected_cat == 'na'), "n"] *= (
            1 - macro_event_["error_incl"])

    # !!!! n is one again from here.
    # print(cat_info_event_iah_.n.groupby(level=event_level).sum())

    # Step 0: define max_aid
    macro_event_["max_aid"] = macro_event_["max_increased_spending"] * macro_event_["gdp_pc_pp"]

    # post disaster support (PDS) calculation depending on option_pds
    # Step 1: Compute the help needed for all helped households to fulfill the policy
    if pds_variant == "no":
        macro_event_["aid"] = 0
        macro_event_['need'] = 0
        cat_info_event_iah_['help_needed'] = 0
        cat_info_event_iah_['help_received'] = 0
        pds_borrowing_ability = 'no'
    elif pds_variant == "unif_poor":
        # help_received for all helped hh = 80% of dk for poor, affected hh
        # share of losses to be covered * (losses of helped, affected, poor households)
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = pds_shareable * cat_info_event_iah_.xs(('helped', 'a', 'q1'), level=('helped_cat', 'affected_cat', 'income_cat'))[loss_measure]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_needed"] = 0
    elif pds_variant == "unif_poor_only":
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped') & (cat_info_event_iah_.income_cat == poor_cat), "help_needed"] = pds_shareable * cat_info_event_iah_.xs(('helped', 'a', 'q1'), level=('helped_cat', 'affected_cat', 'income_cat'))[loss_measure]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped') | (~cat_info_event_iah_.income_cat == poor_cat), "help_received"] = 0
    elif pds_variant == "proportional":
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = pds_shareable * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), loss_measure]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_needed"] = 0

    # Step 2: total need (cost) for all helped hh = sum over help_needed for helped hh
    macro_event_["need"] = agg_to_event_level(cat_info_event_iah_, "help_needed", event_level)

    # actual aid reduced by capacity
    if pds_borrowing_ability == "data":
        # Step 3: total need (cost) for all helped hh clipped at max_aid
        macro_event_["aid"] = macro_event_["need"].clip(upper=macro_event_["max_aid"] * macro_event_["borrowing_ability"])
    elif pds_borrowing_ability == "unif_poor":
        macro_event_["aid"] = macro_event_["need"].clip(upper=macro_event_["max_aid"])
    elif pds_borrowing_ability == "max01":
        macro_event_["max_aid"] = 0.01 * macro_event_["gdp_pc_pp"]
        macro_event_["aid"] = (macro_event_["need"]).clip(upper=macro_event_["max_aid"])
    elif pds_borrowing_ability == "max05":
        macro_event_["max_aid"] = 0.05 * macro_event_["gdp_pc_pp"]
        macro_event_["aid"] = (macro_event_["need"]).clip(upper=macro_event_["max_aid"])
    elif pds_borrowing_ability == "unlimited":
        macro_event_["aid"] = macro_event_["need"]
    elif pds_borrowing_ability == "one_per_affected":
        d = cat_info_event_iah_.loc[(cat_info_event_iah_.affected_cat == 'a')]
        d["un"] = 1
        macro_event_["need"] = agg_to_event_level(d, "un", event_level)
        macro_event_["aid"] = macro_event_["need"]
    elif pds_borrowing_ability == "one_per_helped":
        d = cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')]
        d["un"] = 1
        macro_event_["need"] = agg_to_event_level(d, "un", event_level)
        macro_event_["aid"] = macro_event_["need"]
    elif pds_borrowing_ability == "one":
        macro_event_["aid"] = 1
    elif pds_borrowing_ability == 'no':
        pass
    else:
        print(f"Unknown optionB={pds_borrowing_ability}")

    if pds_variant == "unif_poor":
        # Step 4: help_received = unif_aid = aid/(N hh helped)
        macro_event_["unif_aid"] = (macro_event_["aid"] / (
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == "helped"), "n"].groupby(level=event_level).sum()
        )).fillna(0)  # division by zero is possible if no losses occur
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_received"] = 0
    elif pds_variant == "unif_poor_only":
        macro_event_["unif_aid"] = macro_event_["aid"] / (
            cat_info_event_iah_.loc[
                (cat_info_event_iah_.helped_cat == "helped") & (cat_info_event_iah_.income_cat.isin(poor_cat)), "n"].groupby(
                level=event_level).sum())
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[
            (cat_info_event_iah_.helped_cat == 'not_helped') | (~cat_info_event_iah_.income_cat.isin(poor_cat)), "help_received"] = 0
    elif pds_variant == "proportional":
        cat_info_event_iah_["help_received"] = (macro_event_["aid"] / macro_event_["need"] * cat_info_event_iah_["help_needed"]).fillna(0)

    cat_info_event_iah_.drop(['income_cat', 'helped_cat', 'affected_cat'], axis=1, inplace=True)
    return macro_event_, cat_info_event_iah_


# TODO: add docstring (explain parameters)
def optimize_recovery(macro_event, cat_info_event_iah, capital_t=50, delta_c_h_max=np.nan, num_cores=None):
    opt_data = pd.merge(
        macro_event.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta'}),
        cat_info_event_iah.rename(columns={'k': 'k_h_eff', 'dk': 'delta_k_h_eff', 'liquidity': 'savings_s_h',
                                           'help_received': 'delta_i_h_pds', 'k_household_share': 'sigma_h'}),
        left_index=True, right_index=True
    )[['productivity_pi', 'discount_rate_rho', 'eta', 'k_h_eff', 'delta_k_h_eff', 'savings_s_h',
       'sigma_h', 'delta_i_h_pds', 'delta_tax_sp', 'diversified_share']]
    opt_data['capital_t'] = capital_t
    opt_data['delta_c_h_max'] = delta_c_h_max
    # opt_data = opt_data.xs('a', level='affected_cat', drop_level=False).round(10)
    opt_data = opt_data[opt_data.delta_k_h_eff > 0].round(10)
    recovery_rates_lambda = optimize_data(
        df_in=opt_data,
        tolerance=1e-2,
        min_lambda=.05,
        max_lambda=1e2,
        num_cores=num_cores
    )
    return pd.merge(cat_info_event_iah, recovery_rates_lambda, left_index=True, right_index=True, how='left')


def compute_dw_reco_and_used_savings(cat_info_event_iah, macro_event, event_level_, capital_t=50, delta_c_h_max=np.nan,
                                     num_cores=None):
    recovery_parameters = cat_info_event_iah.xs('a', level='affected_cat')[['n', 'dk', 'lambda_h']]
    recovery_parameters['dk_abs'] = recovery_parameters['dk'] * recovery_parameters['n']
    recovery_parameters.drop(['n', 'dk'], axis=1, inplace=True)
    recovery_parameters['recovery_params'] = recovery_parameters[['dk_abs', 'lambda_h']].apply(tuple, axis=1)
    recovery_parameters = recovery_parameters.groupby(level=event_level_).agg(list).recovery_params
    recompute_data = pd.merge(cat_info_event_iah, macro_event, left_index=True, right_index=True, how='left')
    recompute_data = recompute_data.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta',
                                                    'k_household_share': 'sigma_h', 'k': 'k_h_eff',
                                                    'dk': 'delta_k_h_eff', 'liquidity': 'savings_s_h',
                                                    'help_received': 'delta_i_h_pds'})
    recompute_data['capital_t'] = capital_t
    recompute_data['delta_c_h_max'] = delta_c_h_max
    recompute_data['social_protection_share_gamma_h'] = recompute_data['gamma_SP'] * recompute_data['n']

    recompute_data = recompute_data[['capital_t', 'delta_c_h_max', 'social_protection_share_gamma_h', 'productivity_pi',
                                     'discount_rate_rho', 'eta', 'k_h_eff', 'delta_k_h_eff',
                                     'savings_s_h', 'sigma_h', 'delta_i_h_pds', 'delta_tax_sp', 'lambda_h',
                                     'diversified_share']]

    recompute_data = pd.merge(recompute_data, recovery_parameters, left_index=True, right_index=True, how='left')

    dw_ds_reco = recompute_data_with_tax(recompute_data, num_cores)

    cat_info_event_iah = pd.merge(cat_info_event_iah, dw_ds_reco, left_index=True, right_index=True, how='left')
    cat_info_event_iah = pd.merge(cat_info_event_iah, recovery_parameters, left_index=True, right_index=True, how='left')

    return cat_info_event_iah


def compute_dw_long_term(cat_info_event_iah, macro_event, event_level):#, long_term_horizon_=None):
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


def compute_dw(cat_info_event_iah, macro_event, event_level_, capital_t=50, delta_c_h_max=np.nan, num_cores=None):
    # compute the optimal recovery rates
    cat_info_event_iah_ = optimize_recovery(
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        capital_t=capital_t,
        delta_c_h_max=delta_c_h_max,
        num_cores=num_cores,
    )
    if cat_info_event_iah_.xs('a', level='affected_cat').lambda_h.isna().any():
        failed_optimizations = cat_info_event_iah_[cat_info_event_iah_['lambda_h'].isna()].xs('a', level='affected_cat').index.droplevel(['rp', 'helped_cat', 'income_cat']).unique()
        print(f"Failed to optimize recovery rates for {failed_optimizations}. Dropping entries.")
        cat_info_event_iah_ = cat_info_event_iah_.drop(failed_optimizations)

    # compute the welfare losses from destroyed assets, decreased transfers, and reconstruction
    cat_info_event_iah_ = compute_dw_reco_and_used_savings(cat_info_event_iah_, macro_event, event_level_, capital_t,
                                                           delta_c_h_max, num_cores)

    # compute the long-term welfare losses from public asset reconstruction costs, PDS costs, and used savings
    cat_info_event_iah_, macro_event_ = compute_dw_long_term(cat_info_event_iah_, macro_event, event_level_)#, long_term_horizon_)

    # sum the welfare losses from reconstruction and long-term welfare losses
    cat_info_event_iah_['dw'] = cat_info_event_iah_['dW_reco'] + cat_info_event_iah_['dW_long_term']
    return cat_info_event_iah_, macro_event_


def prepare_output(macro, macro_event, cat_info_event_iah, event_level, hazard_protection_,
                   is_local_welfare=True, return_stats=True):#, long_term_horizon_=None):
    # generate output df
    out = pd.DataFrame(index=macro_event.index)

    # pull 'aid' and 'dk' from macro_event
    out["average_aid_cost_pc"] = macro_event["aid"]
    out["dk"] = macro_event["dk_ctry"]

    # aggregate delta_W at event-level
    out["dw"] = agg_to_event_level(cat_info_event_iah, "dw", event_level)

    out["dk_tot"] = out["dk"] * macro_event["pop"]
    out["dw_tot"] = out["dw"] * macro_event["pop"]

    if return_stats:
        stats = np.setdiff1d(cat_info_event_iah.columns, event_level + ['helped_cat', 'affected_cat', 'income_cat',
                                                                        'has_received_help_from_PDS_cat', 'recovery_params'])
        df_stats = agg_to_event_level(cat_info_event_iah, stats, event_level)
        print("stats are " + ",".join(stats))
        out[df_stats.columns] = df_stats

    # aggregate losses
    # Averages over return periods to get dk_{hazard} and dW_{hazard}
    out = average_over_rp(out, hazard_protection_)

    # Sums over hazard dk, dW (gets one line per economy)
    # TODO: average over axfin, social, gamma_SP does not really carry any meaning. Should be dropped.
    out = out.groupby(level="iso3").aggregate(
        {c: 'sum' if c not in ['axfin', 'social', 'gamma_SP'] else 'mean' for c in out.columns}
    )

    # adds dk and dw-like columns to macro
    out = pd.concat((macro, out), axis=1)

    # computes socio-economic capacity and risk at economy level
    out = calc_risk_and_resilience_from_k_w(df=out, is_local_welfare=is_local_welfare)#, long_term_horizon_=long_term_horizon_)

    return out


def agg_to_event_level(df, seriesname, event_level):
    """ aggregates seriesname in df (string of list of string) to event level (country, hazard, rp) across income_cat and affected_cat using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T * df["n"]).T.groupby(level=event_level).sum()


def calc_risk_and_resilience_from_k_w(df, is_local_welfare=True):#, long_term_horizon_=None):
    """Computes risk and resilience from dk, dw and protection.
    Line by line: multiple return periods or hazard is transparent to this function"""
    df = df.copy()

    # Expressing welfare losses in currency

    # linearly approximated derivative of welfare with respect to NPV of future consumption
    # Note: no longer using NPV of consupmtion with the model update!
    if is_local_welfare:
        w_prime = df["gdp_pc_pp"]**(-df["income_elasticity_eta"])
    else:
        w_prime = df["gdp_pc_pp_nat"]**(-df["income_elasticity_eta"])

    d_w_ref = w_prime * df["dk"]

    # expected welfare loss (per household and total)
    df["dWpc_currency"] = df["dw"] / w_prime
    df["dWtot_currency"] = df["dWpc_currency"] * df["pop"]

    # Risk to welfare as percentage of local GDP
    df["risk"] = df["dWpc_currency"] / df["gdp_pc_pp"]

    # socio-economic resilience
    df["resilience"] = d_w_ref / df["dw"]

    # risk to assets
    df["risk_to_assets"] = df.resilience * df.risk

    return df
