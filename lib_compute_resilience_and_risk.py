import copy

import numpy as np
import pandas as pd
import tqdm
from scipy import integrate

from pandas_helper import concat_categories
from lib_gather_data import average_over_rp

from recovery_optimizer import optimize_data, recompute_data_with_tax

pd.set_option('display.width', 220)


def reshape_input(macro, cat_info, hazard_ratios, event_level):
    # FORMATING
    # gets the event level index
    # index composed on countries, hazards and rps.^
    # event_level_index = hazard_ratios.reset_index().set_index(event_level).index

    # Broadcast macro to event level
    macro_event = pd.merge(macro, hazard_ratios.iloc[:, 0].unstack('income_cat'), left_index=True, right_index=True)[macro.columns]
    macro_event = macro_event.reorder_levels(event_level).sort_index()
    # macro_event = broadcast_simple(macro, event_level_index)

    # cat_info_event = pd.merge(cat_info, hazard_ratios[['fa', 'v_ew', 'macro_multiplier_Gamma']], left_index=True,
    #                           right_index=True)
    # cat_info_event = broadcast_simple(cat_info, event_level_index).reset_index().set_index(event_level + ["income_cat"])
    # cat_info_event[['fa', 'v_ew', 'macro_multiplier_Gamma']] = hazard_ratios.reset_index().set_index(cat_info_event.index.names)[['fa', 'v_ew', 'macro_multiplier_Gamma']]
    cat_info_event = pd.merge(hazard_ratios[['fa', 'v_ew', 'macro_multiplier_Gamma']], cat_info, left_index=True,
                              right_index=True)
    cat_info_event = cat_info_event.reorder_levels(event_level + ["income_cat"]).sort_index()
    print("pulling ['fa', 'v_ew', 'macro_multiplier_Gamma'] into cat_info_event from hazard_ratios")

    return macro_event, cat_info_event


def compute_dK(macro_event, cat_info_event, event_level, affected_cats):
    macro_event_ = copy.deepcopy(macro_event)
    cat_info_event_ = copy.deepcopy(cat_info_event)
    cat_info_event_ia = concat_categories(cat_info_event_, cat_info_event_, index=affected_cats)

    # counts affected and non affected
    n_affected = cat_info_event_["n"] * cat_info_event_.fa
    n_not_affected = cat_info_event_["n"] * (1 - cat_info_event_.fa)
    cat_info_event_ia["n"] = concat_categories(n_affected, n_not_affected, index=affected_cats)

    # de_index so can access cats as columns and index is still event
    # cat_info_event_ia = cat_info_event_ia.reset_index(["income_cat", "affected_cat"]).sort_index()

    # moved computation of actual vunlerability to (lib_)gather_data --> recompute_after_policy_change
    # cat_info_event_ia["v_ew"] = cat_info_event_ia["v"] * (1 - macro_event_["pi"] * cat_info_event_ia["ew"])

    # capital losses and total capital losses
    cat_info_event_ia["dk"] = cat_info_event_ia[["k", "v_ew"]].prod(axis=1, skipna=False)  # capital potentially be damaged
    cat_info_event_ia["dk_reco"] = cat_info_event_ia["dk"] * macro_event_["reconstruction_share_sigma_h"]  # capital to be reconstructed at the expense of the households

    cat_info_event_ia.loc[pd.IndexSlice[:, :, :, :, 'na'], "dk"] = 0
    # cat_info_event_ia.loc[(cat_info_event_ia.affected_cat == 'na'), "dk"] = 0

    # "national" losses
    macro_event_["dk_ctry"] = agg_to_event_level(cat_info_event_ia, "dk", event_level)

    # TODO: can no longer use macro_multiplier_Gamma due to the inclusion of savings. Need to handle time explicitly.
    # immediate consumption losses: direct capital losses plus losses through event-scale depression of transfers
    cat_info_event_ia["dc"] = (
            (1 - macro_event_["tau_tax"]) * cat_info_event_ia["dk"]
            + cat_info_event_ia["gamma_SP"] * macro_event_["tau_tax"] * macro_event_["dk_ctry"]
    )

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    cat_info_event_ia["dc_npv_pre"] = cat_info_event_ia[["dc", "macro_multiplier_Gamma"]].product(axis=1)

    return macro_event_, cat_info_event_ia


def calculate_response(macro_event, cat_info_event_ia, event_level, helped_cats, poor_cat, option_fee="tax", option_t="data",
                       option_pds="unif_poor", option_b="data", loss_measure="dk_reco", fraction_inside=1,
                       share_insured=.25):
    cat_info_event_iah = concat_categories(cat_info_event_ia, cat_info_event_ia, index=helped_cats)
    # cat_info_event_iah = cat_info_event_iah.reset_index(['income_cat', 'affected_cat', 'helped_cat']).sort_index()
    cat_info_event_iah["help_received"] = 0.0
    cat_info_event_iah["help_fee"] = 0.0

    # baseline case (no insurance)
    if option_fee != "insurance_premium":
        macro_event, cat_info_event_iah = compute_response(
            macro_event=macro_event,
            cat_info_event_iah=cat_info_event_iah,
            event_level=event_level,
            poor_cat=poor_cat,
            option_t=option_t,
            option_pds=option_pds,
            option_b=option_b,
            option_fee=option_fee,
            fraction_inside=fraction_inside,
            loss_measure=loss_measure,
        )

    # special case of insurance that adds to existing default PDS
    else:
        # compute post disaster response with default PDS from data ONLY
        m__, c__ = compute_response(macro_event, cat_info_event_iah, event_level, poor_cat=poor_cat, option_t="data", option_pds="unif_poor",
                                    option_b="data", option_fee="tax", fraction_inside=1, loss_measure="dk_reco")
        # change column name helped_cat to has_received_help_from_PDS_cat
        c__h = c__.rename(columns=dict(helped_cat="has_received_help_from_PDS_cat"))

        cats_event_iah_h = concat_categories(c__h, c__h, index=helped_cats).reset_index(helped_cats.name).sort_index()

        # compute post disaster response with insurance ONLY
        macro_event, cat_info_event_iah = compute_response(macro_event.assign(shareable=share_insured), cats_event_iah_h,
                                                           event_level, poor_cat=poor_cat, option_t=option_t, option_pds=option_pds,
                                                           option_b=option_b, option_fee=option_fee,
                                                           fraction_inside=fraction_inside, loss_measure=loss_measure)

        columns_to_add = ["need", "aid"]
        macro_event[columns_to_add] += m__[columns_to_add]

    return macro_event, cat_info_event_iah


def compute_response(macro_event, cat_info_event_iah, event_level, poor_cat, option_t="data", option_pds="unif_poor", option_b="data",
                     option_fee="tax", fraction_inside=1, loss_measure="dk_reco"):

    """Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing,
    and dimensioning of the help. Returns copies of macro_event and cats_event_iah updated with stuff.
    @param macro_event:
    @param cat_info_event_iah:
    @param event_level:
    @param poor_cat: list of income categories to be considered poor
    @param option_t: Targeting error option. Changes how inclusion and exclusion errors are calculated. Values:
        "perfect": no targeting errors, "prop_nonpoor_lms": , "data": , "x33": , "incl": , "excl":
    @param option_pds: Post disaster support options. Values: "unif_poor", "no", "prop", "prop_nonpoor", "unif_poor_only"
    @param option_b: Post disaster support budget option. Values: "data", "unif_poor", "max01", "max05", "unlimited",
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
    if option_t == "perfect":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 0
    elif option_t == "prop_nonpoor_lms":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 1 - 25 / 80  # 25% of pop chosen among top 80 DO receive the aid
    elif option_t == "data":
        # TODO: inclusion error can become negative, which results in negative n!!!
        macro_event_["error_incl"] = ((1 - macro_event_["prepare_scaleup"]) / 2 * macro_event_["fa"]
                                      / (1 - macro_event_["fa"]))  # as in equation 16 of the paper
        macro_event_["error_excl"] = (1 - macro_event_["prepare_scaleup"]) / 2  # as in equation 16 of the paper
    elif option_t == "x33":
        macro_event_["error_incl"] = .33 * macro_event_["fa"] / (1 - macro_event_["fa"])
        macro_event_["error_excl"] = .33
    elif option_t == "incl":
        macro_event_["error_incl"] = .33 * macro_event_["fa"] / (1 - macro_event_["fa"])
        macro_event_["error_excl"] = 0
    elif option_t == "excl":
        macro_event_["error_incl"] = 0
        macro_event_["error_excl"] = 0.33
    else:
        print("unrecognized targeting error option " + option_t)
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

    # TODO: add 'helped_cat' and 'affected_cat' to index and use IndexSlice indexing as in the commented section below
    # cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, :, ['a'], ['helped']], "n"] *= (1 - macro_event_["error_excl"])
    # cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, :, ['a'], ['not_helped']], "n"] *= macro_event_["error_excl"]
    # cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, :, ['na'], ['helped']], "n"] *= macro_event_["error_incl"]
    # cat_info_event_iah_.loc[pd.IndexSlice[:, :, :, :, ['na'], ['not_helped']], "n"] *= (1 - macro_event_["error_incl"])

    # !!!! n is one again from here.
    # print(cat_info_event_iah_.n.groupby(level=event_level).sum())

    # Step 0: define max_aid
    macro_event_["max_aid"] = (macro_event_["max_increased_spending"] * macro_event_["borrowing_ability"]
                               * macro_event_["gdp_pc_pp"])

    if option_fee == 'insurance_premium':
        cats_event_iah_pre_pds = cat_info_event_iah_.copy()

    # post disaster support (PDS) calculation depending on option_pds

    # Step 1: Compute the help needed for all helped households to fulfill the policy
    # TODO make sure to avoid NaNs in help_needed and help_received when creating these columns
    if option_pds == "no":
        macro_event_["aid"] = 0
        macro_event_['need'] = 0
        cat_info_event_iah_['help_needed'] = 0
        cat_info_event_iah_['help_received'] = 0
        option_b = 'no'
    elif option_pds == "unif_poor":
        # help_received for all helped hh = 80% of dk for poor, affected hh
        # share of losses to be covered * (losses of helped, affected, poor households)
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = macro_event_["shareable"] * cat_info_event_iah_.xs(('helped', 'a', 'q1'), level=('helped_cat', 'affected_cat', 'income_cat'))[loss_measure]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_needed"] = 0
    elif option_pds == "unif_poor_only":
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped') & (cat_info_event_iah_.income_cat == poor_cat), "help_needed"] = macro_event_["shareable"] * cat_info_event_iah_.xs(('helped', 'a', 'q1'), level=('helped_cat', 'affected_cat', 'income_cat'))[loss_measure]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped') | (~cat_info_event_iah_.income_cat == poor_cat), "help_received"] = 0
    #TODO: for below PDS options, need to check whether assigned values for help_needed are correctly implemented
    # elif option_pds == "prop_nonpoor":
    #     if "has_received_help_from_PDS_cat" not in cat_info_event_iah_.columns:
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (~cat_info_event_iah_.income_cat.isin(poor_cat)), loss_measure])
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped')
    #                             | (cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = 0
    #     else:
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (~cat_info_event_iah_.income_cat.isin(poor_cat))
    #                                                                 & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped')
    #                             | (cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = 0
    # elif option_pds == "prop":
    #     if "has_received_help_from_PDS_cat" not in cat_info_event_iah_.columns:
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                             & (cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (cat_info_event_iah_.income_cat.isin(poor_cat)), loss_measure])
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                             & (~cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (~cat_info_event_iah_.income_cat.isin(poor_cat)), loss_measure])
    #         cat_info_event_iah_.loc[cat_info_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0
    #     else:
    #         cat_info_event_iah_.loc[
    #             (cat_info_event_iah_.helped_cat == 'helped') & (
    #                         cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (cat_info_event_iah_.income_cat.isin(poor_cat))
    #                                                                 & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
    #         cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                             & (~cat_info_event_iah_.income_cat.isin(poor_cat)), "help_needed"] = (
    #                 macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
    #                                                                 & (cat_info_event_iah_.affected_cat == 'a')
    #                                                                 & (~cat_info_event_iah_.income_cat.isin(poor_cat))
    #                                                                 & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
    #         cat_info_event_iah_.loc[cat_info_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0

    # Step 2: total need (cost) for all helped hh = sum over help_needed for helped hh
    macro_event_["need"] = agg_to_event_level(cat_info_event_iah_, "help_needed", event_level)

    # actual aid reduced by capacity
    if option_b == "data":
        # Step 3: total need (cost) for all helped hh clipped at max_aid
        macro_event_["aid"] = (macro_event_["need"]
                               * macro_event_["prepare_scaleup"]
                               * macro_event_["borrowing_ability"]).clip(upper=macro_event_["max_aid"])
    elif option_b == "unif_poor":
        macro_event_["aid"] = macro_event_["need"].clip(upper=macro_event_["max_aid"])
    elif option_b == "max01":
        macro_event_["max_aid"] = 0.01 * macro_event_["gdp_pc_pp"]
        macro_event_["aid"] = (macro_event_["need"]).clip(upper=macro_event_["max_aid"])
    elif option_b == "max05":
        macro_event_["max_aid"] = 0.05 * macro_event_["gdp_pc_pp"]
        macro_event_["aid"] = (macro_event_["need"]).clip(upper=macro_event_["max_aid"])
    elif option_b == "unlimited":
        macro_event_["aid"] = macro_event_["need"]
    elif option_b == "one_per_affected":
        d = cat_info_event_iah_.loc[(cat_info_event_iah_.affected_cat == 'a')]
        d["un"] = 1
        macro_event_["need"] = agg_to_event_level(d, "un", event_level)
        macro_event_["aid"] = macro_event_["need"]
    elif option_b == "one_per_helped":
        d = cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')]
        d["un"] = 1
        macro_event_["need"] = agg_to_event_level(d, "un", event_level)
        macro_event_["aid"] = macro_event_["need"]
    elif option_b == "one":
        macro_event_["aid"] = 1
    elif option_b == 'no':
        pass
    else:
        print(f"Unknown optionB={option_b}")

    if option_pds == "unif_poor":
        # Step 4: help_received = unif_aid = aid/(N hh helped)
        macro_event_["unif_aid"] = (macro_event_["aid"] / (
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == "helped"), "n"].groupby(level=event_level).sum()
        )).fillna(0)  # division by zero is possible if no losses occur
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_received"] = 0
    elif option_pds == "unif_poor_only":
        macro_event_["unif_aid"] = macro_event_["aid"] / (
            cat_info_event_iah_.loc[
                (cat_info_event_iah_.helped_cat == "helped") & (cat_info_event_iah_.income_cat.isin(poor_cat)), "n"].groupby(
                level=event_level).sum())
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[
            (cat_info_event_iah_.helped_cat == 'not_helped') | (~cat_info_event_iah_.income_cat.isin(poor_cat)), "help_received"] = 0
    elif option_pds == "prop":
        cat_info_event_iah_["help_received"] = macro_event_["aid"] / macro_event_["need"] * cat_info_event_iah_["help_received"]

    # option_fee
    if option_fee == "tax":
        # help_fee is the share of the total help provided, distributed proportionally to the hh's capital shares
        cat_info_event_iah_["help_fee"] = (fraction_inside * macro_event_["aid"] * cat_info_event_iah_["k"]
                                           / agg_to_event_level(cat_info_event_iah_, "k", event_level))
    elif option_fee == "insurance_premium":
        # TODO: this is never used, because the case option_fee == "insurance_premium" is handled in calculate_response
        cat_info_event_iah_.loc[(cat_info_event_iah_.income_cat.isin(poor_cat)), "help_fee"] = fraction_inside * agg_to_event_level(
            cat_info_event_iah_.loc[cat_info_event_iah_.income_cat.isin(poor_cat)], 'help_received', event_level) / (cat_info_event_iah_.loc[cat_info_event_iah_.income_cat.isin(poor_cat)].n.sum())
        cat_info_event_iah_.loc[(~cat_info_event_iah_.income_cat.isin(poor_cat)), "help_fee"] = fraction_inside * agg_to_event_level(
            cat_info_event_iah_.loc[~cat_info_event_iah_.income_cat.isin(poor_cat)], 'help_received', event_level) / (cat_info_event_iah_.loc[~cat_info_event_iah_.income_cat.isin(poor_cat)].n.sum())
        cat_info_event_iah_[['help_received', 'help_fee']] += cats_event_iah_pre_pds[['help_received', 'help_fee']]

    cat_info_event_iah_.drop(['income_cat', 'helped_cat', 'affected_cat'], axis=1, inplace=True)
    return macro_event_, cat_info_event_iah_


def compute_dW(macro_event, cat_info_event_iah_):
    # compute post-support consumption losses including help received and help fee paid
    cat_info_event_iah_["dc_npv_post"] = (cat_info_event_iah_["dc_npv_pre"]
                                          - cat_info_event_iah_["help_received"]
                                          + cat_info_event_iah_["help_fee"])
    cat_info_event_iah_["dw"] = calc_delta_welfare(cat_info_event_iah_, macro_event)

    return cat_info_event_iah_


def optimize_recovery(macro_event, cat_info_event_iah, capital_t=50, delta_c_h_max=np.nan):
    opt_data = pd.merge(
        macro_event.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta',
                                    'reconstruction_share_sigma_h': 'sigma_h'}),
        cat_info_event_iah.rename(columns={'k': 'k_h_eff', 'dk': 'delta_k_h_eff', 'liquidity': 'savings_s_h',
                                           'help_received': 'delta_i_h_pds'}),
        left_index=True, right_index=True
    )[['productivity_pi', 'discount_rate_rho', 'eta', 'k_h_eff', 'delta_k_h_eff', 'savings_s_h',
       'sigma_h', 'delta_i_h_pds', 'delta_tax_sp', 'diversified_share']]
    opt_data['capital_t'] = capital_t
    opt_data['delta_c_h_max'] = delta_c_h_max
    opt_data = opt_data.xs('a', level='affected_cat', drop_level=False).round(10)
    recovery_rates_lambda = optimize_data(
        df_in=opt_data,
        tolerance=1e-2,
        min_lambda=.05,
        max_lambda=6,
    )
    return pd.merge(cat_info_event_iah, recovery_rates_lambda, left_index=True, right_index=True, how='left')


def compute_dw_reco_and_used_savings(cat_info_event_iah, macro_event, event_level_, capital_t=50, delta_c_h_max=np.nan):
    recovery_parameters = cat_info_event_iah.xs('a', level='affected_cat')[['n', 'dk', 'lambda_h']]
    recovery_parameters['dk_abs'] = recovery_parameters['dk'] * recovery_parameters['n']
    recovery_parameters.drop(['n', 'dk'], axis=1, inplace=True)
    recovery_parameters['recovery_params'] = recovery_parameters[['dk_abs', 'lambda_h']].apply(tuple, axis=1)
    recovery_parameters = recovery_parameters.groupby(level=event_level_).agg(list).recovery_params
    recompute_data = pd.merge(cat_info_event_iah, macro_event, left_index=True, right_index=True, how='left')
    recompute_data = recompute_data.rename(columns={'avg_prod_k': 'productivity_pi', 'rho': 'discount_rate_rho',
                                                    'tau_tax': 'delta_tax_sp', 'income_elasticity_eta': 'eta',
                                                    'reconstruction_share_sigma_h': 'sigma_h', 'k': 'k_h_eff',
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

    dw_ds_reco = recompute_data_with_tax(recompute_data)

    cat_info_event_iah = pd.merge(cat_info_event_iah, dw_ds_reco, left_index=True, right_index=True, how='left')
    cat_info_event_iah = pd.merge(cat_info_event_iah, recovery_parameters, left_index=True, right_index=True, how='left')

    return cat_info_event_iah


def compute_dw_long_term(cat_info_event_iah, macro_event, event_level):#, long_term_horizon_=None):
    cat_info_event_iah['dk_pub'] = cat_info_event_iah['dk'] * (1 - macro_event['reconstruction_share_sigma_h'])
    macro_event['delta_tax_pub'] = ((agg_to_event_level(cat_info_event_iah, 'dk_pub', event_level)
                                     / agg_to_event_level(cat_info_event_iah, 'k', event_level))
                                    / macro_event.avg_prod_k)
    # macro_event['delta_tax_pub'] = (agg_to_event_level(cat_info_event_iah, 'dk_pub', event_level)
    #                                 / agg_to_event_level(cat_info_event_iah, 'k', event_level))
    cat_info_event_iah['dc_long_term'] = (
            cat_info_event_iah['dS_reco'] + cat_info_event_iah['help_fee'] + cat_info_event_iah['c']
            * macro_event['delta_tax_pub'] - cat_info_event_iah['help_received']
    )
    cat_info_event_iah['dW_long_term'] = cat_info_event_iah['c'] ** (-macro_event['income_elasticity_eta']) * \
                                         cat_info_event_iah['dc_long_term']
    # if long_term_horizon_ is None:
    #     cat_info_event_iah['dW_long_term'] = cat_info_event_iah['c'] ** (-macro_event['income_elasticity_eta']) * cat_info_event_iah['dc_long_term']
    # else:
    #     merged = pd.merge(cat_info_event_iah, macro_event, left_index=True, right_index=True, how='left')
    #     cat_info_event_iah['dW_long_term'] = merged.apply(lambda x: calc_delta_welfare_discounted(x['c'], x['dc_long_term'] / long_term_horizon_, x['rho'], x['income_elasticity_eta'], long_term_horizon_), axis=1)
    return cat_info_event_iah, macro_event


def compute_dw_new(cat_info_event_iah, macro_event, event_level_, capital_t=50, delta_c_h_max=np.nan):
    # compute the optimal recovery rates
    cat_info_event_iah_ = optimize_recovery(
        macro_event=macro_event,
        cat_info_event_iah=cat_info_event_iah,
        capital_t=capital_t,
        delta_c_h_max=delta_c_h_max,
    )
    if cat_info_event_iah_.xs('a', level='affected_cat').lambda_h.isna().any():
        failed_optimizations = cat_info_event_iah_[cat_info_event_iah_['lambda_h'].isna()].xs('a', level='affected_cat').index.droplevel(['rp', 'helped_cat', 'income_cat']).unique()
        print(f"Failed to optimize recovery rates for {failed_optimizations}. Dropping entries.")
        cat_info_event_iah_ = cat_info_event_iah_.drop(failed_optimizations)

    # compute the welfare losses from destroyed assets, decreased transfers, and reconstruction
    cat_info_event_iah_ = compute_dw_reco_and_used_savings(cat_info_event_iah_, macro_event, event_level_, capital_t,
                                                           delta_c_h_max)

    # compute the long-term welfare losses from public asset reconstruction costs, PDS costs, and used savings
    cat_info_event_iah_, macro_event_ = compute_dw_long_term(cat_info_event_iah_, macro_event, event_level_)#, long_term_horizon_)

    # sum the welfare losses from reconstruction and long-term welfare losses
    cat_info_event_iah_['dw'] = cat_info_event_iah_['dW_reco'] + cat_info_event_iah_['dW_long_term']
    return cat_info_event_iah_, macro_event_


def prepare_output(macro, macro_event, cat_info_event_iah, econ_scope, event_level, hazard_protection_, default_rp,
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
    out = average_over_rp(out, default_rp, hazard_protection_)

    # Sums over hazard dk, dW (gets one line per economy)
    # TODO: average over axfin, social, gamma_SP does not really carry any meaning. Should be dropped.
    out = out.groupby(level=econ_scope).aggregate(
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


def calc_delta_welfare(micro, macro):
    """welfare cost from consumption before (c)
    and after (dc_npv_post) event. Line by line"""
    # computes welfare losses per category
    # as per eqs. 9-11 in the technical paper
    dw = (welf(micro["c"] / macro["rho"], macro["income_elasticity_eta"])
          - welf(micro["c"] / macro["rho"] - micro["dc_npv_post"], macro["income_elasticity_eta"]))  # w(c_0) - w(c_h)

    return dw


def welf(c, elast, marginal=False):
    """"Welfare function"""
    if not marginal:
        y = (c ** (1 - elast) - 1) / (1 - elast)
    else:
        y = c ** (-elast)
    return y


def discounted_w_of_t(t_, c_baseline_, delta_c_, elast, rho, marginal=False):
    return welf(c_baseline_ - delta_c_, elast, marginal) * np.exp(-rho * t_)


def calc_delta_welfare_discounted(c_0, delta_c, rho, eta, t_max):
    """Computes discounted welfare loss from consumption before (c_0) and after (delta_c) event"""
    w_baseline = integrate.quad(discounted_w_of_t, args=(c_0, 0, eta, rho), a=0, b=t_max)[0]
    w_disaster = integrate.quad(discounted_w_of_t, args=(c_0, delta_c, eta, rho), a=0, b=t_max)[0]
    return w_baseline - w_disaster


def calc_risk_and_resilience_from_k_w(df, is_local_welfare=True):#, long_term_horizon_=None):
    """Computes risk and resilience from dk, dw and protection.
    Line by line: multiple return periods or hazard is transparent to this function"""
    df = df.copy()

    # Expressing welfare losses in currency
    # discount rate

    # TODO: changed the definition of w_prime according to PHL paper
    # linearly approximated derivative of welfare with respect to NPV of future consumption
    # NO LONGER USING NPV OF FUTURE CONSUMPTION WITH THE MODEL UPDATE!
    # rho = df["rho"]
    # h = 1e-4
    # if is_local_welfare:
    #     w_prime = (welf(df["gdp_pc_pp"] / rho + h, df["income_elasticity_eta"])
    #                - welf(df["gdp_pc_pp"] / rho - h, df["income_elasticity_eta"])) / (2 * h)
    # else:
    #     w_prime = (welf(df["gdp_pc_pp_nat"] / rho + h, df["income_elasticity_eta"])
    #                - welf(df["gdp_pc_pp_nat"] / rho - h, df["income_elasticity_eta"])) / (2 * h)
    if is_local_welfare:
        w_prime = df["gdp_pc_pp"]**(-df["income_elasticity_eta"])
    else:
        w_prime = df["gdp_pc_pp_nat"]**(-df["income_elasticity_eta"])

    d_w_ref = w_prime * df["dk"]

    # if long_term_horizon_ is None:
    #     d_w_ref = w_prime * df["dk"]
    # else:
    #     d_w_ref = df.apply(lambda x: calc_delta_welfare_discounted(x['gdp_pc_pp'], x['dk'], x['rho'], x['income_elasticity_eta'], long_term_horizon_), axis=1)

    # expected welfare loss (per family and total)
    # TODO: @Bramka why does division by w prime result in a currency value?
    # this is to compute consumption equivalent of welfare loss!
    df["dWpc_currency"] = df["dw"] / w_prime
    df["dWtot_currency"] = df["dWpc_currency"] * df["pop"]

    # Risk to welfare as percentage of local GDP
    df["risk"] = df["dWpc_currency"] / df["gdp_pc_pp"]

    # socio-economic resilience
    # TODO: @Bramka in the paper, socio-econ. resilience = asset losses / welfare losses
    df["resilience"] = d_w_ref / df["dw"]

    # risk to assets
    # TODO: @Bramka this is the same as dk / gdp_pc_pp!
    df["risk_to_assets"] = df.resilience * df.risk

    return df
