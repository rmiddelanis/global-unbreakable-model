import copy

import numpy as np
import pandas as pd
from pandas_helper import broadcast_simple, concat_categories
from lib_gather_data import average_over_rp

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

    cat_info_event_ia.loc[pd.IndexSlice[:, :, :, :, 'na'], "dk"] = 0
    # cat_info_event_ia.loc[(cat_info_event_ia.affected_cat == 'na'), "dk"] = 0

    # "national" losses
    macro_event_["dk"] = agg_to_event_level(cat_info_event_ia, "dk", event_level)

    # immediate consumption losses: direct capital losses plus losses through event-scale depression of transfers
    cat_info_event_ia["dc"] = (
            (1 - macro_event_["tau_tax"]) * cat_info_event_ia["dk"]
            + cat_info_event_ia["gamma_SP"] * macro_event_["tau_tax"] * macro_event_["dk"]
    )

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    cat_info_event_ia["dc_npv_pre"] = cat_info_event_ia[["dc", "macro_multiplier_Gamma"]].product(axis=1)

    return macro_event_, cat_info_event_ia


def calculate_response(macro_event, cat_info_event_ia, event_level, helped_cats, poor_cats, option_fee="tax", option_t="data",
                       option_pds="unif_poor", option_b="data", loss_measure="dk", fraction_inside=1,
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
            poor_categories=poor_cats,
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
        m__, c__ = compute_response(macro_event, cat_info_event_iah, event_level, poor_categories=poor_cats, option_t="data", option_pds="unif_poor",
                                    option_b="data", option_fee="tax", fraction_inside=1, loss_measure="dk")
        # change column name helped_cat to has_received_help_from_PDS_cat
        c__h = c__.rename(columns=dict(helped_cat="has_received_help_from_PDS_cat"))

        cats_event_iah_h = concat_categories(c__h, c__h, index=helped_cats).reset_index(helped_cats.name).sort_index()

        # compute post disaster response with insurance ONLY
        macro_event, cat_info_event_iah = compute_response(macro_event.assign(shareable=share_insured), cats_event_iah_h,
                                                       event_level, poor_categories=poor_cats, option_t=option_t, option_pds=option_pds,
                                                       option_b=option_b, option_fee=option_fee,
                                                       fraction_inside=fraction_inside, loss_measure=loss_measure)

        columns_to_add = ["need", "aid"]
        macro_event[columns_to_add] += m__[columns_to_add]

    return macro_event, cat_info_event_iah


def compute_response(macro_event, cat_info_event_iah, event_level, poor_categories, option_t="data", option_pds="unif_poor", option_b="data",
                     option_fee="tax", fraction_inside=1, loss_measure="dk"):

    """Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing,
    and dimensioning of the help. Returns copies of macro_event and cats_event_iah updated with stuff.
    @param macro_event:
    @param cat_info_event_iah:
    @param event_level:
    @param poor_categories: list of income categories to be considered poor
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
    # TODO make sure to avoid NaNs in help_needed and help_received when creating these columns
    if option_pds == "no":
        macro_event_["aid"] = 0
        macro_event_['need'] = 0
        cat_info_event_iah_['help_needed'] = 0
        cat_info_event_iah_['help_received'] = 0
        option_b = 'no'
    elif option_pds == "unif_poor":
        # Step 1: help_received for all helped hh = 80% of dk for poor, affected hh
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                # share of losses to be covered * (losses of helped, affected, poor households)
                # TODO: check if this is correct: set help_needed only 'helped' households?! or rather all households
                #  that are affected and poor?
                macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (cat_info_event_iah_.income_cat.isin(poor_categories)), loss_measure])
        # cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_needed"] = 0
        # TODO: check if this is correct; all households that do not need help should be set to 0
        cat_info_event_iah_["help_needed"].fillna(0, inplace=True)
    elif option_pds == "unif_poor_only":
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                & (cat_info_event_iah_.affected_cat == 'a')
                                                                & (cat_info_event_iah_.income_cat.isin(poor_categories)), loss_measure])
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped')
                            | (~cat_info_event_iah_.income_cat.isin(poor_categories)), "help_received"] = 0
    elif option_pds == "prop_nonpoor":
        if "has_received_help_from_PDS_cat" not in cat_info_event_iah_.columns:
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (~cat_info_event_iah_.income_cat.isin(poor_categories)), loss_measure])
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped')
                                | (cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = 0
        else:
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (~cat_info_event_iah_.income_cat.isin(poor_categories))
                                                                    & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped')
                                | (cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = 0
    elif option_pds == "prop":
        if "has_received_help_from_PDS_cat" not in cat_info_event_iah_.columns:
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                & (cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (cat_info_event_iah_.income_cat.isin(poor_categories)), loss_measure])
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                & (~cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (~cat_info_event_iah_.income_cat.isin(poor_categories)), loss_measure])
            cat_info_event_iah_.loc[cat_info_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0
        else:
            cat_info_event_iah_.loc[
                (cat_info_event_iah_.helped_cat == 'helped') & (
                            cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (cat_info_event_iah_.income_cat.isin(poor_categories))
                                                                    & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                & (~cat_info_event_iah_.income_cat.isin(poor_categories)), "help_needed"] = (
                    macro_event_["shareable"] * cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped')
                                                                    & (cat_info_event_iah_.affected_cat == 'a')
                                                                    & (~cat_info_event_iah_.income_cat.isin(poor_categories))
                                                                    & (cat_info_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cat_info_event_iah_.loc[cat_info_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0

        # print(cat_info_event_iah_[['helped_cat','affected_cat','income_cat','help_needed','n']])

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
        )).fillna(0)    # if division by 0 occurs, there are no helped households; thus, set to 0
        # TODO: check if fillna(0) is necessary also for other options
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'not_helped'), "help_received"] = 0
    elif option_pds == "unif_poor_only":
        macro_event_["unif_aid"] = macro_event_["aid"] / (
            cat_info_event_iah_.loc[
                (cat_info_event_iah_.helped_cat == "helped") & (cat_info_event_iah_.income_cat.isin(poor_categories)), "n"].groupby(
                level=event_level).sum())
        cat_info_event_iah_.loc[(cat_info_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cat_info_event_iah_.loc[
            (cat_info_event_iah_.helped_cat == 'not_helped') | (~cat_info_event_iah_.income_cat.isin(poor_categories)), "help_received"] = 0
    elif option_pds == "prop":
        cat_info_event_iah_["help_received"] = macro_event_["aid"] / macro_event_["need"] * cat_info_event_iah_["help_received"]

    # option_fee
    if option_fee == "tax":
        # help_fee is the share of the total help provided, distributed proportionally to the hh's capital shares
        cat_info_event_iah_["help_fee"] = (fraction_inside * macro_event_["aid"] * cat_info_event_iah_["k"]
                                       / agg_to_event_level(cat_info_event_iah_, "k", event_level))
    elif option_fee == "insurance_premium":
        # TODO: this is never used, because the case option_fee == "insurance_premium" is handled in calculate_response
        cat_info_event_iah_.loc[(cat_info_event_iah_.income_cat.isin(poor_categories)), "help_fee"] = fraction_inside * agg_to_event_level(
            cat_info_event_iah_.loc[cat_info_event_iah_.income_cat.isin(poor_categories)], 'help_received', event_level) / (cat_info_event_iah_.loc[cat_info_event_iah_.income_cat.isin(poor_categories)].n.sum())
        cat_info_event_iah_.loc[(~cat_info_event_iah_.income_cat.isin(poor_categories)), "help_fee"] = fraction_inside * agg_to_event_level(
            cat_info_event_iah_.loc[~cat_info_event_iah_.income_cat.isin(poor_categories)], 'help_received', event_level) / (cat_info_event_iah_.loc[~cat_info_event_iah_.income_cat.isin(poor_categories)].n.sum())
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


def prepare_output(macro, macro_event, cat_info_event_iah, econ_scope, event_level, hazard_protection_, default_rp,
                   is_local_welfare=True, return_stats=True):
    # generate output df
    out = pd.DataFrame(index=macro_event.index)

    # pull 'aid' and 'dk' from macro_event
    out["average_aid_cost_pc"] = macro_event["aid"]
    out["dk"] = macro_event["dk"]

    # aggregate delta_W at event-level
    out["dw"] = agg_to_event_level(cat_info_event_iah, "dw", event_level)

    out["dk_tot"] = out["dk"] * macro_event["pop"]
    out["dw_tot"] = out["dw"] * macro_event["pop"]

    if return_stats:
        stats = np.setdiff1d(cat_info_event_iah.columns, event_level + ['helped_cat', 'affected_cat', 'income_cat',
                                                                        'has_received_help_from_PDS_cat'])
        df_stats = agg_to_event_level(cat_info_event_iah, stats, event_level)
        print("stats are " + ",".join(stats))
        out[df_stats.columns] = df_stats

    # aggregate losses
    # Averages over return periods to get dk_{hazard} and dW_{hazard}
    out = average_over_rp(out, default_rp, hazard_protection_)

    # Sums over hazard dk, dW (gets one line per economy)
    out = out.groupby(level=econ_scope).aggregate(
        {c: 'sum' if c not in ['axfin', 'social', 'gamma_SP'] else 'mean' for c in out.columns}
    )

    # adds dk and dw-like columns to macro
    out = pd.concat((macro, out), axis=1)

    # computes socio-economic capacity and risk at economy level
    out = calc_risk_and_resilience_from_k_w(df=out, is_local_welfare=is_local_welfare)

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
    dw = (welf(micro["c"] / macro["rho"], macro["income_elasticity"])
          - welf(micro["c"] / macro["rho"] - micro["dc_npv_post"], macro["income_elasticity"]))

    return dw


def welf(c, elast):
    """"Welfare function"""
    y = (c ** (1 - elast) - 1) / (1 - elast)
    return y


def calc_risk_and_resilience_from_k_w(df, is_local_welfare=True):
    """Computes risk and resilience from dk, dw and protection.
    Line by line: multiple return periods or hazard is transparent to this function"""
    df = df.copy()

    # Expressing welfare losses in currency
    # discount rate
    rho = df["rho"]
    h = 1e-4

    # linearly approximated derivative of welfare with respect to NPV of future consumption
    if is_local_welfare:
        w_prime = (welf(df["gdp_pc_pp"] / rho + h, df["income_elasticity"])
                   - welf(df["gdp_pc_pp"] / rho - h, df["income_elasticity"])) / (2 * h)
    else:
        w_prime = (welf(df["gdp_pc_pp_nat"] / rho + h, df["income_elasticity"])
                   - welf(df["gdp_pc_pp_nat"] / rho - h, df["income_elasticity"])) / (2 * h)

    # TODO: @Bramka why? assuming that in the reference case, capital loss equals consumption loss? The paper
    #  states that this would be the case for \mu = \rho
    d_w_ref = w_prime * df["dk"]

    # expected welfare loss (per family and total)
    # TODO: @Bramka why does division by w prime result in a currency value?
    # this is to compute consumption equivalent of welfare loss!
    df["dWpc_currency"] = df["dw"] / w_prime
    df["dWtot_currency"] = df["dWpc_currency"] * df["pop"]

    # Risk to welfare as percentage of local GDP
    df["risk"] = df["dWpc_currency"] / df["gdp_pc_pp"]

    # socio-economic capacity
    # TODO: @Bramka in the paper, socio-econ. resilience = asset losses / welfare losses
    df["resilience"] = d_w_ref / df["dw"]

    # risk to assets
    # TODO: @Bramka this is the same as d_w_ref / (w_prime * df["gdp_pc_pp"]). What does it mean?
    df["risk_to_assets"] = df.resilience * df.risk

    return df
