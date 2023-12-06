import copy

import numpy as np
import pandas as pd
from pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
from scipy.interpolate import interp1d
from lib_gather_data import social_to_tx_and_gsp

pd.set_option('display.width', 220)


def process_input(macro, cat_info, hazard_ratios, econ_scope, event_level, default_rp, verbose_replace=True):
    flag1 = False
    flag2 = False
    macro = macro.dropna()
    cat_info = cat_info.dropna()

    if type(hazard_ratios) is pd.DataFrame:
        # clean and harmonize data frames
        hazard_ratios = hazard_ratios.dropna()
        common_regions = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index]
        macro = macro.loc[common_regions]
        cat_info = cat_info.loc[common_regions]
        hazard_ratios = hazard_ratios.loc[common_regions]
        if hazard_ratios.empty:
            hazard_ratios = None

    # default hazard if no hazard is passed
    if hazard_ratios is None:
        hazard_ratios = pd.Series(
            data=1,
            index=pd.MultiIndex.from_product([macro.index, ["default_hazard"]], names=[econ_scope, "hazard"])
        )

    # if hazard data has no hazard, it is broadcasted to default hazard
    if "hazard" not in get_list_of_index_names(hazard_ratios):
        hazard_ratios = broadcast_simple(hazard_ratios, pd.Index(["default_hazard"], name="hazard"))

    # if hazard data has no rp, it is broadcasted to default rp
    if "rp" not in get_list_of_index_names(hazard_ratios):
        hazard_ratios_event = broadcast_simple(hazard_ratios, pd.Index([default_rp], name="rp"))
    else:
        # interpolates data to a more granular grid for return periods that includes all protection values that are
        # potentially not the same in hazard_ratios.
        hazard_ratios_event = interpolate_rps(hazard_ratios, macro.protection, default_rp=default_rp)

    # recompute
    # TODO: why recompute? The results appear to be the same as the original input...
    # here we assume that gdp = consumption = prod_from_k
    macro["gdp_pc_pp"] = macro["avg_prod_k"] * agg_to_economy_level(cat_info, "k", econ_scope)
    cat_info["c"] = ((1 - macro["tau_tax"]) * macro["avg_prod_k"] * cat_info["k"] +
                     cat_info["gamma_SP"] * macro["tau_tax"] * macro["avg_prod_k"]
                     * agg_to_economy_level(cat_info, "k", econ_scope))

    # add finance to diversification and taxation
    cat_info["social"] = unpack_social(macro, cat_info)
    cat_info["social"] += 0.1 * cat_info["axfin"]
    macro["tau_tax"], cat_info["gamma_SP"] = social_to_tx_and_gsp(econ_scope, cat_info)

    # Recompute consumption from k and new gamma_SP and tau_tax
    cat_info["c"] = ((1 - macro["tau_tax"]) * macro["avg_prod_k"] * cat_info["k"] +
                     cat_info["gamma_SP"] * macro["tau_tax"] * macro["avg_prod_k"]
                     * agg_to_economy_level(cat_info, "k", econ_scope))

    # rebuilding exponentially to 95% of initial stock in reconst_duration
    three = np.log(1 / 0.05)
    recons_rate = three / macro["T_rebuild_K"]

    # Calculation of macroeconomic resilience (Gamma in the technical paper)
    # \Gamma = (\mu + 3/N) / (\rho + 3/N)
    macro["macro_multiplier_Gamma"] = (macro["avg_prod_k"] + recons_rate) / (macro["rho"] + recons_rate)

    # FORMATING

    # gets the event level index
    # index composed on countries, hazards and rps.
    event_level_index = hazard_ratios_event.reset_index().set_index(event_level).index

    # Broadcast macro to event level
    macro_event = broadcast_simple(macro, event_level_index)

    # updates columns in macro with columns in hazard_ratios_event
    common_cols = [c for c in macro_event if c in hazard_ratios_event]  # cols in both macro_event, hazard_ratios_event
    if not common_cols == []:
        if verbose_replace:
            flag1 = True
            print("Replaced in macro: " + ", ".join(common_cols))
            macro_event[common_cols] = hazard_ratios_event[common_cols]

    # Broadcast categories to event level
    cats_event = broadcast_simple(cat_info, event_level_index)
    cats_event['v'] = hazard_ratios['v']
    print("pulling 'v' into cats_event from hazard_ratios")

    # updates columns in cats with columns in hazard_ratios_event
    # applies mh ratios to relevant columns
    cols_c = [c for c in cats_event if c in hazard_ratios_event]  # cols in both macro_event, hazard_ratios_event
    if not cols_c == []:
        hrb = broadcast_simple(hazard_ratios_event[cols_c], cat_info.index).reset_index().set_index(
            get_list_of_index_names(cats_event))  # explicitly broadcasts hazard ratios to contain income categories
        cats_event[cols_c] = hrb
        if verbose_replace:
            flag2 = True
            print("Replaced in cats: " + ", ".join(cols_c))
    if (flag1 and flag2):
        print("Replaced in both: " + ", ".join(np.intersect1d(common_cols, cols_c)))

    return macro_event, cats_event, hazard_ratios_event, macro


def compute_dK(macro_event, cats_event, event_level, affected_cats):
    cats_event_ia = concat_categories(cats_event, cats_event, index=affected_cats)
    # counts affected and non affected
    n_affected = cats_event["n"] * cats_event.fa
    n_not_affected = cats_event["n"] * (1 - cats_event.fa)
    cats_event_ia["n"] = concat_categories(n_affected, n_not_affected, index=affected_cats)

    # de_index so can access cats as columns and index is still event
    cats_event_ia = cats_event_ia.reset_index(["income_cat", "affected_cat"]).sort_index()

    # actual vulnerability
    # equation 12: V_a = V(1 - HFA_P2C3 / 5 * \pi)
    cats_event_ia["v_shew"] = cats_event_ia["v"] * (1 - macro_event["pi"] * cats_event_ia["shew"])

    # capital losses and total capital losses
    cats_event_ia["dk"] = cats_event_ia[["k", "v_shew"]].prod(axis=1, skipna=False)  # capital potentially be damaged

    cats_event_ia.loc[(cats_event_ia.affected_cat == 'na'), "dk"] = 0

    # "national" losses
    macro_event["dk_event"] = agg_to_event_level(cats_event_ia, "dk", event_level)

    # immediate consumption losses: direct capital losses plus losses through event-scale depression of transfers
    cats_event_ia["dc"] = (
            (1 - macro_event["tau_tax"]) * cats_event_ia["dk"]
            + cats_event_ia["gamma_SP"] * macro_event["tau_tax"] * macro_event["dk_event"]
    )

    # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
    cats_event_ia["dc_npv_pre"] = cats_event_ia["dc"] * macro_event["macro_multiplier_Gamma"]

    return macro_event, cats_event_ia


def calculate_response(macro_event, cats_event_ia, event_level, helped_cats, option_fee="tax", option_t="data",
                       option_pds="unif_poor", option_b="data", loss_measure="dk", fraction_inside=1,
                       share_insured=.25):
    cats_event_iah = concat_categories(cats_event_ia, cats_event_ia, index=helped_cats).reset_index(
        helped_cats.name).sort_index()
    cats_event_iah["help_received"] = 0.0
    cats_event_iah["help_fee"] = 0.0

    # baseline case (no insurance)
    if option_fee != "insurance_premium":
        macro_event, cats_event_iah = compute_response(macro_event, cats_event_iah, event_level, option_t=option_t,
                                                       option_pds=option_pds, option_b=option_b, option_fee=option_fee,
                                                       fraction_inside=fraction_inside, loss_measure=loss_measure)

    # special case of insurance that adds to existing default PDS
    else:
        # compute post disaster response with default PDS from data ONLY
        m__, c__ = compute_response(macro_event, cats_event_iah, event_level, option_t="data", option_pds="unif_poor",
                                    option_b="data", option_fee="tax", fraction_inside=1, loss_measure="dk")
        # change column name helped_cat to has_received_help_from_PDS_cat
        c__h = c__.rename(columns=dict(helped_cat="has_received_help_from_PDS_cat"))

        cats_event_iah_h = concat_categories(c__h, c__h, index=helped_cats).reset_index(helped_cats.name).sort_index()

        # compute post disaster response with insurance ONLY
        macro_event, cats_event_iah = compute_response(macro_event.assign(shareable=share_insured), cats_event_iah_h,
                                                       event_level, option_t=option_t, option_pds=option_pds,
                                                       option_b=option_b, option_fee=option_fee,
                                                       fraction_inside=fraction_inside, loss_measure=loss_measure)

        columns_to_add = ["need", "aid"]
        macro_event[columns_to_add] += m__[columns_to_add]

    return macro_event, cats_event_iah


def compute_response(macro_event, cats_event_iah, event_level, option_t="data", option_pds="unif_poor", option_b="data",
                     option_fee="tax", fraction_inside=1, loss_measure="dk"):

    """Computes aid received,  aid fee, and other stuff, from losses and PDS options on targeting, financing,
    and dimensioning of the help. Returns copies of macro_event and cats_event_iah updated with stuff.
    @param macro_event:
    @param cats_event_iah:
    @param event_level:
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
    cats_event_iah_ = cats_event_iah.copy()

    # because cats_event_ia is duplicated in cats_event_iah_, cats_event_iah_.n.groupby(level=event_level).sum() is
    # 2 instead of 1, here /2 is to correct it. macro_event_["fa"] =  agg_to_event_level(cats_event_ia,"fa") would work
    # but needs to pass a new variable cats_event_ia.
    macro_event_["fa"] = agg_to_event_level(cats_event_iah_, "fa", event_level) / 2

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
    cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped') & (cats_event_iah_.affected_cat == 'a'), "n"] *= (
            1 - macro_event_["error_excl"])
    cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped') & (cats_event_iah_.affected_cat == 'a'), "n"] *= (
        macro_event_["error_excl"])
    cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped') & (cats_event_iah_.affected_cat == 'na'), "n"] *= (
        macro_event_["error_incl"])
    cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped') & (cats_event_iah_.affected_cat == 'na'), "n"] *= (
            1 - macro_event_["error_incl"])
    # !!!! n is one again from here.
    # print(cats_event_iah_.n.groupby(level=event_level).sum())

    # Step 0: define max_aid
    macro_event_["max_aid"] = (macro_event_["max_increased_spending"] * macro_event_["borrow_abi"]
                               * macro_event_["gdp_pc_pp"])

    if option_fee == 'insurance_premium':
        cats_event_iah_pre_pds = cats_event_iah_.copy()

    # post disaster support (PDS) calculation depending on option_pds
    if option_pds == "no":
        macro_event_["aid"] = 0
        macro_event_['need'] = 0
        cats_event_iah_['help_needed'] = 0
        cats_event_iah_['help_received'] = 0
        option_b = 'no'
    elif option_pds == "unif_poor":
        # Step 1: help_received for all helped hh = 80% of dk for poor, affected hh
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                # share of losses to be covered * (losses of helped, affected, poor households)
                macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                & (cats_event_iah_.affected_cat == 'a')
                                                                & (cats_event_iah_.income_cat == 'poor'), loss_measure])
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped'), "help_needed"] = 0
    elif option_pds == "unif_poor_only":
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                & (cats_event_iah_.affected_cat == 'a')
                                                                & (cats_event_iah_.income_cat == 'poor'), loss_measure])
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped')
                            | (cats_event_iah_.income_cat == 'nonpoor'), "help_received"] = 0
    elif option_pds == "prop_nonpoor":
        if "has_received_help_from_PDS_cat" not in cats_event_iah_.columns:
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'nonpoor'), loss_measure])
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped')
                                | (cats_event_iah_.income_cat == 'poor'), "help_needed"] = 0
        else:
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'nonpoor')
                                                                    & (cats_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped')
                                | (cats_event_iah_.income_cat == 'poor'), "help_needed"] = 0
    elif option_pds == "prop":
        if "has_received_help_from_PDS_cat" not in cats_event_iah_.columns:
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                & (cats_event_iah_.income_cat == 'poor'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'poor'), loss_measure])
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                & (cats_event_iah_.income_cat == 'nonpoor'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'nonpoor'), loss_measure])
            cats_event_iah_.loc[cats_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0
        else:
            cats_event_iah_.loc[
                (cats_event_iah_.helped_cat == 'helped') & (cats_event_iah_.income_cat == 'poor'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'poor')
                                                                    & (cats_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                & (cats_event_iah_.income_cat == 'nonpoor'), "help_needed"] = (
                    macro_event_["shareable"] * cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')
                                                                    & (cats_event_iah_.affected_cat == 'a')
                                                                    & (cats_event_iah_.income_cat == 'nonpoor')
                                                                    & (cats_event_iah_.has_received_help_from_PDS_cat == 'helped'), loss_measure])
            cats_event_iah_.loc[cats_event_iah_.helped_cat == 'not_helped', "help_needed"] = 0

    # print(cats_event_iah_[['helped_cat','affected_cat','income_cat','help_needed','n']])

    # Step 2: total need (cost) for all helped hh = sum over help_needed for helped hh
    macro_event_["need"] = agg_to_event_level(cats_event_iah_, "help_needed", event_level)

    # actual aid reduced by capacity
    if option_b == "data":
        # Step 3: total need (cost) for all helped hh clipped at max_aid
        macro_event_["aid"] = (macro_event_["need"]
                               * macro_event_["prepare_scaleup"]
                               * macro_event_["borrow_abi"]).clip(upper=macro_event_["max_aid"])
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
        d = cats_event_iah_.loc[(cats_event_iah_.affected_cat == 'a')]
        d["un"] = 1
        macro_event_["need"] = agg_to_event_level(d, "un", event_level)
        macro_event_["aid"] = macro_event_["need"]
    elif option_b == "one_per_helped":
        d = cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped')]
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
        macro_event_["unif_aid"] = macro_event_["aid"] / (
            cats_event_iah_.loc[(cats_event_iah_.helped_cat == "helped"), "n"].groupby(level=event_level).sum()
        )
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'not_helped'), "help_received"] = 0
    elif option_pds == "unif_poor_only":
        macro_event_["unif_aid"] = macro_event_["aid"] / (
            cats_event_iah_.loc[
                (cats_event_iah_.helped_cat == "helped") & (cats_event_iah_.income_cat == 'poor'), "n"].groupby(
                level=event_level).sum())
        cats_event_iah_.loc[(cats_event_iah_.helped_cat == 'helped'), "help_received"] = macro_event_["unif_aid"]
        cats_event_iah_.loc[
            (cats_event_iah_.helped_cat == 'not_helped') | (cats_event_iah_.income_cat == 'nonpoor'), "help_received"] = 0
    elif option_pds == "prop":
        cats_event_iah_["help_received"] = macro_event_["aid"] / macro_event_["need"] * cats_event_iah_["help_received"]

    # option_fee
    if option_fee == "tax":
        # help_fee is the share of the total help provided, distributed proportionally to the hh's capital shares
        cats_event_iah_["help_fee"] = (fraction_inside * macro_event_["aid"] * cats_event_iah_["k"]
                                       / agg_to_event_level(cats_event_iah_, "k", event_level))
    elif option_fee == "insurance_premium":
        # TODO: this is never used, because the case option_fee == "insurance_premium" is handled in calculate_response
        cats_event_iah_.loc[(cats_event_iah_.income_cat == 'poor'), "help_fee"] = fraction_inside * agg_to_event_level(
            cats_event_iah_.query("income_cat=='poor'"), 'help_received', event_level) / (cats_event_iah_.query(
            "income_cat=='poor'").n.sum())
        cats_event_iah_.loc[(cats_event_iah_.income_cat == 'nonpoor'), "help_fee"] = fraction_inside * agg_to_event_level(
            cats_event_iah_.query("income_cat=='nonpoor'"), 'help_received', event_level) / (cats_event_iah_.query(
            "income_cat=='nonpoor'").n.sum())
        cats_event_iah_[['help_received', 'help_fee']] += cats_event_iah_pre_pds[['help_received', 'help_fee']]
    return macro_event_, cats_event_iah_


def compute_dW(macro_event, cats_event_iah, event_level, return_stats=True, return_iah=True):
    # compute post-support consumption losses including help received and help fee paid
    cats_event_iah["dc_npv_post"] = (cats_event_iah["dc_npv_pre"]
                                     - cats_event_iah["help_received"]
                                     + cats_event_iah["help_fee"])
    cats_event_iah["dw"] = calc_delta_welfare(cats_event_iah, macro_event)

    # aggregates dK and delta_W at event-level
    d_k = agg_to_event_level(cats_event_iah, "dk", event_level)
    d_w = agg_to_event_level(cats_event_iah, "dw", event_level)

    # generate output df
    df_out = pd.DataFrame(index=macro_event.index)
    df_out["dK"] = d_k
    df_out["dKtot"] = d_k * macro_event["pop"]
    df_out["delta_W"] = d_w
    df_out["delta_W_tot"] = d_w * macro_event["pop"]
    df_out["average_aid_cost_pc"] = macro_event["aid"]

    if return_stats:
        if "has_received_help_from_PDS_cat" not in cats_event_iah.columns:
            stats = np.setdiff1d(cats_event_iah.columns, event_level + ['helped_cat', 'affected_cat', 'income_cat'])
        else:
            stats = np.setdiff1d(cats_event_iah.columns, event_level + ['helped_cat', 'affected_cat', 'income_cat',
                                                                        'has_received_help_from_PDS_cat'])

        df_stats = agg_to_event_level(cats_event_iah, stats, event_level)
        # if verbose_replace:
        print("stats are " + ",".join(stats))
        df_out[df_stats.columns] = df_stats

    if return_iah:
        return df_out, cats_event_iah
    else:
        return df_out


def process_output(macro, out, macro_event, econ_scope, default_rp, is_local_welfare=True):
    # aggregate losses
    # Averages over return periods to get dk_{hazard} and dW_{hazard}
    # TODO: flopros protection seems to have gotten lost at some point?!
    dkdw_h = average_over_rp(out, default_rp, macro_event["protection"])

    # Sums over hazard dk, dW (gets one line per economy)
    dkdw = dkdw_h.groupby(level=econ_scope).sum()

    # summing over hazards gives wrong cat_info data which needs to be corrected for
    for i in ['axfin', 'social', 'gamma_SP']:
        dkdw[i] = dkdw_h[i].groupby(level=econ_scope).mean()

    # adds dk and dw-like columns to macro
    macro[dkdw.columns] = dkdw

    # computes socio-economic capacity and risk at economy level
    macro = calc_risk_and_resilience_from_k_w(df=macro, is_local_welfare=is_local_welfare)

    return macro


def unpack_social(macro, cat_info):
    """Compute social from gamma_SP, taux tax and k and avg_prod_k"""
    c = cat_info.c
    gs = cat_info.gamma_SP

    # gdp*tax should give the total social protection. gs=each one's social protection/(total social protection).
    # social is defined as t(=social protection)/c_i(=consumption)
    social = gs * macro.gdp_pc_pp * macro.tau_tax / c
    return social


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
        hazard_ratios_ = hazard_ratios_.unstack("rp")
        flag_stack = True

    if type(protection_list_) in [pd.Series, pd.DataFrame]:
        protection_list_ = protection_list_.squeeze().unique().tolist()

    # in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(hazard_ratios_.columns) is pd.MultiIndex:
        keys = hazard_ratios_.columns.get_level_values(0).unique()
        return pd.concat(
            {col: interpolate_rps(hazard_ratios_[col], protection_list_, default_rp) for col in keys},
            axis=1
        ).stack("rp")

    # actual function
    # figures out all the return periods to be included
    all_rps = list(set(protection_list_ + hazard_ratios_.columns.tolist()))

    fa_ratios_rps = hazard_ratios_.copy()

    # extrapolates linearly towards the 0 return period exposure (this creates negative exposure that is tackled after
    # interp.) (mind the 0 rp when computing probabilities)
    if len(fa_ratios_rps.columns) == 1:
        fa_ratios_rps[0] = fa_ratios_rps.squeeze()
    else:
        fa_ratios_rps[0] = (
            # exposure of smallest return period
            fa_ratios_rps.iloc[:, 0] -
            # smallest return period * (exposure of second-smallest rp - exposure of smallest rp) /
            fa_ratios_rps.columns[0] * (fa_ratios_rps.iloc[:, 1] - fa_ratios_rps.iloc[:, 0]) /
            # (second-smallest rp - smallest rp)
            (fa_ratios_rps.columns[1] - fa_ratios_rps.columns[0])
        )

    # add new, interpolated values for fa_ratios, assuming constant exposure on the right
    x = fa_ratios_rps.columns.values
    y = fa_ratios_rps.values
    fa_ratios_rps = pd.DataFrame(
        data=interp1d(x, y, bounds_error=False)(all_rps),
        index=fa_ratios_rps.index,
        columns=all_rps
    ).sort_index(axis=1).clip(lower=0).ffill(axis=1)
    fa_ratios_rps.columns.name = "rp"

    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack("rp")

    return fa_ratios_rps


def agg_to_economy_level(df, seriesname, economy):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T * df["n"]).T.groupby(level=economy).sum()


def agg_to_event_level(df, seriesname, event_level):
    """ aggregates seriesname in df (string of list of string) to event level (country, hazard, rp) across income_cat and affected_cat using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T * df["n"]).T.groupby(level=event_level).sum()


def calc_delta_welfare(micro, macro):
    """welfare cost from consumption before (c)
    and after (dc_npv_post) event. Line by line"""
    # computes welfare losses per category
    # as per eqs. 9-11 in the technical paper
    dw = (welf(micro["c"] / macro["rho"], macro["income_elast"])
          - welf(micro["c"] / macro["rho"] - micro["dc_npv_post"], macro["income_elast"]))

    return dw


def welf(c, elast):
    """"Welfare function"""
    y = (c ** (1 - elast) - 1) / (1 - elast)
    return y


def average_over_rp(df, default_rp, protection=None):
    """Aggregation of the outputs over return periods"""
    if protection is None:
        protection = pd.Series(0, index=df.index)

    # just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values("rp"):
        print("default_rp detected, droping rp")
        return (df.T / protection).T.reset_index("rp", drop=True)

    df = df.copy().reset_index("rp")
    protection = protection.copy().reset_index("rp", drop=True)

    # computes frequency of each return period
    return_periods = np.unique(df["rp"].dropna())

    proba = pd.Series(np.diff(np.append(1 / return_periods, 0)[::-1])[::-1],
                      index=return_periods)  # removes 0 from the rps

    # matches return periods and their frequency
    proba_serie = df["rp"].replace(proba)

    # removes events below the protection level
    proba_serie[protection > df.rp] = 0

    # handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels == [0]:
        idxlevels = 0

    # average weighted by proba
    averaged = df.mul(proba_serie, axis=0).groupby(
        level=idxlevels).sum()  # frequency times each variables in the columns including rp.

    return averaged.drop("rp", axis=1)  # here drop rp.


def calc_risk_and_resilience_from_k_w(df, is_local_welfare=True):
    """Computes risk and resilience from dk, dw and protection.
    Line by line: multiple return periods or hazard is transparent to this function"""
    df = df.copy()

    # Expressing welfare losses in currency
    # discount rate
    rho = df["rho"]
    h = 1e-4

    # linearly approximated derivative of welfare with respect to consumption
    if is_local_welfare:
        w_prime = (welf(df["gdp_pc_pp"] / rho + h, df["income_elast"])
                   - welf(df["gdp_pc_pp"] / rho - h, df["income_elast"])) / (2 * h)
    else:
        w_prime = (welf(df["gdp_pc_pp_nat"] / rho + h, df["income_elast"])
                   - welf(df["gdp_pc_pp_nat"] / rho - h, df["income_elast"])) / (2 * h)

    # TODO: why? assuming that in the reference case, capital loss equals consumption loss? The paper states that this
    #   would be the case for \mu = \rho
    d_w_ref = w_prime * df["dK"]

    # expected welfare loss (per family and total)
    # TODO: why does division by w prime result in a currency value?
    df["dWpc_currency"] = df["delta_W"] / w_prime
    df["dWtot_currency"] = df["dWpc_currency"] * df["pop"]

    # Risk to welfare as percentage of local GDP
    df["risk"] = df["dWpc_currency"] / df["gdp_pc_pp"]

    # socio-economic capacity
    # TODO: in the paper, socio-econ. resilience = asset losses / welfare losses
    df["resilience"] = d_w_ref / df["delta_W"]

    # risk to assets
    # TODO: this is the same as d_w_ref / (w_prime * df["gdp_pc_pp"]). What does it mean?
    df["risk_to_assets"] = df.resilience * df.risk

    return df
