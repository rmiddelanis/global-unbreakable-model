from pandas_helper import *
import os
from lib_compute_resilience_and_risk import agg_to_event_level

def apply_policy(macro_, cat_info_, hazard_ratios_, event_level, policy_name=None, policy_opt=None, verbose=True):
    """
    Chooses a policy by name, applies it to macro, cat_info, and/or hazard_ratios, and returns new values as well
    as a policy description
    """

    # duplicate inputes
    macro = macro_.copy(deep=True)
    cat_info = cat_info_.copy(deep=True)
    hazard_ratios = hazard_ratios_.copy(deep=True)
    desc = ''

    if policy_name is None:
        desc = "Baseline"

    elif policy_name == 'borrowing_abibility':
        macro.borrowing_abibility = 2
        desc = 'Increase borrowing_abibility to 2 for all countries '

    # Pov reduction
    elif policy_name == "kp":
        cat_info.k = cat_info.k.unstack().assign(q1=lambda df: df.q1 * 1.1).stack()
        desc = "Increase income of the poor 10%"  # Pov reduction
        # TODO: description says income, but does this not increase capital?!
        # TODO: currently, only 'q1' is considered poor. Adjust this to be more flexible, using variable
        #  'poor_categories'

    elif policy_name == "pov_head":
        cat_info.n = cat_info.n.unstack().assign(q1=.18, q2=.22).stack()
        desc = "Reduce poverty from 20% to 18%"
        # TODO: with five income categories, decide how to distribute the 2% reduction; for now, adding to q2
        # TODO: currently, only 'q1' is considered poor. Adjust this to be more flexible, using variable
        #  'poor_categories'

    # Borrow abi
    elif policy_name == "borrowing_abibility":
        macro.borrowing_abibility = macro.borrowing_abibility.clip(lower=1)
        desc = "Develop contingent finance and reserve funds"

    # Scale up abi
    elif policy_name == "prepare_scaleup":
        macro.prepare_scaleup = 1
        desc = "Make social protection scalable after a shock"

    # Early warnings to 1, except for earthquake
    elif policy_name == "ew":
        hazard_ratios.ew = 1
        hazard_ratios.loc[pd.IndexSlice[:, 'Earthquake', :, :], 'ew'] = 0
        desc = "Universal access to early warnings"

    # reconstruction to X years
    elif policy_name == "T_rebuild_K":
        macro.T_rebuild_K = policy_opt
        desc = "Accelerate reconstruction (by some%)"
        # TODO: T_rebuild_K will be moved to another df, since it will be income-dependent

    # social_p needs to be at least 50%
    elif policy_name == "social_p":
        macro, cat_info = clip_social_p(macro, cat_info, fun=lambda x: x.clip(lower=.33))
        desc = "Increase social transfers to poor people to at least 33%"
        # TODO: needs adapting for five income categories

    elif policy_name == "axfin":
        cat_info.axfin = 1
        desc = "Universal access to finance"

    # vpoor = vnonpoor
    elif policy_name == "vpvr":
        # cat_info.v = cat_info.v.unstack().assign(poor=lambda x: x.nonpoor).stack()
        hazard_ratios.loc[pd.IndexSlice[:, :, :, ['q1']], 'v'] = hazard_ratios.loc[
            pd.IndexSlice[:, :, :, ['q2', 'q3', 'q4', 'q5']], 'v'].groupby(['iso3', 'hazard', 'rp']).mean()
        desc = "Make poor people's assets as resistant as nonpoor people's assets"
        # TODO: currently, only 'q1' is considered poor. Adjust this to be more flexible, using variable
        #  'poor_categories'

    # 30% or poor people see their v reduced 30% (not 10pts!!)
    elif policy_name == "vp":
        n = cat_info.n.unstack().q1
        dv = .3  # reduction in v
        f = .05  # fractionof nat pop would get the reduction
        hazard_ratios.v = hazard_ratios.v.unstack().assign(q1=lambda df: (df.q1 * (1 - dv * f / n))).stack().clip(lower=0)
        desc = "Reduce asset\nvulnerability\n(by 30%) of\npoor people\n(5% of the population)"
        # TODO: currently, only 'q1' is considered poor. Adjust this to be more flexible, using variable
        #  'poor_categories'

    # Borrow abi
    elif policy_name == "bbb_incl":
        macro.borrowing_abibility = policy_opt
        macro.prepare_scaleup = policy_opt

    # reconstruction to X years
    elif policy_name == "bbb_fast":
        macro.T_rebuild_K = policy_opt
        # TODO: T_rebuild_K will be moved to another df, since it will be income-dependent


    # previously affected people see their v reduced 30%
    elif 'bbb' in policy_name:
        # hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info['v'].reset_index(),on=['country','income_cat'])

        if policy_name == "bbb":
            dv = policy_opt  # reduction in v
            hazard_ratios.v = hazard_ratios.v * (1 - dv)
            desc = "Reduce asset\nvulnerability\n(by 30%)"

        elif policy_name == "bbb_uncor":
            disaster_years = 20
            dv = policy_opt  # reduction in v
            hr_ = hazard_ratios.reset_index()
            for i in range(disaster_years):
                hr_.loc[hr_.rp == 1, 'v'] *= hr_.fa * (1 - dv) + (1 - hr_.fa)
                hr_.loc[hr_.rp != 1, 'v'] *= 1 - (1 / hr_.rp.astype('int')) * (hr_.fa * (1 - dv) + (1 - hr_.fa))
            hazard_ratios = hr_.set_index(['iso3', 'hazard', 'rp', 'income_cat'])

        elif policy_name == "bbb_cor":
            disaster_years = 20
            dv = policy_opt  # reduction in v
            hr_ = hazard_ratios.reset_index()
            for i in range(disaster_years):
                hr_.loc[hr_.rp == 1, 'v'] *= (1 - dv)
                hr_.loc[hr_.rp != 1, 'v'] *= 1 - (1 / hr_.rp.astype('int')) * (1 - dv)
            hazard_ratios = hr_.set_index(['iso3', 'hazard', 'rp', 'income_cat'])

        elif policy_name == "bb_standard":
            disaster_years = 20
            hazard_ratios.fa *= (1 - hazard_ratios.fa) ** disaster_years

        elif policy_name == 'bbb_50yrstand':
            n_years = 20
            hazard_ratios = hazard_ratios.reset_index()
            # hazard_ratios['p_annual_occur'] = (1./hazard_ratios.rp)
            # hazard_ratios['exp_val'] = n_years*hazard_ratios['p_annual_occur']
            # ^ expectation value of binomial dist is np
            # hazard_ratios['fa_scale_fac'] = (1.-hazard_ratios['fa'])**hazard_ratios['exp_val']
            # hazard_ratios['cum_fa_scale_fac'] = hazard_ratios.groupby(['country','hazard','income_cat'])['fa_scale_fac'].transform('prod')
            # These 2 approaches (above and below) produce near-identical (within 0.1%) results.
            # hazard_ratios['dfa_1yr'] = (1.-hazard_ratios.fa/hazard_ratios.rp)
            # scale_fac on fa, including probability that the event occurs in a single year
            # --> but this doesn't include the fact that fa is going down each year...

            hazard_ratios['tmp_fa'] = hazard_ratios['fa'].copy()
            for iyr in range(n_years):
                hazard_ratios['tmp_fa'] *= (1. - hazard_ratios['tmp_fa'] / hazard_ratios['rp'])
            # calculate scale fac on fa after n_years
            # --> updates tmp_fa each step of the way, so this is recursive

            hazard_ratios['dfa'] = hazard_ratios['tmp_fa'] / hazard_ratios['fa']
            hazard_ratios['cum_dfa'] = hazard_ratios.groupby(['iso3', 'hazard', 'income_cat'])['dfa'].transform('prod')
            # cum_scale_fac includes all rps and their probabilities...
            # --> If a hh is affected by the 1-year or the 2000-year event, it rebuilds in such a way that it is invulnerable to the 50-year event

            hazard_ratios.loc[hazard_ratios.rp <= 50, 'fa'] *= hazard_ratios.loc[hazard_ratios.rp <= 50, 'cum_dfa']

            hazard_ratios = hazard_ratios.drop(['dfa', 'tmp_fa', 'cum_dfa'], axis=1)
            hazard_ratios.set_index(['iso3', 'hazard', 'rp', 'income_cat'], inplace=True)
            # print(hazard_ratios.head())

        elif policy_name == 'bbb_complete':
            macro.borrowing_abibility = 1
            macro.prepare_scaleup = 1
            macro.T_rebuild_K = policy_opt
            n_years = 20
            hazard_ratios = hazard_ratios.reset_index()

            hazard_ratios['tmp_fa'] = hazard_ratios['fa'].copy()
            for iyr in range(n_years): hazard_ratios['tmp_fa'] *= (1. - hazard_ratios['tmp_fa'] / hazard_ratios['rp'])

            hazard_ratios['dfa'] = hazard_ratios['tmp_fa'] / hazard_ratios['fa']
            hazard_ratios['cum_dfa'] = hazard_ratios.groupby(['iso3', 'hazard', 'income_cat'])['dfa'].transform('prod')
            hazard_ratios.loc[hazard_ratios.rp <= 50, 'fa'] *= hazard_ratios.loc[hazard_ratios.rp <= 50, 'cum_dfa']

            hazard_ratios = hazard_ratios.drop(['dfa', 'tmp_fa', 'cum_dfa'], axis=1)
            hazard_ratios.set_index(['iso3', 'hazard', 'rp', 'income_cat'], inplace=True)
            # print(hazard_ratios.head())
        # build back better & faster - previously affected people see their v reduced 50%, T_rebuild is reduced too
        elif policy_name == "bbbf":
            macro.T_rebuild_K = 3 - (policy_opt * 5)
            dv = policy_opt  # reduction in v
            hazard_ratios.v = hazard_ratios.v * (dv - 1)

        hazard_ratios = hazard_ratios.reset_index().set_index(['iso3', 'hazard', 'rp', 'income_cat']).drop('index', axis=1)

    # TODO: need to revise the following policies; appear to be specific to the bbb report
    elif policy_name == "20dK":
        model = os.getcwd()
        intermediate = model + '/intermediate/'
        economy = "iso3"
        # print "OK up to here1"
        the_bbb_file = intermediate + "K_inputs.csv"
        # print "OK up to here1.5"
        df_bbb = pd.read_csv(the_bbb_file).set_index(economy)
        # print "OK up to here2"
        df_bbb["20dKtot"] = pd.read_csv(intermediate + "K_inputs.csv").set_index(economy)["20dKtot"]  # read data
        df_bbb["K_stock"] = pd.read_csv(intermediate + "K_inputs.csv").set_index(economy)["K"]
        # print "OK up to here3"
        twdKtot = df_bbb["20dKtot"]
        K_stock = df_bbb["K_stock"]
        dv = policy_opt  # reduction in v
        cat_info.v = (twdKtot / K_stock) * (1 - dv) * cat_info.v + (1 - twdKtot / K_stock) * cat_info.v
        # print K_stock

    elif policy_name == "dK_rp20":
        model = os.getcwd()
        intermediate = model + '/intermediate/'
        economy = "iso3"
        the_bbb_file = intermediate + "K_inputs.csv"
        df_bbb = pd.read_csv(the_bbb_file).set_index(economy)
        df_bbb["dKtot_rp20"] = pd.read_csv(intermediate + "K_inputs.csv").set_index(economy)["dKtot_rp20"]  # read data
        df_bbb["K_stock"] = pd.read_csv(intermediate + "K_inputs.csv").set_index(economy)["K"]
        dK_rp20 = df_bbb["dKtot_rp20"]
        K_stock = df_bbb["K_stock"]
        dv = policy_opt  # reduction in v
        cat_info.v = (dK_rp20 / K_stock) * (1 - dv) * cat_info.v + (1 - dK_rp20 / K_stock) * cat_info.v

    # 10% or nonpoor people see their v reduced 30%
    elif policy_name == "vr":
        n = cat_info.n.unstack().nonpoor
        dv = .3  # reduction in v
        f = .05  # fractionof nat pop would get the reduction
        cat_info.v = cat_info.v.unstack().assign(nonpoor=lambda x: (x.nonpoor * (1 - dv * f / n))).stack().clip(lower=0)
        desc = "Reduce asset\nvulnerability\n(by 30%) of\nnonpoor people\n(5% of the population)"

    # 10% or poor people see their fA reduced 10%
    elif policy_name == "fap":
        n = 0.2
        dfa = .05  # reduction in fa

        hazard_ratios["n"] = hazard_ratios.assign(n=1).n.unstack().assign(poor=0.2, nonpoor=0.8).stack()
        fa_event = agg_to_event_level(hazard_ratios, "fa")

        hazard_ratios = hazard_ratios.drop("n", axis=1)

        hazard_ratios.fa = hazard_ratios.fa.unstack().assign(poor=lambda x: (x.poor - dfa * fa_event / n)).stack().clip(lower=0)
        desc = "Reduce exposure\nof the poor by 5%\nof total exposure"



    # 10% or NONpoor people see their fA reduced 10%
    elif policy_name == "far":
        n = 0.8
        dfa = .05  # fractionof nat pop would get the reduction

        hazard_ratios["n"] = hazard_ratios.assign(n=1).n.unstack().assign(poor=0.2, nonpoor=0.8).stack()
        fa_event = agg_to_event_level(hazard_ratios, "fa")

        hazard_ratios.fa = hazard_ratios.fa.unstack().assign(nonpoor=lambda x: (x.nonpoor - dfa * fa_event / n)).stack().clip(lower=0)
        desc = "Reduce\nexposure of the nonpoor by 5%\nof total exposure"

    # Exposure equals to nonpoor exposure (mind that exposure in cat_info is OVERWRITTEN by exposure in hazard_ratios)
    elif policy_name == "fapfar":
        hazard_ratios.fa = hazard_ratios.fa.unstack().assign(poor=lambda x: x.nonpoor).stack()
        desc = "Make poor people's exposure the same as the rest of population"

    elif policy_name == "prop_nonpoor":
        # a.update(optionPDS = "prop_nonpoor", optionT="perfect",optionB="unlimited",optionFee="insurance_premium", share_insured=.25)
        desc = "Develop market\ninsurance\n(nonpoor people)"

    elif policy_name == "prop_nonpoor_lms":
        # a.update(optionPDS = "prop_nonpoor_lms", optionT="prop_nonpoor_lms",optionB="unlimited",optionFee="insurance_premium", share_insured=.5)
        desc = "Develop market insurance (25% of population, only nonpoor)"


    elif policy_name == "PDSpackage":
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "borrowing_abibility")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "prepare_scaleup")

        desc = "Postdisaster\nsupport\npackage"

    elif policy_name == "ResiliencePackage":
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "PDSpackage")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "axfin")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "T_rebuild_K")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "social_p")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "prop_nonpoor_lms")
        desc = "Resilience package"


    elif policy_name == "ResiliencePlusEW":
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "ResiliencePackage")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "shew")

        desc = "Resilience Package + Early Warning Package"

    elif policy_name == "Asset_losses":
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "fap")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "far")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "vp")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "vr")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "shew")

        desc = "Asset losses package "

    elif policy_name == "Asset_losses_no_EW":
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "fap")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "far")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "vp")
        macro, cat_info, hazard_ratios, desc = apply_policy(macro, cat_info, hazard_ratios, event_level, "vr")
        desc = "Asset losses package (excluding early warnings)"

    elif policy_name == "":
        pass

    if verbose:
        print(desc + "\n")

    return macro, cat_info, hazard_ratios, desc


def clip_social_p(m, c, fun=lambda x: x.clip(lower=.50)):
    """Compute social p from cat_info (c) and macro (m).
    Then changes social p to something applying fun
        for instance fun=lambda x:.3333 will set social_p at .3333
        fun = lambda x: x.clip(lower=.50) gets social p at least at 50%
    """

    m = m.copy(deep=True)
    c = c.copy(deep=True)

    #############
    #### preparation

    # current conso and share of average transfer
    cp = c.c.loc[:, "poor"]
    gsp = c.gamma_SP.loc[:, "poor"]

    # social_p
    cur_soc_sh = gsp * m.gdp_pc_pp * m.tau_tax / cp

    # social_p needs to be at least 50%
    obj_soc_sh = fun(cur_soc_sh)
    # obj_soc_sh.sample(10)

    # increase of transfer for poor people
    money_needed = (obj_soc_sh * (1 - cur_soc_sh) / (1 - obj_soc_sh) - cur_soc_sh) * cp
    money_needed.head()

    # financing this transfer with a tax on the whole population
    nu_tx = (m.gdp_pc_pp * m.tau_tax + money_needed * c.n.loc[:, "poor"]) / m.gdp_pc_pp
    nu_tx.head()

    # increasing the share of avg transfer that goes to poor
    nu_gsp = (money_needed + cur_soc_sh * cp) / (m.gdp_pc_pp * nu_tx)

    # balancing with less for nonpoor
    nu_gs = (1 - 0.2 * nu_gsp) / 0.8

    # double checking that that the objectife of 50% at least has been met
    # new_soc_sh = nu_gsp* m.gdp_pc_pp  *nu_tx /(cp+money_needed)
    # ((new_soc_sh-obj_soc_sh)**2).sum()

    # updating c and m
    c.gamma_SP = concat_categories(nu_gsp, nu_gs, index=income_cats)
    m.tau_tax = nu_tx

    return m, c
