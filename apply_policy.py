import pandas as pd


def apply_policy(macro_, cat_info_, hazard_ratios_, policy_name=None, policy_opt=None):
    """
    Chooses a policy by name, applies it to macro, cat_info, and/or hazard_ratios, and returns new values as well
    as a policy description
    """

    # duplicate inputes
    macro = macro_.copy(deep=True)
    cat_info = cat_info_.copy(deep=True)
    hazard_ratios = hazard_ratios_.copy(deep=True)
    desc = ''

    if policy_name in [None, 'baseline']:
        desc = "Baseline"
        policy_name = 'baseline'
    elif policy_name == 'reduce_poor_exposure':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Reduce poor exposure s.th. country exposure decreases by {policy_opt * 100}%"
        hazard_ratios = pd.merge(hazard_ratios, cat_info.n, left_index=True, right_index=True)
        hazard_ratios = hazard_ratios.reorder_levels(hazard_ratios_.index.names)
        total_exposure = (hazard_ratios[['fa', 'n']].product(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() /
                          hazard_ratios.n.groupby(['iso3', 'hazard', 'rp']).sum())
        poor_sel = hazard_ratios.index.get_level_values('income_cat').isin(['q1'])
        fa_poor_new = hazard_ratios.loc[poor_sel, 'fa'] - policy_opt * total_exposure / hazard_ratios.loc[poor_sel, 'n']
        fa_poor_new = fa_poor_new.fillna(0).clip(lower=0)
        hazard_ratios.fa.update(fa_poor_new)
        policy_name = 'reduce_poor_exposure_{}'.format(policy_opt)
    elif policy_name == 'reduce_rich_exposure':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Reduce rich exposure s.th. country exposure decreases by {policy_opt * 100}%"
        hazard_ratios = pd.merge(hazard_ratios, cat_info.n, left_index=True, right_index=True)
        hazard_ratios = hazard_ratios.reorder_levels(hazard_ratios_.index.names)
        total_exposure = (hazard_ratios[['fa', 'n']].product(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() /
                          hazard_ratios.n.groupby(['iso3', 'hazard', 'rp']).sum())
        rich_sel = hazard_ratios.index.get_level_values('income_cat').isin(['q5'])
        fa_rich_new = hazard_ratios.loc[rich_sel, 'fa'] - policy_opt * total_exposure / hazard_ratios.loc[rich_sel, 'n']
        fa_rich_new = fa_rich_new.fillna(0).clip(lower=0)
        hazard_ratios.fa.update(fa_rich_new)
        policy_name = 'reduce_rich_exposure_{}'.format(policy_opt)
    elif policy_name == 'reduce_nonpoor_exposure':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Reduce non-poor exposure s.th. country exposure decreases by {policy_opt * 100}%"
        hazard_ratios = pd.merge(hazard_ratios, cat_info.n, left_index=True, right_index=True)
        hazard_ratios = hazard_ratios.reorder_levels(hazard_ratios_.index.names)
        total_exposure = (hazard_ratios[['fa', 'n']].product(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() /
                          hazard_ratios.n.groupby(['iso3', 'hazard', 'rp']).sum())
        nonpoor_sel = hazard_ratios.index.get_level_values('income_cat').isin(['q2', 'q3', 'q4', 'q5'])
        fa_nonpoor_new = hazard_ratios.loc[nonpoor_sel, 'fa'] - 1 / 4 * policy_opt * total_exposure / hazard_ratios.loc[nonpoor_sel, 'n']
        fa_nonpoor_new = fa_nonpoor_new.fillna(0).clip(lower=0)
        hazard_ratios.fa.update(fa_nonpoor_new)
        policy_name = 'reduce_nonpoor_exposure_{}'.format(policy_opt)
    elif policy_name == 'reduce_total_exposure':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Uniformly reduce exposure by {policy_opt * 100}%"
        hazard_ratios.fa = hazard_ratios.fa * (1 - policy_opt)
        policy_name = 'reduce_total_exposure_{}'.format(policy_opt)
    elif policy_name == 'reduce_poor_vulnerability':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Reduce poor vulnerability s.th. country vulnerability decreases by {policy_opt * 100}%"
        hazard_ratios = pd.merge(hazard_ratios, cat_info.n, left_index=True, right_index=True)
        hazard_ratios = hazard_ratios.reorder_levels(hazard_ratios_.index.names)
        total_vulnerability = (hazard_ratios[['v', 'n']].product(axis=1).groupby(['iso3', 'hazard', 'rp']).sum() /
                          hazard_ratios.n.groupby(['iso3', 'hazard', 'rp']).sum())
        poor_sel = hazard_ratios.index.get_level_values('income_cat').isin(['q1'])
        v_poor_new = hazard_ratios.loc[poor_sel, 'v'] - policy_opt * total_vulnerability / hazard_ratios.loc[poor_sel, 'n']
        v_poor_new = v_poor_new.fillna(0).clip(lower=0)
        hazard_ratios.v.update(v_poor_new)
        policy_name = 'reduce_poor_vulnerability_{}'.format(policy_opt)
    elif policy_name == 'reduce_total_vulnerability':
        if policy_opt is None:
            policy_opt = 0.05
        desc = f"Uniformly reduce vulnerability by {policy_opt * 100}%"
        hazard_ratios.v = hazard_ratios.v * (1 - policy_opt)
        policy_name = 'reduce_total_vulnerability_{}'.format(policy_opt)
    elif policy_name == 'no_liquidity':
        desc = "No liquidity"
        cat_info['liquidity'] = 0
    elif policy_name == 'reduce_ew':
        desc = f"Reduce early warning to {policy_opt * 100}%"
        macro.ew = macro.ew * policy_opt
        hazard_ratios.ew = hazard_ratios.ew * policy_opt
        policy_name = 'reduce_ew_{}'.format(policy_opt)
    elif policy_name == 'increase_ew_to_max':
        desc = "Increase early warning to maximum"
        macro.ew = macro.ew.max()
        hazard_ratios.ew = (hazard_ratios.ew / hazard_ratios.ew * hazard_ratios.ew.max()).fillna(0)
    elif policy_name == 'set_ew':
        desc = "Set early warning to {}".format(policy_opt)
        macro.ew = policy_opt
        hazard_ratios.ew = (hazard_ratios.ew / hazard_ratios.ew * policy_opt).fillna(0)
        policy_name = 'set_ew_{}'.format(policy_opt)

    return macro, cat_info, hazard_ratios, policy_name, desc
