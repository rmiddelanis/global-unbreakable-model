import pandas as pd

from lib_compute_resilience_and_risk import agg_to_event_level


def apply_policy(macro_, cat_info_, hazard_ratios_, policy_name=None, policy_opt=None, verbose=True):
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

    return macro, cat_info, hazard_ratios, policy_name, desc
