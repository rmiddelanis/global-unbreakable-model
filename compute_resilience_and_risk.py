from lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd

warnings.filterwarnings("always", category=UserWarning)

# define directory
use_published_inputs = False

model = os.getcwd()  # get current directory
input_dir = model + '/inputs/'  # get inputs data directory
intermediate_dir = model + '/intermediate/'  # get outputs data directory

if use_published_inputs:
    input_dir = model + '/orig_inputs/'  # get inputs data directory
    intermediate_dir = model + '/orig_intermediate/'  # get outputs data directory

results_policy_summary = pd.DataFrame(index=pd.read_csv(intermediate_dir + "macro.csv", index_col='country').dropna().index)
# for pol_str in ['', '_bbb_complete1', '_bbb_incl1', '_bbb_fast1', '_bbb_fast2', '_bbb_fast4', '_bbb_fast5',
#                 '_bbb_50yrstand1']:
for pol_str in ['']:

    print(pol_str)
    option_fee = "tax"
    option_pds = "unif_poor"

    if option_fee == "insurance_premium":
        option_b = 'unlimited'
        option_t = 'perfect'
    else:
        option_b = 'data'
        option_t = 'data'

    print('optionFee =', option_fee, 'optionPDS =', option_pds, 'optionB =', option_b, 'optionT =', option_t)

    # Options and parameters
    econ_scope = "country"  # province, deparmtent
    event_level = [econ_scope, "hazard", "rp"]  # levels of index at which one event happens
    default_rp = "default_rp"  # return period to use when no rp is provided (mind that this works with protection)
    affected_cats = pd.Index(["a", "na"], name="affected_cat")  # categories for social protection
    helped_cats = pd.Index(["helped", "not_helped"], name="helped_cat")

    # read data

    # macro-economic country economic data
    macro = pd.read_csv(os.path.join(intermediate_dir, 'macro' + pol_str + ".csv"), index_col=econ_scope).dropna()

    # consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
    cat_info = pd.read_csv(os.path.join(intermediate_dir, 'cat_info' + pol_str + ".csv"),
                           index_col=[econ_scope, "income_cat"]).dropna()

    # exposure, vulnerability, and access to early warning by country, hazard, return period, income category
    hazard_ratios = pd.read_csv(os.path.join(intermediate_dir, 'hazard_ratios' + pol_str + ".csv"),
                                index_col=event_level + ["income_cat"]).dropna()

    # compute
    # (re)compute some of the data
    # replace common columns in macro_event and cats_event with those in hazard_ratios_event
    macro_event, cats_event, hazard_ratios_event, macro = process_input(
        macro=macro,
        cat_info=cat_info,
        hazard_ratios=hazard_ratios,
        econ_scope=econ_scope,
        event_level=event_level,
        default_rp=default_rp,
        verbose_replace=True
    )

    # calculate the actual vulnerability, the potential damage to capital, and consumption
    macro_event, cats_event_ia = compute_dK(
        macro_event=macro_event,
        cats_event=cats_event,
        event_level=event_level,
        affected_cats=affected_cats
    )

    # calculate the post-disaster response
    macro_event, cats_event_iah = calculate_response(
        macro_event=macro_event, 
        cats_event_ia=cats_event_ia,
        event_level=event_level, 
        helped_cats=helped_cats,
        option_fee=option_fee, 
        option_t=option_t, 
        option_pds=option_pds,
        option_b=option_b, 
        loss_measure="dk", 
        fraction_inside=1,
        share_insured=.25
    )

    # Save output
    macro_event.to_csv('output/macro_' + option_fee + '_' + option_pds + '_' + pol_str + '.csv', encoding="utf-8",
                       header=True)
    cats_event_iah.to_csv('output/cats_event_iah_' + option_fee + '_' + option_pds + '_' + pol_str + '.csv',
                          encoding="utf-8", header=True)

    # compute welfare losses and aggregate to event-level (ie no more income_cat, helped_cat, affected_cat, n)
    out, iah = compute_dW(
        macro_event=macro_event,
        cats_event_iah=cats_event_iah,
        event_level=event_level,
        return_stats=True,
        return_iah=True
    )

    # Computes
    # results
    results = process_output(
        macro=macro,
        out=out,
        macro_event=macro_event,
        econ_scope=econ_scope,
        default_rp=default_rp,
        is_local_welfare=True
    )

    # Save output
    results.to_csv('output/results_' + option_fee + '_' + option_pds + '_' + pol_str + '.csv', encoding="utf-8",
                   header=True)
    iah.to_csv('output/iah_' + option_fee + '_' + option_pds + '_' + pol_str + '.csv', encoding="utf-8", header=True)

    results_policy_summary[pol_str + '_dw_tot_curr'] = results['dWtot_currency']
results_policy_summary.to_csv('output/results_policy_summary.csv')
