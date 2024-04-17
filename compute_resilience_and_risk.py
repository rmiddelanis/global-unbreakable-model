from lib_compute_resilience_and_risk import *
import os
import warnings
import pandas as pd

warnings.filterwarnings("always", category=UserWarning)

if __name__ == '__main__':
    # define directory
    model = os.getcwd()  # get current directory
    input_dir = model + '/inputs/'  # get inputs data directory
    intermediate_dir = model + '/intermediate/'  # get outputs data directory

    scenarios = os.listdir(os.path.join(intermediate_dir, 'scenarios'))
    scenarios = [s for s in scenarios if os.path.isdir(os.path.join(intermediate_dir, 'scenarios', s))]

    # TODO: remove this later; only for testing purposes
    scenarios = ['baseline']

    for scenario in scenarios:
        print(scenario)
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
        macro = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__macro.csv"), index_col=econ_scope)

        # consumption, access to finance, gamma, capital, exposure, early warning access by country and income category
        cat_info = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__cat_info.csv"),
                               index_col=[econ_scope, "income_cat"])

        # exposure, vulnerability, and access to early warning by country, hazard, return period, income category
        hazard_ratios = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_ratios.csv"),
                                    index_col=event_level + ["income_cat"])

        hazard_protection = pd.read_csv(os.path.join(intermediate_dir, 'scenarios', scenario, "scenario__hazard_protection.csv"),
                                        index_col=[econ_scope, "hazard"])

        # compute
        # reshape macro and cat_info to event level, move hazard_ratios data to cat_info_event
        macro_event, cat_info_event = reshape_input(
            macro=macro,
            cat_info=cat_info,
            hazard_ratios=hazard_ratios,
            event_level=event_level,
        )

        # TODO: remove this later; only for testing purposes
        macro_event = macro_event.loc[['ISL']]
        cat_info_event = cat_info_event.loc[['ISL']]
        cat_info_event['liquidity'] = 0

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

        cat_info_event_iah = compute_dw_new(
            cat_info_event_iah=cat_info_event_iah,
            macro_event=macro_event,
            event_level_=event_level,
            capital_t=20,
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

        if not os.path.exists(f'output/scenarios/{scenario}'):
            os.makedirs(f'output/scenarios/{scenario}')

        # save macro_event
        macro_event.to_csv(f'output/scenarios/{scenario}/macro_' + option_fee + '_' + option_pds + '.csv', encoding="utf-8",
                           header=True)

        # save cat_info_event_iah
        cat_info_event_iah.to_csv(f'output/scenarios/{scenario}/iah_' + option_fee + '_' + option_pds + '.csv', encoding="utf-8",
                                  header=True)

        # Save results
        results.to_csv(f'output/scenarios/{scenario}/results_' + option_fee + '_' + option_pds + '.csv', encoding="utf-8",
                       header=True)
